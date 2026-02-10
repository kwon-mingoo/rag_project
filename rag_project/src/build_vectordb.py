import os
import glob
import json
import argparse
import pickle
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np
import pandas as pd

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_community.llms import LlamaCpp

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    import faiss
except ImportError as e:
    raise ImportError("FAISS가 필요합니다. (pip faiss-cpu) 또는 (conda faiss-gpu)") from e



# -----------------------------
# System/User Prompts (운영형 템플릿)
# -----------------------------
SYSTEM_PROMPT = """당신은 건설 현장 RAG 기반 ‘법령/기준 근거 제시’ 어시스턴트입니다.

원칙:
- 제공된 [컨텍스트]에 있는 내용만 근거로 답변합니다.
- 컨텍스트에 없는 법령/조항/수치/요구사항은 추측하지 않습니다.
- 모든 요구사항/권고사항은 반드시 출처([1],[2]...)를 붙입니다.
- 근거가 부족하면 '근거 부족'으로 표시하고 추가 확인이 필요한 정보를 질문합니다.

출력 형식은 반드시 지정된 템플릿을 따릅니다.
"""

def build_user_prompt(question: str, context_blocks: str) -> str:
    return f"""[질의]
{question}

[컨텍스트]
{context_blocks}

[출력 템플릿]
## 1) 결론(한 줄)
- (핵심 요구사항 1문장) [출처]

## 2) 적용해야 할 요구사항(필수)
- 요구사항 A: … [출처]
- 요구사항 B: … [출처]

## 3) 근거(법령/기준/가이드)
- 근거 1: 문서명 p.페이지 (요약) [출처]
- 근거 2: 문서명 p.페이지 (요약) [출처]

## 4) 현장 실행 체크리스트(권고)
- [ ] 작업 전: … [출처]
- [ ] 작업 중: … [출처]
- [ ] 작업 후: … [출처]
- [ ] 문서/기록: … [출처]

## 5) 계획변경/민원 대응 시 리스크와 권고
- 리스크: … [출처 또는 근거부족]
- 권고: … [출처 또는 근거부족]

## 6) 근거 부족/추가 확인 질문
- (컨텍스트만으로 판단 불가한 항목을 질문 형태로 3개 이내)
"""

# -----------------------------
# Tokenizer (BM25용)
# -----------------------------
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣_]+")

def tokenize_ko_en(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(str(text).lower())


def minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if abs(mx - mn) < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalConfig:
    vectordb_dir: str = "VectorDB"
    eval_data_dir: str = "/home/nami/Desktop/rag_project_v1/eval_data/eval_dataset_rag_queries.csv"
    out_report: str = "eval_data/eval_report.json"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    gguf_path: str = "/home/nami/Desktop/rag_project_v1/models/llama8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"

    # Hybrid retrieval
    alpha: float = 0.55
    top_k: int = 6
    fetch_k: int = 40

    # LLM generation + judging
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.2
    max_tokens: int = 512

    # Judge settings
    judge_max_tokens: int = 512
    judge_temperature: float = 0.0

    # BLEU settings
    bleu_max_ngram: int = 4  # BLEU-4

    # Semantic similarity score mapping
    # cosine(-1..1) -> (0..1) via (cos+1)/2 then clamp
    semantic_map_to_01: bool = True


# -----------------------------
# Load VectorDB
# -----------------------------
def load_vectordb(vectordb_dir: str):
    faiss_path = os.path.join(vectordb_dir, "faiss.index")
    docs_path = os.path.join(vectordb_dir, "docs.pkl")
    bm25_path = os.path.join(vectordb_dir, "bm25.pkl")

    if not (os.path.exists(faiss_path) and os.path.exists(docs_path) and os.path.exists(bm25_path)):
        raise FileNotFoundError("VectorDB가 완전하지 않습니다. 먼저 src/build_vectordb.py 실행하세요.")

    index = faiss.read_index(faiss_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm = pickle.load(f)
    bm25: BM25Okapi = bm["bm25"]
    return index, docs, bm25


# -----------------------------
# Hybrid Retriever
# -----------------------------
class HybridRetriever:
    def __init__(self, docs: List[Document], bm25: BM25Okapi, faiss_index, embed_model: SentenceTransformer):
        self.docs = docs
        self.bm25 = bm25
        self.index = faiss_index
        self.embed_model = embed_model

    def embed_query(self, q: str) -> np.ndarray:
        v = self.embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
        faiss.normalize_L2(v)
        return v

    def retrieve(self, query: str, alpha: float, top_k: int, fetch_k: int) -> List[Tuple[Document, float, Dict[str, Any]]]:
        # BM25
        q_tokens = tokenize_ko_en(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)
        bm25_n = minmax_norm(bm25_scores)

        # FAISS
        qv = self.embed_query(query)
        sim, idx = self.index.search(qv, fetch_k)
        sim = sim[0].astype(np.float32)
        idx = idx[0].astype(np.int64)

        vec_scores = np.zeros(len(self.docs), dtype=np.float32)
        vec_scores[idx] = sim
        vec_n = minmax_norm(vec_scores)

        # Hybrid
        hybrid = alpha * vec_n + (1.0 - alpha) * bm25_n
        top_idx = np.argsort(-hybrid)[:top_k]

        results = []
        for rank_i, i in enumerate(top_idx, start=1):
            d = self.docs[i]
            dbg = {
                "rank": rank_i,
                "hybrid": float(hybrid[i]),
                "bm25": float(bm25_scores[i]),
                "vec": float(vec_scores[i]),
                "chunk_id": d.metadata.get("chunk_id"),
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "source_basename": os.path.basename(str(d.metadata.get("source", ""))),
            }
            results.append((d, float(hybrid[i]), dbg))
        return results


# -----------------------------
# LLM (generation + judge)
# -----------------------------
def build_llm(cfg: EvalConfig) -> LlamaCpp:
    if not os.path.exists(cfg.gguf_path):
        raise FileNotFoundError(f"GGUF 파일이 없습니다: {cfg.gguf_path}")

    return LlamaCpp(
        model_path=cfg.gguf_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        n_batch=512,
        temperature=cfg.temperature,
        top_p=0.9,
        repeat_penalty=1.12,
        max_tokens=cfg.max_tokens,
        # Llama 3.1 Instruct는 chat template 필수
        chat_format="llama-3",
        stop=["<|eot_id|>"],
        verbose=False,
    )


def build_judge(cfg: EvalConfig) -> LlamaCpp:
    return LlamaCpp(
        model_path=cfg.gguf_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        n_batch=512,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.05,
        max_tokens=512,
        chat_format="llama-3",
        stop=["<|eot_id|>"],
        verbose=False,
    )







def make_context_blocks(retrieved: List[Tuple[Document, float, Dict[str, Any]]]) -> str:
    blocks = []
    for i, (d, _, dbg) in enumerate(retrieved, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        where = f"{os.path.basename(str(src))}" + (f" p.{page}" if page is not None else "")
        blocks.append(f"[{i}] ({where}, chunk_id={d.metadata.get('chunk_id')})\n{d.page_content}")
    return "\n\n".join(blocks)


def gen_prompt(question: str, context_blocks: str) -> str:
    return f"""당신은 문서 기반 QA 어시스턴트입니다.
아래 '컨텍스트'에 있는 내용만 근거로 답변하세요.
정답은 한 문장(또는 한 줄)로만 짧게 답하라.
컨텍스트에 없는 내용은 추측하지 말고 '문서에 근거가 부족합니다'라고 말하세요.
가능하면 핵심 근거 문장을 짧게 인용하고, 출처 조각 번호([1],[2]...)를 함께 표시하세요.

질문: {question}

컨텍스트:
{context_blocks}

답변:"""


# -----------------------------
# Metrics: BLEU, Semantic Similarity
# -----------------------------
def compute_bleu(reference: str, hypothesis: str, max_ngram: int = 4) -> float:
    ref_tokens = tokenize_ko_en(reference)
    hyp_tokens = tokenize_ko_en(hypothesis)
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    weights = tuple([1.0 / max_ngram] * max_ngram)
    smoothie = SmoothingFunction().method4
    # BLEU is in [0, 1]
    return float(sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothie))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def semantic_similarity(embed_model: SentenceTransformer, a: str, b: str, map_to_01: bool = True) -> float:
    va = embed_model.encode([a], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    vb = embed_model.encode([b], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    cos = cosine_sim(va, vb)  # [-1, 1]
    if map_to_01:
        score = (cos + 1.0) / 2.0
        return float(max(0.0, min(1.0, score)))
    return float(cos)


# -----------------------------
# LLM Judging -> always score 0..1
# -----------------------------
def judge_json(judge_llm: LlamaCpp, prompt: str) -> Dict[str, Any]:
    try:
        raw = judge_llm.invoke(prompt)
    except Exception as e:
        # invoke 자체가 실패한 경우에도 점수는 0으로 반환
        return {"ok": False, "score": 0.0, "error": str(e)}

    # ✅ 디버그 출력은 raw가 생긴 뒤에만
    
    # JSON만 추출
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {"ok": False, "score": 0.0, "raw": raw[:2000]}

    try:
        obj = json.loads(m.group(0))
        if "score" not in obj:
            obj["score"] = 0.0
        obj["ok"] = True
        # clamp
        obj["score"] = float(max(0.0, min(1.0, float(obj.get("score", 0.0)))))
        return obj
    except Exception as e:
        return {"ok": False, "score": 0.0, "raw": raw[:2000], "error": str(e)}



def faithfulness_prompt(question: str, answer: str, context_blocks: str) -> str:
    return f"""역할: 당신은 RAG 평가자입니다.
목표: '답변'이 '컨텍스트'에 의해 충분히 뒷받침되는지(환각 여부) 평가합니다.

규칙:
- 컨텍스트에 명시/강하게 함의된 내용만 'supported'로 봅니다.
- 컨텍스트에 없는 새로운 사실/수치/고유명사/규정 조항을 단정하면 'unsupported'입니다.
- 출력은 반드시 JSON 하나만.

입력:
[질문]
{question}

[답변]
{answer}

[컨텍스트]
{context_blocks}

JSON 출력 스키마:
{{
  "supported": true/false,
  "score": 0~1,
  "unsupported_claims": ["...","..."],
  "notes": "짧은 근거"
}}
"""


def truthfulness_prompt(question: str, answer: str, ground_truth: str) -> str:
    return f"""역할: 당신은 QA 평가자입니다.
목표: '답변'이 '정답(GT)'의 핵심 사실관계를 정확히 포함하는지 평가합니다.

규칙:
- 정답의 핵심 사실(주체/행위/조건/수치/기준)이 누락되거나 틀리면 감점.
- 표현/문장 형태가 달라도 의미가 같으면 OK.
- 출력은 반드시 JSON 하나만.

입력:
[질문]
{question}

[정답(GT)]
{ground_truth}

[답변]
{answer}

JSON 출력 스키마:
{{
  "correct": true/false,
  "score": 0~1,
  "missing_facts": ["..."],
  "wrong_facts": ["..."],
  "notes": "짧은 근거"
}}
"""


def context_recall_prompt(question: str, ground_truth: str, context_blocks: str) -> str:
    return f"""역할: 당신은 RAG 평가자입니다.
목표: 제공된 '컨텍스트'만으로 '정답(GT)'에 도달 가능했는지(답변 가능성) 평가합니다.

규칙:
- 컨텍스트에 정답을 직접 말하지 않아도, 합리적으로 유추 가능한 근거가 있으면 true.
- 컨텍스트가 부족하면 false.
- 출력은 반드시 JSON 하나만.

입력:
[질문]
{question}

[정답(GT)]
{ground_truth}

[컨텍스트]
{context_blocks}

JSON 출력 스키마:
{{
  "answerable": true/false,
  "score": 0~1,
  "evidence": ["컨텍스트 근거 요약..."],
  "notes": "짧은 근거"
}}
"""


# -----------------------------
# Retrieval Metrics -> always numeric score
# - CSV: source_file + page or relevant_docs + page
# - doc.metadata["source"] may be path; compare by basename
# -----------------------------
def normalize_filename(x: str) -> str:
    return os.path.basename(str(x)).strip()


def get_relevant_sources_from_row(row: pd.Series) -> Set[Tuple[str, int]]:
    # Try common columns:
    # 1) relevant_sources like "file.pdf:12;file2.pdf:3"
    # Prefer 0-index labels if present
    for col in ["relevant_sources_0idx", "relevant_sources", "relevant_sources_1idx"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            s = str(row[col]).strip()
            out = set()
            for part in re.split(r"[;,	]+", s):
                part = part.strip()
                if not part:
                    continue
                m = re.match(r"(.+):(\d+)$", part)
                if m:
                    out.add((normalize_filename(m.group(1)), int(m.group(2))))
            if out:
                return out

    # 2) source_file + page
    src_col = None
    for c in ["source_file", "relevant_docs", "relevant_doc", "doc", "source"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            src_col = c
            break

    page = None
    for pc in ["page", "pages", "page_no"]:
        if pc in row.index and pd.notna(row[pc]):
            try:
                # CSV page is typically 1-index; convert to 0-index for docs.pkl
                page = int(row[pc]) - 1
                break
            except Exception:
                pass

    if src_col is not None and page is not None:
        return {(normalize_filename(row[src_col]), int(page))}
    return set()


def get_relevant_chunk_ids_from_row(row: pd.Series) -> Set[int]:
    # optional: relevant_chunk_ids like "12;98"
    if "relevant_chunk_ids" not in row.index or pd.isna(row["relevant_chunk_ids"]):
        return set()
    s = str(row["relevant_chunk_ids"]).strip()
    if not s:
        return set()
    parts = re.split(r"[;, \t]+", s)
    out = set()
    for p in parts:
        if p.strip().isdigit():
            out.add(int(p.strip()))
    return out


def derive_relevant_chunk_ids_from_excerpt(
    row: pd.Series,
    docs: List[Document],
    page_to_chunkids: Dict[Tuple[str, int], List[int]],
) -> Set[int]:
    """Derive ground-truth chunk_ids from (source_file, page, evidence_excerpt).
    This makes Recall@k/MRR harder and more meaningful than page-level labels.
    """
    needed = ["source_file", "page", "evidence_excerpt"]
    for c in needed:
        if c not in row.index or pd.isna(row[c]) or not str(row[c]).strip():
            return set()

    src = os.path.basename(str(row["source_file"]).strip())
    try:
        # CSV page is 1-index; docs are 0-index
        page0 = int(row["page"]) - 1
    except Exception:
        return set()

    excerpt = str(row["evidence_excerpt"]).strip()
    if not excerpt:
        return set()

    cids = page_to_chunkids.get((src, page0), [])
    if not cids:
        return set()

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    ex_n = norm(excerpt)
    # Use prefix for robust matching (OCR/spacing differences)
    ex_prefix = ex_n[:120]

    hit: Set[int] = set()
    cid_set = set(cids)
    for d in docs:
        cid = d.metadata.get("chunk_id", None)
        if cid is None or int(cid) not in cid_set:
            continue
        if ex_prefix and ex_prefix in norm(d.page_content):
            hit.add(int(cid))

    # Fallback: if no match, pick the first chunk on that page
    if not hit and cids:
        hit.add(int(cids[0]))

    return hit


def recall_at_k(relevant_chunk_ids: Set[int], relevant_sources: Set[Tuple[str, int]], retrieved_docs: List[Document]) -> float:
    # If no labels, score is 0.0 (always numeric per requirement)
    if not relevant_chunk_ids and not relevant_sources:
        return 0.0

    hit = 0
    total = 0

    if relevant_chunk_ids:
        total += len(relevant_chunk_ids)
        got = {d.metadata.get("chunk_id") for d in retrieved_docs}
        hit += len(relevant_chunk_ids.intersection({x for x in got if x is not None}))

    if relevant_sources:
        total += len(relevant_sources)
        got2 = {(normalize_filename(d.metadata.get("source", "")), int(d.metadata.get("page")))
                for d in retrieved_docs
                if d.metadata.get("source") is not None and d.metadata.get("page") is not None}
        hit += len(relevant_sources.intersection(got2))

    return float(hit / total) if total > 0 else 0.0


def mrr(relevant_chunk_ids: Set[int], relevant_sources: Set[Tuple[str, int]], retrieved_docs: List[Document]) -> float:
    # If no labels, score is 0.0 (always numeric per requirement)
    if not relevant_chunk_ids and not relevant_sources:
        return 0.0

    for rank_i, d in enumerate(retrieved_docs, start=1):
        cid = d.metadata.get("chunk_id")
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        if (cid is not None and cid in relevant_chunk_ids) or (
            src is not None and page is not None and (normalize_filename(src), int(page)) in relevant_sources
        ):
            return float(1.0 / rank_i)
    return 0.0


# -----------------------------
# Eval data loader
# -----------------------------
def find_single_csv(eval_dir_or_file: str) -> str:
    # 1) 파일 경로가 직접 들어온 경우
    if os.path.isfile(eval_dir_or_file):
        if not eval_dir_or_file.lower().endswith(".csv"):
            raise ValueError("평가 파일은 .csv 여야 합니다.")
        return eval_dir_or_file

    # 2) 폴더 경로인 경우
    if not os.path.isdir(eval_dir_or_file):
        raise FileNotFoundError(f"eval_data 경로가 없습니다: {eval_dir_or_file}")

    csvs = glob.glob(os.path.join(eval_dir_or_file, "*.csv"))
    if len(csvs) != 1:
        raise FileNotFoundError(
            f"{eval_dir_or_file} 안에 CSV가 정확히 1개 있어야 합니다. 현재: {len(csvs)}개"
        )
    return csvs[0]



def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    # question column
    q_candidates = ["question", "query", "q"]
    gt_candidates = ["answer", "ground_truth", "gt", "gold"]
    q_col = next((c for c in q_candidates if c in df.columns), None)
    gt_col = next((c for c in gt_candidates if c in df.columns), None)
    if q_col is None or gt_col is None:
        raise ValueError(
            f"CSV에는 질문/정답 컬럼이 필요합니다. "
            f"질문 후보={q_candidates}, 정답 후보={gt_candidates}. 현재 컬럼={list(df.columns)}"
        )
    return q_col, gt_col



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Hybrid RAG (BM25+FAISS) with retrieval/debug options.")
    p.add_argument("--eval_path", type=str, default=None,
                   help="평가 CSV 파일 경로 또는 평가 CSV가 1개 들어있는 폴더 경로. (기본: cfg.eval_data_dir)")
    p.add_argument("--debug_n", type=int, default=0,
                   help="처음 N개 샘플에 대해 retrieval 디버그(Top-k, GT 매칭, 텍스트 일부) 출력 및 JSONL 저장")
    p.add_argument("--retrieval_only", action="store_true",
                   help="Retrieval 지표(Recall@k, MRR)만 계산. LLM 생성/판정 및 BLEU/SemSim 스킵")
    p.add_argument("--save_jsonl", type=str, default=None,
                   help="디버그/분석용 JSONL 저장 경로. (기본: eval_data/retrieval_debug.jsonl)")
    # Optional overrides
    p.add_argument("--alpha", type=float, default=None, help="Hybrid alpha override")
    p.add_argument("--top_k", type=int, default=None, help="Top-k override")
    p.add_argument("--fetch_k", type=int, default=None, help="Fetch-k override")
    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
def main(args: argparse.Namespace | None = None):
    cfg = EvalConfig()

    if args is None:
        args = parse_args()


    
    # CLI overrides
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.fetch_k is not None:
        cfg.fetch_k = int(args.fetch_k)

    # NLTK 준비(이미 설치되어 있으면 스킵됨)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # Load VectorDB
    index, docs, bm25 = load_vectordb(cfg.vectordb_dir)

    # Models
    embed_model = SentenceTransformer(cfg.embedding_model)
    llm = None
    judge = None
    if not args.retrieval_only:
        llm = build_llm(cfg)
        judge = build_judge(cfg)

    retriever = HybridRetriever(docs, bm25, index, embed_model)

    # Build (source_basename, page) -> [chunk_id,...] map for chunk-level GT derivation
    page_to_chunkids: Dict[Tuple[str, int], List[int]] = {}
    for d in docs:
        src = os.path.basename(str(d.metadata.get("source", "")))
        page = d.metadata.get("page", None)
        cid = d.metadata.get("chunk_id", None)
        if src and page is not None and cid is not None:
            page_to_chunkids.setdefault((src, int(page)), []).append(int(cid))

    # Load eval CSV
    csv_path = find_single_csv(args.eval_path or cfg.eval_data_dir)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin1")

    if args.save_jsonl:
        save_jsonl = args.save_jsonl
    else:
        base_dir = (cfg.eval_data_dir if os.path.isdir(cfg.eval_data_dir) else os.path.dirname(cfg.eval_data_dir))
        if args.eval_path:
            base_dir = (args.eval_path if os.path.isdir(args.eval_path) else os.path.dirname(args.eval_path))
        save_jsonl = os.path.join(base_dir or ".", "retrieval_debug.jsonl")

    q_col, gt_col = detect_columns(df)

    # Accumulators (all numeric)
    bleu_list: List[float] = []
    semsim_list: List[float] = []
    faithful_scores: List[float] = []
    truthful_scores: List[float] = []
    ctxrecall_scores: List[float] = []
    recallk_list: List[float] = []
    mrr_list: List[float] = []

    per_item: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        q = str(row[q_col])
        gt = str(row[gt_col])

        rel_sources = get_relevant_sources_from_row(row)

        # Derive chunk-level GT from evidence_excerpt when possible (more meaningful than page-level)
        derived_chunk_ids = derive_relevant_chunk_ids_from_excerpt(row, docs, page_to_chunkids)
        rel_chunk_ids = get_relevant_chunk_ids_from_row(row) | derived_chunk_ids

        # If we have chunk-level labels, focus retrieval metrics on chunk_ids (avoid overly-easy page hits)
        if derived_chunk_ids:
            rel_sources = set()

        # Retrieve
        retrieved = retriever.retrieve(q, cfg.alpha, cfg.top_k, cfg.fetch_k)
        retrieved_docs = [d for (d, _, _) in retrieved]
        context_blocks = make_context_blocks(retrieved)

        # Retrieval debug/logging (LLM output 없이도 검증 가능)
        def _hit_rank(rel_sources, rel_chunk_ids, retrieved_docs):
            for rnk, dd in enumerate(retrieved_docs, start=1):
                src = os.path.basename(str(dd.metadata.get("source", "")))
                page = dd.metadata.get("page", None)
                cid = dd.metadata.get("chunk_id", None)
                if cid is not None and int(cid) in rel_chunk_ids:
                    return rnk, "chunk"
                if src and page is not None and (src, int(page)) in rel_sources:
                    return rnk, "page"
            return None, None

        if args.debug_n and i < args.debug_n:
            print("\n" + "=" * 100)
            print(f"[DEBUG row {i}]")
            print("QUERY(first 250):", q[:250].replace("\n", " "))
            print("GT sources:", sorted(list(rel_sources))[:5])
            print("GT chunk_ids:", sorted(list(rel_chunk_ids))[:10])

            print("\n[TOP-K Retrieved]")
            for (dd, _sc, dbg) in retrieved:
                print(f" rank={dbg['rank']} src={dbg['source_basename']} page={dbg['page']} chunk={dbg['chunk_id']} hybrid={dbg['hybrid']:.3f} bm25={dbg['bm25']:.2f} vec={dbg['vec']:.3f}")
                print("  text[0:200]:", dd.page_content[:200].replace("\n", " "))

            rnk, kind = _hit_rank(rel_sources, rel_chunk_ids, retrieved_docs)
            print("\n[HIT CHECK] hit_rank:", rnk, "kind:", kind)

            # JSONL 저장
            os.makedirs(os.path.dirname(save_jsonl) or ".", exist_ok=True)
            with open(save_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps({
                    "row": int(i),
                    "query": q,
                    "ground_truth": gt,
                    "labels": {
                        "relevant_sources": sorted(list(rel_sources)),
                        "relevant_chunk_ids": sorted(list(rel_chunk_ids)),
                    },
                    "retrieval_topk": [
                        {
                            "rank": dbg["rank"],
                            "source": dbg["source_basename"],
                            "page": dbg["page"],
                            "chunk_id": dbg["chunk_id"],
                            "hybrid": dbg["hybrid"],
                            "bm25": dbg["bm25"],
                            "vec": dbg["vec"],
                            "text_preview": dd.page_content[:300],
                        }
                        for (dd, _s, dbg) in retrieved
                    ],
                }, ensure_ascii=False) + "\n")

        # Generate (optional)
        pred = ""
        bleu = 0.0
        semsim = 0.0
        faithful_score = 0.0
        truthful_score = 0.0
        ctxrec_score = 0.0

        if not args.retrieval_only:
            prompt = gen_prompt(q, context_blocks)
            if args.debug_n and i < args.debug_n:
                print("\n[PROMPT preview]")
                print(str(prompt)[:800])
            pred = llm.invoke(prompt)
            if args.debug_n and i < args.debug_n:
                print("\n[GENERATION OUTPUT preview]")
                print(str(pred)[:1200])

            # Generation metrics
            bleu = compute_bleu(gt, pred, max_ngram=cfg.bleu_max_ngram)  # 0..1
            semsim = semantic_similarity(embed_model, gt, pred, map_to_01=cfg.semantic_map_to_01)  # 0..1

            # LLM judge (0..1) - best effort
            faithful = judge_json(judge, faithfulness_prompt(q, pred, context_blocks))
            truthful = judge_json(judge, truthfulness_prompt(q, pred, gt))
            ctxrec = judge_json(judge, context_recall_prompt(q, gt, context_blocks))

            faithful_score = float(faithful.get("score", 0.0))
            truthful_score = float(truthful.get("score", 0.0))
            ctxrec_score = float(ctxrec.get("score", 0.0))
        # Retrieval metrics (0..1)
        r_at_k = recall_at_k(rel_chunk_ids, rel_sources, retrieved_docs)
        mrr_v = mrr(rel_chunk_ids, rel_sources, retrieved_docs)

        # Save accumulators
        bleu_list.append(bleu)
        semsim_list.append(semsim)
        faithful_scores.append(faithful_score)
        truthful_scores.append(truthful_score)
        ctxrecall_scores.append(ctxrec_score)
        recallk_list.append(r_at_k)
        mrr_list.append(mrr_v)

        per_item.append({
            "row": int(i),
            "question": q,
            "ground_truth": gt,
            "prediction": pred,

            "scores": {
                # Generation quality
                "bleu": bleu,
                "semantic_similarity": semsim,
                "faithfulness": faithful_score,
                "truthfulness": truthful_score,
                # Retrieval quality
                "context_recall": ctxrec_score,
                "recall@k": r_at_k,
                "mrr": mrr_v,
            },

            # Optional detailed judge outputs for debugging
            "judge_details": {
                "faithfulness": faithful,
                "truthfulness": truthful,
                "context_recall": ctxrec,
            },

            "retrieval_topk": [
                {
                    "rank": dbg["rank"],
                    "chunk_id": dbg["chunk_id"],
                    "source": dbg["source_basename"],
                    "page": dbg["page"],
                    "hybrid": dbg["hybrid"],
                    "bm25": dbg["bm25"],
                    "vec": dbg["vec"],
                }
                for (_, _, dbg) in retrieved
            ],

            "relevance_labels": {
                "relevant_chunk_ids": sorted(list(rel_chunk_ids)),
                "relevant_sources": sorted(list(rel_sources)),
            }
        })

        print(
            f"[{i+1}/{len(df)}] "
            f"BLEU={bleu:.3f} SemSim={semsim:.3f} "
            f"Faith={faithful_score:.3f} Truth={truthful_score:.3f} "
            f"CtxRecall={ctxrec_score:.3f} Recall@k={r_at_k:.3f} MRR={mrr_v:.3f}"
        )

    report = {
        "eval_csv": os.path.basename(csv_path),
        "count": int(len(df)),
        "config": {
            "alpha": cfg.alpha,
            "top_k": cfg.top_k,
            "fetch_k": cfg.fetch_k,
            "embedding_model": cfg.embedding_model,
            "gguf_path": cfg.gguf_path,
            "semantic_map_to_01": cfg.semantic_map_to_01,
        },
        "averages": {
            "bleu": float(np.mean(bleu_list)) if bleu_list else 0.0,
            "semantic_similarity": float(np.mean(semsim_list)) if semsim_list else 0.0,
            "faithfulness": float(np.mean(faithful_scores)) if faithful_scores else 0.0,
            "truthfulness": float(np.mean(truthful_scores)) if truthful_scores else 0.0,
            "context_recall": float(np.mean(ctxrecall_scores)) if ctxrecall_scores else 0.0,
            "recall@k": float(np.mean(recallk_list)) if recallk_list else 0.0,
            "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        },
        "items": per_item,
    }

    os.makedirs(os.path.dirname(cfg.out_report) or ".", exist_ok=True)
    with open(cfg.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ Eval done.")
    print(f"- input : {csv_path}")
    print(f"- report: {cfg.out_report}")


if __name__ == "__main__":
    main(parse_args())