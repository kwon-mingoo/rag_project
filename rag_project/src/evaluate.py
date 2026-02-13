#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_patched.py
- rag_engine.py를 사용해 운영/평가 동일 코어로 평가
- 장기 재현성 강화:
  1) GT를 chunk_id에 고정하지 않음 (source_basename+page(+excerpt) 기반, chunk는 파생)
  2) page 인덱싱 규칙을 스펙으로 고정 (CSV page는 기본 1-index)
  3) 라벨 파서 유연화 (컬럼명/포맷 변화에 강함)
  4) retrieval 구현/설정 변경을 report에 signature로 기록

지표:
(Generation) BLEU, Semantic Similarity, Faithfulness, Truthfulness
(Retrieval) Context Recall, Recall@k, MRR
"""

import os
import re
import json
import argparse
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer

import rag_engine as eng


# -----------------------------
# Eval Spec (재현성용)
# -----------------------------
@dataclass
class EvalSpec:
    # CSV page numbering: 대부분 1-index. (docs metadata는 0-index로 가정)
    csv_page_base: int = 1

    # GT 기본키: source_basename + page
    use_source_page_labels: bool = True

    # evidence_excerpt가 있으면, 해당 페이지의 chunk 중 excerpt 매칭으로 chunk_id를 파생해 더 엄밀한 평가를 수행
    use_excerpt_to_derive_chunk: bool = True

    # 파일명 매칭 정책: basename 기준
    match_source_by_basename: bool = True


# -----------------------------
# Eval Config
# -----------------------------
@dataclass
class EvalConfig:
    eval_csv: str = "eval_data/eval_dataset_rag_queries.csv"
    report_path: str = "eval_data/eval_report.json"
    save_jsonl: str = "eval_data/retrieval_debug.jsonl"

    rag: eng.RAGConfig = field(default_factory=eng.RAGConfig)
    spec: EvalSpec = field(default_factory=EvalSpec)

    max_rows: int = -1
    debug_n: int = 0
    retrieval_only: bool = False
    enable_judge: bool = True

    judge_max_tokens: int = 2048
    judge_temperature: float = 0.0

    bleu_max_ngram: int = 4
    semantic_map_to_01: bool = True


# -----------------------------
# Metrics
# -----------------------------
def compute_bleu(ref: str, hyp: str, max_ngram: int = 4) -> float:
    ref_tokens = eng.tokenize_ko_en(ref)
    hyp_tokens = eng.tokenize_ko_en(hyp)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    weights = tuple([1.0 / max_ngram] * max_ngram)
    smooth = SmoothingFunction().method4
    return float(sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smooth))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_similarity(embed_model: SentenceTransformer, ref: str, hyp: str, map_to_01: bool = True) -> float:
    v = embed_model.encode([ref, hyp], normalize_embeddings=True)
    cos = cosine_sim(np.asarray(v[0]), np.asarray(v[1]))
    if map_to_01:
        return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
    return float(cos)


# -----------------------------
# Robust CSV loading + column detection
# -----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    q_candidates = ["question", "query", "q"]
    gt_candidates = ["ground_truth", "answer", "gt", "gold"]
    q_col = next((c for c in q_candidates if c in df.columns), None)
    gt_col = next((c for c in gt_candidates if c in df.columns), None)
    if not q_col or not gt_col:
        raise ValueError(
            f"CSV에는 질문/정답 컬럼이 필요합니다. "
            f"q 후보={q_candidates}, gt 후보={gt_candidates}. 현재={list(df.columns)}"
        )
    return q_col, gt_col


def normalize_filename(s: Any) -> str:
    return os.path.basename(str(s)).strip()


def _parse_int(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(str(val).strip())
    except Exception:
        return None


def _split_parts(val: str) -> List[str]:
    return [
        p.strip()
        for p in re.split(r"\s*[|;,]\s*|\s*\n\s*|\s*\t\s*", str(val))
        if p.strip()
    ]



def extract_page_1based(row: pd.Series) -> Optional[int]:
    for c in ["page", "pages", "page_no", "page_num"]:
        if c in row.index:
            p = _parse_int(row.get(c))
            if p is not None:
                return p
    return None


def extract_excerpt(row: pd.Series) -> str:
    for c in ["evidence_excerpt", "excerpt", "evidence", "quote", "snippet"]:
        if c in row.index and not pd.isna(row.get(c)):
            s = str(row.get(c)).strip()
            if s:
                return s
    return ""


def parse_relevant_sources_pages(row: pd.Series, spec: EvalSpec) -> Set[Tuple[str, int]]:
    """Return set of (source_basename, page0). Page is required if possible."""
    out: Set[Tuple[str, int]] = set()

    # 1) combined formats: "file.pdf:12;file2.pdf:3"
    for col in ["relevant_sources_0idx", "relevant_sources", "relevant_sources_1idx"]:
        if col in row.index and not pd.isna(row.get(col)) and str(row.get(col)).strip():
            s = str(row.get(col)).strip()
            for part in _split_parts(s):
                m = re.match(r"(.+):(\d+)$", part)
                if not m:
                    continue
                src = normalize_filename(m.group(1)) if spec.match_source_by_basename else str(m.group(1)).strip()
                page_raw = int(m.group(2))
                # interpret idx flavor
                if col.endswith("_0idx"):
                    page0 = page_raw
                else:
                    # assume 1idx or unknown
                    page0 = page_raw - 1
                if page0 >= 0:
                    out.add((src, page0))
            if out:
                return out

    # 2) source_file + page columns
    page1 = extract_page_1based(row)
    if page1 is not None:
        page0 = page1 - spec.csv_page_base
        if page0 >= 0:
            for c in ["source_file", "relevant_docs", "relevant_doc", "doc", "source"]:
                if c in row.index and not pd.isna(row.get(c)) and str(row.get(c)).strip():
                    src = normalize_filename(row.get(c)) if spec.match_source_by_basename else str(row.get(c)).strip()
                    out.add((src, int(page0)))
                    break

    return out


def parse_relevant_chunk_ids(row: pd.Series) -> Set[int]:
    """Optional direct chunk id labels. Not required for long-term GT, but accept if present."""
    out: Set[int] = set()

    def _parse_int_set(val: Any) -> Set[int]:
        s = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val).strip()
        if not s:
            return set()
        s = s.strip("[](){}")
        got = set()
        for tok in re.split(r"[|,;\s]+", s):
            tok = tok.strip()
            if tok.isdigit():
                got.add(int(tok))
        return got

    for col in row.index:
        if str(col).lower().startswith("relevant_chunk"):
            out |= _parse_int_set(row.get(col))
    for col in ("relevant_doc_chunk_ids", "chunk_id", "chunk_ids"):
        if col in row.index:
            out |= _parse_int_set(row.get(col))

    return out


def build_page_to_chunkids(docs: Sequence[Any]) -> Dict[Tuple[str, int], List[int]]:
    mp: Dict[Tuple[str, int], List[int]] = {}
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = normalize_filename(meta.get("source", "")) or normalize_filename(meta.get("source_basename", ""))
        page = meta.get("page", None)
        cid = meta.get("chunk_id", None)
        if src and page is not None and cid is not None:
            mp.setdefault((src, int(page)), []).append(int(cid))
    return mp


def derive_chunk_ids_from_excerpt(
    row: pd.Series,
    docs: Sequence[Any],
    page_to_chunkids: Dict[Tuple[str, int], List[int]],
    spec: EvalSpec,
) -> Set[int]:
    """(source,page,excerpt) 기반으로 chunk_id 파생. chunk_id는 GT의 '파생값'으로만 사용."""
    excerpt = extract_excerpt(row)
    if not excerpt:
        return set()

    src_pages = parse_relevant_sources_pages(row, spec)
    if not src_pages:
        return set()

    # for robust matching
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    ex_n = norm(excerpt)
    ex_prefix = ex_n[:120]

    hits: Set[int] = set()
    for (src, page0) in src_pages:
        cids = page_to_chunkids.get((src, page0), [])
        if not cids:
            continue
        cid_set = set(cids)
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            cid = meta.get("chunk_id", None)
            if cid is None or int(cid) not in cid_set:
                continue
            if ex_prefix and ex_prefix in norm(getattr(d, "page_content", "")):
                hits.add(int(cid))

        # fallback: excerpt 매칭이 실패하면 해당 페이지 첫 chunk를 대표로 사용
        if not hits and cids:
            hits.add(int(cids[0]))

    return hits


# -----------------------------
# Retrieval metrics (chunk + page)
# -----------------------------
def recall_at_k(
    relevant_chunk_ids: Set[int],
    relevant_sources_pages: Set[Tuple[str, int]],
    retrieved_docs: Sequence[Any],
    k: int,
) -> float:
    if not relevant_chunk_ids and not relevant_sources_pages:
        return 0.0

    top = list(retrieved_docs)[: int(k)]
    got_chunks = {int(getattr(d, "metadata", {}).get("chunk_id")) for d in top if getattr(d, "metadata", {}).get("chunk_id") is not None}
    got_pages = {
        (normalize_filename(getattr(d, "metadata", {}).get("source", "")), int(getattr(d, "metadata", {}).get("page")))
        for d in top
        if getattr(d, "metadata", {}).get("source") is not None and getattr(d, "metadata", {}).get("page") is not None
    }

    hit = 0
    total = 0
    if relevant_chunk_ids:
        total += len(relevant_chunk_ids)
        hit += len(relevant_chunk_ids & got_chunks)
    if relevant_sources_pages:
        total += len(relevant_sources_pages)
        hit += len(relevant_sources_pages & got_pages)
    return float(hit / total) if total > 0 else 0.0


def mrr(
    relevant_chunk_ids: Set[int],
    relevant_sources_pages: Set[Tuple[str, int]],
    retrieved_docs: Sequence[Any],
) -> float:
    if not relevant_chunk_ids and not relevant_sources_pages:
        return 0.0
    for rank, d in enumerate(retrieved_docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        cid = meta.get("chunk_id", None)
        src = meta.get("source", None)
        page = meta.get("page", None)
        if cid is not None and int(cid) in relevant_chunk_ids:
            return 1.0 / float(rank)
        if src is not None and page is not None and (normalize_filename(src), int(page)) in relevant_sources_pages:
            return 1.0 / float(rank)
    return 0.0


# -----------------------------
# Judge prompts (JSON only)
# -----------------------------
def _json_only_rule() -> str:
    return "반드시 JSON만 출력하세요. 설명 텍스트/마크다운/코드펜스는 금지합니다."


def faithfulness_prompt(question: str, answer: str, context: str) -> str:
    return f"""{_json_only_rule()}
다음은 문서 기반 QA의 '근거 충실성(Faithfulness)' 평가입니다.
- 기준: 답변의 모든 핵심 주장(법령/수치/요구사항/권고)이 컨텍스트에 의해 직접 뒷받침되면 score=1.0
- 컨텍스트에 없는 내용을 지어냈거나 과장/추가하면 score를 낮추세요.
0.0~1.0 사이 score를 반환하세요.

JSON 스키마:
{{"score": <0.0~1.0>, "notes": "<짧은 이유>", "hallucinated_claims": ["..."], "supported_claims": ["..."]}}

[질문]
{question}

[답변]
{answer}

[컨텍스트]
{context}
"""


def truthfulness_prompt(question: str, answer: str, ground_truth: str) -> str:
    return f"""{_json_only_rule()}
다음은 '정답성(Truthfulness)' 평가입니다.
- 기준: 답변이 ground_truth의 핵심 사실관계를 얼마나 정확히 포함하는지(의미 기반) score로 평가
- 과도한 누락/오답/왜곡이 있으면 score를 낮추세요.
0.0~1.0 사이 score를 반환하세요.

JSON 스키마:
{{"score": <0.0~1.0>, "notes": "<짧은 이유>", "missing_facts": ["..."], "wrong_facts": ["..."]}}

[질문]
{question}

[답변]
{answer}

[ground_truth]
{ground_truth}
"""


def context_recall_prompt(question: str, ground_truth: str, context: str) -> str:
    return f"""{_json_only_rule()}
다음은 'Context Recall(답변 가능성)' 평가입니다.
- 기준: 컨텍스트 안에 ground_truth를 유추할 핵심 정보가 하나라도 있으면 score=1에 가깝게
- 없으면 score=0에 가깝게
0.0~1.0 사이 score를 반환하세요.

JSON 스키마:
{{"score": <0.0~1.0>, "notes": "<짧은 이유>", "evidence": ["<컨텍스트 근거 일부>"]}}

[질문]
{question}

[ground_truth]
{ground_truth}

[컨텍스트]
{context}
"""


def judge_json(llm_call, prompt: str, max_tokens: int, temperature: float) -> Tuple[float, str]:
    raw = ""
    try:
        raw = llm_call(prompt, max_tokens=max_tokens, temperature=temperature)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return 0.0, raw
        obj = json.loads(m.group(0))
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return score, raw
    except Exception:
        return 0.0, raw


# -----------------------------
# Signature helpers
# -----------------------------
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Hybrid RAG with reproducible labeling & signatures.")
    p.add_argument("--eval_csv", type=str, default=None)
    p.add_argument("--report_path", type=str, default=None)
    p.add_argument("--save_jsonl", type=str, default=None)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--debug_n", type=int, default=None)
    p.add_argument("--retrieval_only", action="store_true")
    p.add_argument("--disable_judge", action="store_true")

    # RAG overrides
    p.add_argument("--gguf_path", type=str, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--fetch_k", type=int, default=None)

    # Spec overrides
    p.add_argument("--csv_page_base", type=int, default=None, help="CSV page base (default 1)")
    p.add_argument("--no_excerpt_derivation", action="store_true", help="Disable excerpt->chunk derivation")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = EvalConfig()

    # 평가 모드는 통합 전 스타일로 짧은 답(원하면 rag_engine에서 qa_short로 바꾸세요)
    cfg.rag.prompt_mode = "eval"

    if args.eval_csv: cfg.eval_csv = args.eval_csv
    if args.report_path: cfg.report_path = args.report_path
    if args.save_jsonl: cfg.save_jsonl = args.save_jsonl
    if args.max_rows is not None: cfg.max_rows = int(args.max_rows)
    if args.debug_n is not None: cfg.debug_n = int(args.debug_n)
    if args.retrieval_only: cfg.retrieval_only = True
    if args.disable_judge: cfg.enable_judge = False

    if args.gguf_path: cfg.rag.gguf_path = args.gguf_path
    if args.alpha is not None: cfg.rag.alpha = float(args.alpha)
    if args.top_k is not None: cfg.rag.top_k = int(args.top_k)
    if args.fetch_k is not None: cfg.rag.fetch_k = int(args.fetch_k)

    if args.csv_page_base is not None: cfg.spec.csv_page_base = int(args.csv_page_base)
    if args.no_excerpt_derivation: cfg.spec.use_excerpt_to_derive_chunk = False

    os.makedirs(os.path.dirname(cfg.report_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.save_jsonl) or ".", exist_ok=True)

    df = read_csv_robust(cfg.eval_csv)
    if cfg.max_rows and cfg.max_rows > 0:
        df = df.head(cfg.max_rows)

    q_col, gt_col = detect_columns(df)

    resources = eng.load_resources(cfg.rag)
    page_to_chunkids = build_page_to_chunkids(resources.docs)

    sem_model = SentenceTransformer(cfg.rag.embedding_model)

    def judge_call(prompt: str, max_tokens: int, temperature: float) -> str:
        return eng.llama_chat(
            resources.llm,
            "당신은 엄격한 평가자입니다. JSON만 출력하세요.",
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )

    rows_out: List[Dict[str, Any]] = []
    debug_written = 0

    with open(cfg.save_jsonl, "w", encoding="utf-8") as jf:
        for idx, row in df.iterrows():
            q = str(row.get(q_col, "")).strip()
            gt = str(row.get(gt_col, "")).strip()

            retrieved = eng.retrieve(resources, q)
            retrieved_docs = [d for (d, _s, _dbg) in retrieved]

            # --- GT labels (reproducible) ---
            rel_sources_pages = parse_relevant_sources_pages(row, cfg.spec) if cfg.spec.use_source_page_labels else set()
            rel_chunk_ids = parse_relevant_chunk_ids(row)

            # excerpt -> chunk derivation (preferred when available)
            derived = set()
            if cfg.spec.use_excerpt_to_derive_chunk:
                derived = derive_chunk_ids_from_excerpt(row, resources.docs, page_to_chunkids, cfg.spec)
                if derived:
                    rel_chunk_ids |= derived
                    # chunk 라벨이 생기면 page 라벨만으로 쉬운 hit를 피하고 싶다면 아래를 켜세요.
                    # rel_sources_pages = set()

            rec_k = recall_at_k(rel_chunk_ids, rel_sources_pages, retrieved_docs, cfg.rag.top_k)
            mrr_v = mrr(rel_chunk_ids, rel_sources_pages, retrieved_docs)

            # generation
            if cfg.retrieval_only:
                pred = ""
                bleu = semsim = faith = truth = ctxrec = 0.0
                ctx_blocks = eng.make_context_blocks(retrieved)
                user_prompt = ""
            else:
                pred, ctx_blocks, user_prompt = eng.generate(resources, q, retrieved)
                bleu = compute_bleu(gt, pred, max_ngram=cfg.bleu_max_ngram)
                semsim = semantic_similarity(sem_model, gt, pred, map_to_01=cfg.semantic_map_to_01)

                faith = truth = ctxrec = 0.0
                if cfg.enable_judge:
                    faith, _ = judge_json(judge_call, faithfulness_prompt(q, pred, ctx_blocks), cfg.judge_max_tokens, cfg.judge_temperature)
                    truth, _ = judge_json(judge_call, truthfulness_prompt(q, pred, gt), cfg.judge_max_tokens, cfg.judge_temperature)
                    ctxrec, _ = judge_json(judge_call, context_recall_prompt(q, gt, ctx_blocks), cfg.judge_max_tokens, cfg.judge_temperature)

            out = {
                "row": int(idx),
                "bleu": float(bleu),
                "semantic_similarity": float(semsim),
                "faithfulness": float(faith),
                "truthfulness": float(truth),
                "context_recall": float(ctxrec),
                "recall_at_k": float(rec_k),
                "mrr": float(mrr_v),
            }
            rows_out.append(out)

            if cfg.debug_n and debug_written < cfg.debug_n:
                debug_written += 1
                jf.write(json.dumps({
                    "row": int(idx),
                    "query": q,
                    "ground_truth": gt,
                    "labels": {
                        "relevant_sources_pages": sorted(list(rel_sources_pages)),
                        "relevant_chunk_ids": sorted(list(rel_chunk_ids)),
                        "derived_chunk_ids": sorted(list(derived)),
                        "excerpt": extract_excerpt(row)[:500],
                        "csv_page_base": cfg.spec.csv_page_base,
                    },
                    "retrieval_topk": [dbg for (_d, _s, dbg) in retrieved],
                    "prompt": user_prompt,
                    "pred": pred,
                }, ensure_ascii=False) + "\n")

            print(f"[{len(rows_out)}/{len(df)}] BLEU={bleu:.3f} SemSim={semsim:.3f} "
                  f"Faith={faith:.3f} Truth={truth:.3f} CtxRecall={ctxrec:.3f} "
                  f"Recall@k={rec_k:.3f} MRR={mrr_v:.3f}")

    def mean(key: str) -> float:
        vals = [r[key] for r in rows_out]
        return float(np.mean(vals)) if vals else 0.0

    # record signature + engine hash for reproducibility
    try:
        engine_path = os.path.abspath(eng.__file__)
        engine_md5 = file_md5(engine_path)
    except Exception:
        engine_path = ""
        engine_md5 = ""

    report = {
        "input": cfg.eval_csv,
        "count": len(rows_out),
        "means": {
            "bleu": mean("bleu"),
            "semantic_similarity": mean("semantic_similarity"),
            "faithfulness": mean("faithfulness"),
            "truthfulness": mean("truthfulness"),
            "context_recall": mean("context_recall"),
            "recall_at_k": mean("recall_at_k"),
            "mrr": mean("mrr"),
        },
        "rows": rows_out,
        "repro": {
            "eval_spec": asdict(cfg.spec),
            "retriever_signature": eng.get_retriever_signature(cfg.rag),
            "rag_engine_file": engine_path,
            "rag_engine_md5": engine_md5,
        },
        "config": {
            "eval": asdict(cfg),
            "rag": asdict(cfg.rag),
        },
    }

    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nEval done.")
    print(f"- input : {cfg.eval_csv}")
    print(f"- report: {cfg.report_path}")


if __name__ == "__main__":
    nltk.download("punkt", quiet=True)
    main()