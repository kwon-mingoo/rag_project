#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py
- rag_engine.py를 사용해 "동일한 검색/생성 로직"으로 평가합니다.

지표:
(Generation)
- BLEU
- Semantic Similarity (cosine on embeddings)
- Faithfulness (LLM judge, 0~1)
- Truthfulness (LLM judge, 0~1)

(Retrieval)
- Context Recall (LLM judge, 0~1)
- Recall@k
- MRR

입력 CSV: eval_data/*.csv (기본: eval_data/eval_dataset_rag_queries.csv)
출력 JSON: eval_data/eval_report.json
"""

import os
import re
import json
import argparse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sentence_transformers import SentenceTransformer

import rag_engine as eng


# -----------------------------
# Eval Config (확장: judge 옵션, 저장 옵션)
# -----------------------------
@dataclass
class EvalConfig:
    # paths
    eval_csv: str = "eval_data/eval_dataset_rag_queries.csv"
    report_path: str = "eval_data/eval_report.json"
    save_jsonl: str = "eval_data/retrieval_debug.jsonl"

    # reuse RAGConfig fields
    rag: eng.RAGConfig = field(default_factory=eng.RAGConfig)

    # evaluation behavior
    max_rows: int = -1
    debug_n: int = 0
    retrieval_only: bool = False
    enable_judge: bool = True  # faith/truth/ctxrecall

    # judge settings (same model by default)
    judge_max_tokens: int = 2048   # 평가 토큰 수
    judge_temperature: float = 0.0 # 평가 랜덤성 조절   

    # metric settings (통합 전 evaluate.py와 동일하게 맞추기)
    bleu_max_ngram: int = 4
    semantic_map_to_01: bool = True


def compute_bleu(ref: str, hyp: str, max_ngram: int = 4) -> float:
    """BLEU-4 (통합 전 evaluate.py와 동일한 설정)"""
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
    cos = cosine_sim(np.asarray(v[0]), np.asarray(v[1]))  # -1..1 (대부분 0..1 근처)
    if map_to_01:
        return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
    return float(cos)


# -----------------------------
# Robust CSV loading
# -----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def normalize_filename(s: str) -> str:
    s = os.path.basename(str(s))
    return s.strip()


def get_relevant_sources_from_row(row: pd.Series) -> Set[str]:
    sources: Set[str] = set()
    # common columns
    for col in row.index:
        if str(col).lower().startswith("relevant_sources"):
            val = row.get(col, "")
            if pd.isna(val) or val == "":
                continue
            # allow list-like strings
            parts = re.split(r"[|;,]\s*|\s*\n\s*", str(val))
            for p in parts:
                p = p.strip()
                if p:
                    sources.add(normalize_filename(p))
    # fallback
    if "source_file" in row.index and not pd.isna(row.get("source_file")):
        sources.add(normalize_filename(row.get("source_file")))
    if "relevant_docs" in row.index and not pd.isna(row.get("relevant_docs")):
        sources.add(normalize_filename(row.get("relevant_docs")))
    return {s for s in sources if s}


def _parse_int_set(val: Any) -> Set[int]:
    out: Set[int] = set()
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return out
    s = str(val).strip()
    if not s:
        return out
    # try list-like
    s = s.strip("[](){}")
    for tok in re.split(r"[|,;\s]+", s):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            out.add(int(tok))
    return out


def get_relevant_chunk_ids_from_row(row: pd.Series) -> Set[int]:
    ids: Set[int] = set()
    for col in row.index:
        if str(col).lower().startswith("relevant_chunk"):
            ids |= _parse_int_set(row.get(col))
    # also accept these legacy columns
    for col in ("relevant_doc_chunk_ids", "chunk_id", "chunk_ids"):
        if col in row.index:
            ids |= _parse_int_set(row.get(col))
    return ids


def derive_relevant_chunk_ids_from_source_page(
    docs: Sequence[Any],
    sources: Set[str],
    page_1based: Optional[int],
) -> Set[int]:
    """If CSV provides (source_file + page), map to chunk_ids by matching docs metadata."""
    if not sources or not page_1based:
        return set()
    page0 = int(page_1based) - 1
    out: Set[int] = set()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = normalize_filename(meta.get("source") or meta.get("source_basename") or "")
        if src in sources and int(meta.get("page", -999)) == page0:
            cid = meta.get("chunk_id")
            if cid is not None:
                out.add(int(cid))
    return out


def recall_at_k(relevant: Set[int], retrieved_ids: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved_ids[:k])
    hit = len(relevant & top)
    return float(hit) / float(len(relevant))


def mrr(relevant: Set[int], retrieved_ids: List[int]) -> float:
    if not relevant:
        return 0.0
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / float(i)
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


def judge_json(llm, prompt: str, max_tokens: int, temperature: float) -> Tuple[float, str]:
    raw = ""
    try:
        raw = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        # extract first JSON object
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return 0.0, raw
        obj = json.loads(m.group(0))
        score = float(obj.get("score", 0.0))
        if score < 0.0: score = 0.0
        if score > 1.0: score = 1.0
        return score, raw
    except Exception:
        return 0.0, raw


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Hybrid RAG (BM25+FAISS) using rag_engine shared core.")
    p.add_argument("--eval_csv", type=str, default=None, help="평가 CSV 경로")
    p.add_argument("--report_path", type=str, default=None)
    p.add_argument("--save_jsonl", type=str, default=None)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--debug_n", type=int, default=None)
    p.add_argument("--retrieval_only", action="store_true")
    p.add_argument("--disable_judge", action="store_true")
    p.add_argument("--gguf_path", type=str, default=None, help="GGUF 경로 override")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--fetch_k", type=int, default=None)

    p.add_argument("--show", action="store_true", help="터미널에 Q/컨텍스트/프롬프트/답변 출력")
    p.add_argument("--show_every", type=int, default=1, help="N개마다 출력(기본: 매 샘플)")
    p.add_argument("--show_topk", type=int, default=3, help="출력할 컨텍스트 Top-k 개수")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = EvalConfig()

    # 통합 전 evaluate.py와 동일한 생성 프롬프트 스타일 사용
    cfg.rag.prompt_mode = 'eval'

    if args.eval_csv: cfg.eval_csv = args.eval_csv
    if args.report_path: cfg.report_path = args.report_path
    if args.save_jsonl: cfg.save_jsonl = args.save_jsonl
    if args.max_rows is not None: cfg.max_rows = int(args.max_rows)
    if args.debug_n is not None: cfg.debug_n = int(args.debug_n)
    if args.retrieval_only: cfg.retrieval_only = True
    if args.disable_judge: cfg.enable_judge = False

    # RAGConfig overrides
    if args.gguf_path: cfg.rag.gguf_path = args.gguf_path
    if args.alpha is not None: cfg.rag.alpha = float(args.alpha)
    if args.top_k is not None: cfg.rag.top_k = int(args.top_k)
    if args.fetch_k is not None: cfg.rag.fetch_k = int(args.fetch_k)

    os.makedirs(os.path.dirname(cfg.report_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.save_jsonl) or ".", exist_ok=True)

    df = read_csv_robust(cfg.eval_csv)
    if cfg.max_rows and cfg.max_rows > 0:
        df = df.head(cfg.max_rows)

    # Load shared resources once
    resources = eng.load_resources(cfg.rag)

    # embedding model for SemSim (reuse same embedding model as retrieval)
    sem_model = SentenceTransformer(cfg.rag.embedding_model)

    # judge uses same llama (stable direct call) by default
    def judge_call(prompt: str, max_tokens: int, temperature: float) -> str:
        return eng.llama_chat(
            resources.llm,
            "당신은 엄격한 평가자입니다. " + "JSON만 출력하세요.",
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )

    # Metrics accum
    rows_out: List[Dict[str, Any]] = []
    debug_written = 0
    with open(cfg.save_jsonl, "w", encoding="utf-8") as jf:
        for idx, row in df.iterrows():
            q = str(row.get("question", row.get("query", ""))).strip()
            gt = str(row.get("ground_truth", "")).strip()

            retrieved = eng.retrieve(resources, q)
            retrieved_ids = [int(dbg.get("chunk_id")) for (_d, _s, dbg) in retrieved if dbg.get("chunk_id") is not None]

            # ground truth ids (robust)
            rel_sources = get_relevant_sources_from_row(row)
            rel_ids = get_relevant_chunk_ids_from_row(row)
            # 추가: relevant_sources_1idx/0idx 를 chunk_id로도 해석 (CSV 호환)
            if not rel_ids:
                for col in ("relevant_sources_1idx", "relevant_sources_0idx"):
                    if col in row.index:
                        rel_ids |= _parse_int_set(row.get(col))

            # fallback mapping by (source + page)
            page = None
            if "page" in row.index and not pd.isna(row.get("page")):
                try:
                    page = int(row.get("page"))
                except Exception:
                    page = None
            if not rel_ids:
                rel_ids = derive_relevant_chunk_ids_from_source_page(resources.docs, rel_sources, page)

            rec_k = recall_at_k(rel_ids, retrieved_ids, cfg.rag.top_k)
            mrr_v = mrr(rel_ids, retrieved_ids)

            # generation
            if cfg.retrieval_only:
                pred = ""
                bleu = 0.0
                semsim = 0.0
                faith = 0.0
                truth = 0.0
                ctxrec = 0.0
                ctx_blocks = eng.make_context_blocks(retrieved)
                user_prompt = ""
            else:
                pred, ctx_blocks, user_prompt = eng.generate(resources, q, retrieved)
                print("\n" + "="*120)
                print("[Q]", q)
                print("[A]", pred)
                print("="*120 + "\n")
                
                bleu = compute_bleu(gt, pred, max_ngram=cfg.bleu_max_ngram)
                semsim = semantic_similarity(sem_model, gt, pred, map_to_01=cfg.semantic_map_to_01)

                faith = truth = ctxrec = 0.0
                if cfg.enable_judge:
                    faith, _rawf = judge_json(
                        lambda p, max_tokens, temperature: judge_call(p, max_tokens, temperature),
                        faithfulness_prompt(q, pred, ctx_blocks),
                        max_tokens=cfg.judge_max_tokens,
                        temperature=cfg.judge_temperature,
                    )
                    truth, _rawt = judge_json(
                        lambda p, max_tokens, temperature: judge_call(p, max_tokens, temperature),
                        truthfulness_prompt(q, pred, gt),
                        max_tokens=cfg.judge_max_tokens,
                        temperature=cfg.judge_temperature,
                    )
                    ctxrec, _rawc = judge_json(
                        lambda p, max_tokens, temperature: judge_call(p, max_tokens, temperature),
                        context_recall_prompt(q, gt, ctx_blocks),
                        max_tokens=cfg.judge_max_tokens,
                        temperature=cfg.judge_temperature,
                    )

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

            # debug print + jsonl
            if cfg.debug_n and debug_written < cfg.debug_n:
                debug_written += 1
                print("\n" + "=" * 100)
                print(f"[DEBUG row {idx}]")
                print("QUERY(first 250):", q[:250].replace("\n", " "))
                print("GT sources:", sorted(list(rel_sources))[:5])
                print("GT chunk_ids:", sorted(list(rel_ids))[:10])
                print("\n[TOP-K Retrieved]")
                for (_d, _s, dbg) in retrieved:
                    print(
                        f" rank={dbg['rank']} src={dbg['source_basename']} page={dbg['page']} "
                        f"chunk={dbg['chunk_id']} hybrid={dbg['hybrid']:.3f} bm25={dbg['bm25']:.2f} vec={dbg['vec']:.3f}"
                    )
                if not cfg.retrieval_only:
                    print("\n[PROMPT preview]\n", user_prompt[:1200])
                    print("\n[GENERATION OUTPUT preview]\n", pred[:800])

                jf.write(json.dumps({
                    "row": int(idx),
                    "query": q,
                    "gt_sources": sorted(list(rel_sources)),
                    "gt_chunk_ids": sorted(list(rel_ids)),
                    "retrieved": [dbg for (_d, _s, dbg) in retrieved],
                    "prompt": user_prompt if not cfg.retrieval_only else "",
                    "pred": pred if not cfg.retrieval_only else "",
                }, ensure_ascii=False) + "\n")

            print(f"[{len(rows_out)}/{len(df)}] BLEU={bleu:.3f} SemSim={semsim:.3f} Faith={faith:.3f} Truth={truth:.3f} "
                  f"CtxRecall={ctxrec:.3f} Recall@k={rec_k:.3f} MRR={mrr_v:.3f}")

    # aggregate report
    def mean(key: str) -> float:
        vals = [r[key] for r in rows_out]
        return float(np.mean(vals)) if vals else 0.0

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