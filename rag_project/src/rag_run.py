#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_run.py (fixed)
- 운영(질의응답)용 실행 스크립트
- rag_engine_fixed.py(공용 엔진) 사용: evaluate.py(통합 전)과 동일한
  프롬프트/하이브리드 스코어링(전역 정규화)로 맞춰 재현성 확보

사용 예)
  python rag_run_fixed.py --query "산업안전보건기준에 관한 규칙 제291조는?" --show_topk
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import rag_engine as eng  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hybrid RAG (BM25+FAISS) using rag_engine_fixed.py")
    p.add_argument("--query", "-q", type=str, required=True, help="사용자 질의")
    p.add_argument("--alpha", type=float, default=None, help="Hybrid alpha override (0~1)")
    p.add_argument("--top_k", type=int, default=None, help="Top-k override")
    p.add_argument("--fetch_k", type=int, default=None, help="Fetch-k override")
    p.add_argument("--show_context", action="store_true", help="검색 컨텍스트(Top-k) 전문을 출력")
    p.add_argument("--show_topk", action="store_true", help="Top-k 메타정보(문서/페이지/스코어) 출력")
    p.add_argument("--max_context_chars", type=int, default=1200, help="show_context=False일 때 각 chunk preview 길이")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = eng.RAGConfig()

    # CLI overrides
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.fetch_k is not None:
        cfg.fetch_k = int(args.fetch_k)

    # Load shared resources
    res = eng.load_resources(cfg)
    retriever = eng.HybridRetriever(res.docs, res.bm25, res.index, res.embed_model)

    print("\n" + "=" * 90)
    print("[QUERY]")
    print(args.query)

    retrieved = retriever.retrieve(args.query, cfg.alpha, cfg.top_k, cfg.fetch_k)
    context_blocks = eng.make_context_blocks(retrieved)

    if args.show_topk:
        print("\n[TOP-K Retrieved]")
        for (dd, _s, dbg) in retrieved:
            print(
                f" rank={dbg['rank']} src={dbg['source_basename']} page={dbg['page']} "
                f"chunk={dbg['chunk_id']} hybrid={dbg['hybrid']:.3f} bm25={dbg['bm25']:.2f} vec={dbg['vec']:.3f}"
            )

    if args.show_context:
        print("\n[CONTEXT BLOCKS]\n")
        print(context_blocks)
    else:
        print("\n[CONTEXT PREVIEW]\n")
        for i, (dd, _s, dbg) in enumerate(retrieved, start=1):
            src = dbg.get("source_basename")
            page = dbg.get("page")
            prev = dd.page_content[: args.max_context_chars].replace("\n", " ")
            print(f"[{i}] {src} p.{page} chunk={dbg.get('chunk_id')} :: {prev}")
            print()

    user_prompt = eng.build_user_prompt(args.query, context_blocks)
    llm = eng.build_llama(cfg)

    answer = eng.llama_chat(
        llm,
        eng.SYSTEM_PROMPT,
        user_prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    print("\n" + "=" * 90)
    print("[ANSWER]\n")
    print(answer.strip())
    print()


if __name__ == "__main__":
    main()