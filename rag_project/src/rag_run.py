#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_run.py
- 운영(질의응답)용 실행 스크립트
- src/evaluate.py에 이미 구현된 로딩/하이브리드 검색/llama_cpp 생성 로직을 재사용합니다.

사용 예)
  python src/rag_run.py --query "가설울타리 계획변경 시 안전관리계획 수립 대상인가?"
  python src/rag_run.py --query "산업안전보건기준에 관한 규칙 제291조는?" --show_context --show_topk
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import evaluate as ev  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hybrid RAG (BM25+FAISS) using evaluate.py components.")
    p.add_argument("--query", "-q", type=str, required=True, help="사용자 질의")
    p.add_argument("--alpha", type=float, default=None, help="Hybrid alpha override (0~1)")
    p.add_argument("--top_k", type=int, default=None, help="Top-k override")
    p.add_argument("--fetch_k", type=int, default=None, help="Fetch-k override")
    p.add_argument("--show_context", action="store_true", help="검색 컨텍스트(Top-k) 전문을 출력")
    p.add_argument("--show_topk", action="store_true", help="Top-k 메타정보(문서/페이지/스코어) 출력")
    p.add_argument("--max_context_chars", type=int, default=1200, help="show_context=False일 때 각 chunk preview 길이")
    return p.parse_args()


def run_single_query(cfg: ev.EvalConfig, query: str, show_context: bool, show_topk: bool, max_context_chars: int) -> str:
    # Load VectorDB
    index, docs, bm25 = ev.load_vectordb(cfg.vectordb_dir)

    # Models
    embed_model = ev.SentenceTransformer(cfg.embedding_model)
    llm = ev.build_llama(cfg)

    # Retriever
    retriever = ev.HybridRetriever(docs, bm25, index, embed_model)

    # Retrieve
    retrieved = retriever.retrieve(query, cfg.alpha, cfg.top_k, cfg.fetch_k)
    context_blocks = ev.make_context_blocks(retrieved)

    if show_topk:
        print("\n[TOP-K Retrieved]")
        for (dd, _s, dbg) in retrieved:
            print(
                f" rank={dbg['rank']} src={dbg['source_basename']} page={dbg['page']} "
                f"chunk={dbg['chunk_id']} hybrid={dbg['hybrid']:.3f} bm25={dbg['bm25']:.2f} vec={dbg['vec']:.3f}"
            )

    if show_context:
        print("\n[CONTEXT BLOCKS]\n")
        print(context_blocks)
    else:
        print("\n[CONTEXT PREVIEW]\n")
        for i, (dd, _s, dbg) in enumerate(retrieved, start=1):
            src = dbg.get("source_basename")
            page = dbg.get("page")
            prev = dd.page_content[:max_context_chars].replace("\n", " ")
            print(f"[{i}] {src} p.{page} chunk={dbg.get('chunk_id')} :: {prev}")
            print()

    # 운영형 프롬프트(평가 코드에 있는 템플릿 그대로 재사용)
    user_prompt = ev.build_user_prompt(query, context_blocks)

    # Generate
    answer = ev.llama_chat(
        llm,
        ev.SYSTEM_PROMPT,
        user_prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    return answer


def main():
    args = parse_args()
    cfg = ev.EvalConfig()

    # CLI overrides
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.fetch_k is not None:
        cfg.fetch_k = int(args.fetch_k)

    print("\n" + "=" * 90)
    print("[QUERY]")
    print(args.query)

    answer = run_single_query(
        cfg,
        args.query,
        show_context=args.show_context,
        show_topk=args.show_topk,
        max_context_chars=args.max_context_chars,
    )

    print("\n" + "=" * 90)
    print("[ANSWER]\n")
    print(answer.strip())
    print()


if __name__ == "__main__":
    main()