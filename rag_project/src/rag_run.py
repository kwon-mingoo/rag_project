#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_run.py
- 운영(질의응답)용 실행 스크립트
- rag_engine.py(공용 엔진) 사용

사용 예)
  python src/rag_run.py --query "산업안전보건기준에 관한 규칙 제291조는?" --show_topk
  python src/rag_run.py --query "..." --report doc_draft
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import rag_engine as eng  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hybrid RAG (BM25+FAISS) using rag_engine.py")
    p.add_argument("--query", "-q", type=str, required=True, help="사용자 질의")

    # Retrieval overrides
    p.add_argument("--alpha", type=float, default=None, help="Hybrid alpha override (0~1)")
    p.add_argument("--top_k", type=int, default=None, help="Top-k override")
    p.add_argument("--fetch_k", type=int, default=None, help="Fetch-k override")

    # MMR overrides (optional but recommended)
    p.add_argument("--mmr", action="store_true", help="MMR 강제 ON (cfg 기본값 override)")
    p.add_argument("--mmr_candidates", type=int, default=None, help="MMR 후보 수 override")
    p.add_argument("--mmr_lambda", type=float, default=None, help="MMR lambda override (0~1)")

    # Output controls
    p.add_argument("--show_context", action="store_true", help="검색 컨텍스트(Top-k) 전문을 출력")
    p.add_argument("--show_topk", action="store_true", help="Top-k 메타정보(문서/페이지/스코어) 출력")
    p.add_argument("--max_context_chars", type=int, default=1200, help="show_context=False일 때 각 chunk preview 길이")

    # Report mode
    p.add_argument(
        "--report",
        type=str,
        default="doc_draft",
        choices=["review_report", "official_letter", "checklist", "qa_short", "doc_draft"],
        help="출력 문서 형식 선택",
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        default="prod",
        choices=["prod", "eval"],
        help="프롬프트 모드(prod=운영 리포트, eval=평가/짧은 답)",
    )
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

    # report/prompt settings
    cfg.report_type = args.report
    cfg.prompt_mode = args.prompt_mode

    # MMR settings
    if args.mmr:
        cfg.mmr_enabled = True
    if args.mmr_candidates is not None:
        cfg.mmr_candidates = int(args.mmr_candidates)
    if args.mmr_lambda is not None:
        cfg.mmr_lambda = float(args.mmr_lambda)

    # Load shared resources once (docs/bm25/faiss/embed/llm/retriever)
    res = eng.load_resources(cfg)

    print("\n" + "=" * 90)
    print("[QUERY]")
    print(args.query)

    # ✅ 통합 엔진 단일 진입점
    # retrieve → make_context_blocks → build_user_prompt(report_type/prompt_mode) → generate
    answer, retrieved, context_blocks, user_prompt = eng.answer_query(res, args.query)

    # Show retrieval (dbg 키 엔진 기준으로 맞춤)
    if args.show_topk:
        print("\n[TOP-K Retrieved]")
        for rank, (_d, _s, dbg) in enumerate(retrieved, start=1):
            # dbg에는 rank가 없으니 rank는 enumerate로 계산
            print(
                f" rank={rank} src={dbg.get('source_basename')} page={dbg.get('page')} "
                f"chunk={dbg.get('chunk_id')} hybrid={dbg.get('hybrid'):.3f} "
                f"bm25_n={dbg.get('bm25_n'):.3f} vec_n={dbg.get('vec_n'):.3f}"
            )

    # Context output
    if args.show_context:
        print("\n[CONTEXT BLOCKS]\n")
        print(context_blocks)
    else:
        print("\n[CONTEXT PREVIEW]\n")
        for i, (dd, _s, dbg) in enumerate(retrieved, start=1):
            src = dbg.get("source_basename")
            page = dbg.get("page")
            prev = (dd.page_content or "")[: args.max_context_chars].replace("\n", " ")
            print(f"[{i}] {src} p.{page} chunk={dbg.get('chunk_id')} :: {prev}\n")

    print("\n" + "=" * 90)
    print("[ANSWER]\n")
    print(answer.strip())
    print()


if __name__ == "__main__":
    main()
