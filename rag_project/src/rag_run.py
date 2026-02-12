#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_run.py
- 운영(질의응답)용 실행 스크립트
- rag_engine.py(공용 엔진) 사용

사용 예)
  python src/rag_run.py --query "산업안전보건기준에 관한 규칙 제291조는?" --show_topk
  python src/rag_run.py --query "..." --report official_letter
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

    # Output controls
    p.add_argument("--show_context", action="store_true", help="검색 컨텍스트(Top-k) 전문을 출력")
    p.add_argument("--show_topk", action="store_true", help="Top-k 메타정보(문서/페이지/스코어) 출력")
    p.add_argument("--max_context_chars", type=int, default=1200, help="show_context=False일 때 각 chunk preview 길이")

    # Report mode
    p.add_argument(
        "--report",
        type=str,
        default="review_report",
        choices=["review_report", "official_letter", "checklist", "qa_short"],
        help="출력 문서 형식 선택",
    )
    # 필요하면 CLI에서 eval/prod도 바꿀 수 있게 열어둠(기본 prod)
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

    # ✅ report/prompt settings
    cfg.report_type = args.report
    cfg.prompt_mode = args.prompt_mode

    # Load shared resources once (docs/bm25/faiss/embed/llm/retriever)
    res = eng.load_resources(cfg)

    print("\n" + "=" * 90)
    print("[QUERY]")
    print(args.query)

    # Retrieve
    retrieved = res.retriever.retrieve(args.query, cfg.alpha, cfg.top_k, cfg.fetch_k)
    context_blocks = eng.make_context_blocks(retrieved)

    # Show retrieval
    if args.show_topk:
        print("\n[TOP-K Retrieved]")
        for (_d, _s, dbg) in retrieved:
            print(
                f" rank={dbg.get('rank')} src={dbg.get('source_basename')} page={dbg.get('page')} "
                f"chunk={dbg.get('chunk_id')} hybrid={dbg.get('hybrid'):.3f} "
                f"bm25={dbg.get('bm25'):.2f} vec={dbg.get('vec'):.3f}"
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
            print(f"[{i}] {src} p.{page} chunk={dbg.get('chunk_id')} :: {prev}\n")

    # Build prompt (report_type 반영)
    user_prompt = eng.build_user_prompt(
        args.query,
        context_blocks,
        report_type=cfg.report_type,
        prompt_mode=cfg.prompt_mode,
    )

    # Generate (reuse res.llm)
    answer = eng.llama_chat(
        res.llm,
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
