#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py (fixed)
- 운영 진입점: VectorDB 없으면 build_vectordb.py 실행 → 이후 대화형 질의(RAG)
- rag_engine_fixed.py 사용 (evaluate.py 통합 전과 동일한 retrieval/prompt 재현)

사용 예)
  python main_fixed.py
  python main_fixed.py --show_topk
  python main_fixed.py --alpha 0.6 --top_k 8
"""

import os
import sys
import subprocess
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
VDB_DIR = os.path.join(ROOT_DIR, "VectorDB")

# rag_engine_fixed.py는 src/에 두거나, main과 같은 폴더에 둔 뒤 sys.path에 추가하세요.
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import rag_engine as eng  # noqa: E402


def vectordb_ready(vdb_dir: str) -> bool:
    required = ["faiss.index", "docs.pkl", "bm25.pkl"]
    return all(os.path.exists(os.path.join(vdb_dir, f)) for f in required)


def run_build_vectordb() -> None:
    build_script = os.path.join(SRC_DIR, "build_vectordb.py")
    if not os.path.exists(build_script):
        raise FileNotFoundError(f"build_vectordb.py를 찾을 수 없습니다: {build_script}")
    print("VectorDB가 없어서 build_vectordb.py를 실행합니다...")
    subprocess.run([sys.executable, build_script], check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main entry for Hybrid RAG (BM25+FAISS)")
    p.add_argument("--no_build", action="store_true", help="VectorDB가 없어도 build를 시도하지 않음(에러)")
    p.add_argument("--alpha", type=float, default=None, help="Hybrid alpha override")
    p.add_argument("--top_k", type=int, default=None, help="Top-k override")
    p.add_argument("--fetch_k", type=int, default=None, help="Fetch-k override")
    p.add_argument("--show_topk", action="store_true", help="각 질의마다 Top-k 메타를 출력")
    p.add_argument("--show_context", action="store_true", help="각 질의마다 컨텍스트 전문 출력")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) VectorDB 체크 / 빌드
    if not vectordb_ready(VDB_DIR):
        if args.no_build:
            raise FileNotFoundError("VectorDB가 없습니다. src/build_vectordb.py를 먼저 실행하세요.")
        run_build_vectordb()
    else:
        print("VectorDB 확인 완료")

    # 2) 공통 리소스 1회 로딩
    cfg = eng.RAGConfig()
    cfg.vectordb_dir = VDB_DIR

    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.fetch_k is not None:
        cfg.fetch_k = int(args.fetch_k)

    res = eng.load_resources(cfg)
    retriever = eng.HybridRetriever(res.docs, res.bm25, res.index, res.embed_model)
    llm = eng.build_llama(cfg)

    print("\n" + "=" * 90)
    print("RAG 질의 모드입니다. 종료하려면 exit/quit 입력")
    print("=" * 90)

    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("종료합니다.")
            break

        retrieved = retriever.retrieve(q, cfg.alpha, cfg.top_k, cfg.fetch_k)
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

        user_prompt = eng.build_user_prompt(q, context_blocks)
        answer = eng.llama_chat(
            llm,
            eng.SYSTEM_PROMPT,
            user_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )

        print("\n" + "-" * 90)
        print(answer.strip())
        print("-" * 90)


if __name__ == "__main__":
    main()