#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
- 운영 진입점: VectorDB 없으면 build_vectordb.py 실행 → 이후 대화형 질의(RAG)
- src/evaluate.py의 로직을 그대로 재사용 (BM25+FAISS+llama.cpp)

사용 예)
  python main.py
  python main.py --show_topk
  python main.py --alpha 0.6 --top_k 8
"""

import os
import sys
import subprocess
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
VDB_DIR = os.path.join(ROOT_DIR, "VectorDB")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import evaluate as ev  # noqa: E402


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

    # 2) 공통 리소스 1회 로딩 (모델/인덱스 재로딩 방지)
    cfg = ev.EvalConfig()
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.fetch_k is not None:
        cfg.fetch_k = int(args.fetch_k)

    index, docs, bm25 = ev.load_vectordb(cfg.vectordb_dir)
    embed_model = ev.SentenceTransformer(cfg.embedding_model)
    llm = ev.build_llama(cfg)
    retriever = ev.HybridRetriever(docs, bm25, index, embed_model)

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
        context_blocks = ev.make_context_blocks(retrieved)

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

        user_prompt = ev.build_user_prompt(q, context_blocks)
        answer = ev.llama_chat(
            llm,
            ev.SYSTEM_PROMPT,
            user_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

        print("\n" + "-" * 90)
        print(answer.strip())
        print("-" * 90)


if __name__ == "__main__":
    main()