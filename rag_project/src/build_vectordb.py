#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_vectordb.py
PDF 문서들로부터:
- 텍스트 추출 (pymupdf)
- chunking (RecursiveCharacterTextSplitter)
- 임베딩 (sentence-transformers)
- FAISS 인덱스 저장
- BM25 인덱스 저장
- docs.pkl 저장

출력:
VectorDB/
  - docs.pkl
  - bm25.pkl
  - faiss.index
  - config.json
"""
import re
import os
import glob
import json
import argparse
import pickle
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise ImportError("pymupdf가 필요합니다. (pip install pymupdf)") from e

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError("FAISS가 필요합니다. (conda install -c pytorch faiss-gpu) 또는 (pip faiss-cpu)") from e

from rag_engine import RAGConfig, tokenize_ko_en


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    # 줄바꿈 과다 정리
    text = re.sub(r"\n{3,}", "\n\n", text)

    # OCR/추출 깨짐: 같은 단어/구가 줄바꿈으로 반복되는 패턴 완화
    # 예) "안전관리비\n안전관리비\n안전관리비" -> "안전관리비"
    text = re.sub(r"(\b[가-힣A-Za-z0-9]{2,}\b)(\s*\n\s*\1){1,}", r"\1", text) # 검색 성능 떨어지면 이 부분 완화

    # 공백 정리
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Return list of (page_index_0based, text)."""
    doc = fitz.open(pdf_path)
    out = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = text.replace("\x00", " ").strip()
        if text:
            out.append((i, text))
    doc.close()
    return out


def build_documents(cfg: RAGConfig) -> List[Document]:
    pdfs = sorted(glob.glob(os.path.join(cfg.data_dir, "**", "*.pdf"), recursive=True))
    if not pdfs:
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {cfg.data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    docs: List[Document] = []
    chunk_id = 0
    for pdf in pdfs:
        pages = load_pdf_pages(pdf)
        for page_idx, text in pages:
            text = clean_text(text)
            chunks = splitter.split_text(text)
            for ch in chunks:
                ch = ch.strip()
                if not ch:
                    continue
                docs.append(
                    Document(
                        page_content=ch,
                        metadata={
                            "source": pdf,
                            "source_basename": os.path.basename(pdf),
                            "page": int(page_idx),  # 0-based (평가 CSV는 1-based일 수 있음)
                            "chunk_id": int(chunk_id),
                        },
                    )
                )
                chunk_id += 1
    return docs


def build_bm25(docs: List[Document]) -> BM25Okapi:
    corpus_tokens = [tokenize_ko_en(d.page_content) for d in docs]
    return BM25Okapi(corpus_tokens)


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    # embeddings: (N, D), normalized -> cosine = inner product
    embeddings = np.asarray(embeddings, dtype=np.float32)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help="PDF 폴더(기본: cfg.data_dir)")
    ap.add_argument("--vectordb_dir", type=str, default=None, help="VectorDB 출력 폴더(기본: cfg.vectordb_dir)")
    ap.add_argument("--embedding_model", type=str, default=None, help="임베딩 모델명(기본: cfg.embedding_model)")
    ap.add_argument("--chunk_size", type=int, default=None)
    ap.add_argument("--chunk_overlap", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = RAGConfig()
    if args.data_dir: cfg.data_dir = args.data_dir
    if args.vectordb_dir: cfg.vectordb_dir = args.vectordb_dir
    if args.embedding_model: cfg.embedding_model = args.embedding_model
    if args.chunk_size is not None: cfg.chunk_size = int(args.chunk_size)
    if args.chunk_overlap is not None: cfg.chunk_overlap = int(args.chunk_overlap)

    os.makedirs(cfg.vectordb_dir, exist_ok=True)

    print("[1/4] Load PDFs...")
    docs = build_documents(cfg)
    print(f"  - pages/chunks: {len(docs)}")

    print("[2/4] Build BM25...")
    bm25 = build_bm25(docs)

    print("[3/4] Build embeddings + FAISS...")
    embed_model = SentenceTransformer(cfg.embedding_model)
    texts = [d.page_content for d in docs]
    embs = embed_model.encode(texts, batch_size=int(args.batch_size), normalize_embeddings=True, show_progress_bar=True)
    embs = np.asarray(embs, dtype=np.float32)
    index = build_faiss_index(embs)

    print("[4/4] Save VectorDB...")
    with open(os.path.join(cfg.vectordb_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    with open(os.path.join(cfg.vectordb_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(index, os.path.join(cfg.vectordb_dir, "faiss.index"))

    cfg_dump = asdict(cfg)
    with open(os.path.join(cfg.vectordb_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"- VectorDB: {cfg.vectordb_dir}")
    print(f"- Files   : docs.pkl / bm25.pkl / faiss.index / config.json")


if __name__ == "__main__":
    main()