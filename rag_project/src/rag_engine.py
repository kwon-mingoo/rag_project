#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_engine.py
공용 RAG 엔진 (운영/평가 공통)

- VectorDB 로드 (docs.pkl / bm25.pkl / faiss.index)
- BM25 + FAISS 하이브리드 검색
- llama.cpp (llama-cpp-python) 직접 호출로 생성 (LangChain 래퍼 미사용)
- 프롬프트 템플릿 제공 (SYSTEM_PROMPT / build_user_prompt)

이 파일은 evaluate.py / rag_run.py / main.py(선택)에서 공통으로 import해서 사용합니다.
"""

from __future__ import annotations

import os
import json
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError("FAISS가 필요합니다. (conda install -c pytorch faiss-gpu) 또는 (pip faiss-cpu)") from e

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

try:
    from llama_cpp import Llama  # llama-cpp-python
except Exception as e:
    raise ImportError("llama-cpp-python이 필요합니다. (pip install llama-cpp-python)") from e


# -----------------------------
# Prompts
# -----------------------------
SYSTEM_PROMPT = """당신은 건설 관련 법령·행정 문서를 근거로
검토 의견을 작성하는 문서 기반 QA 어시스턴트입니다.

원칙:
- 아래 [컨텍스트]에 포함된 문서 내용만 근거로 답변합니다.
- 컨텍스트에 없는 법령·조항·수치·행정 해석은 추측하거나 단정하지 않습니다.
- 모든 판단은 조건부·검토형 표현만 사용합니다.
  (예: "~할 수 있음", "~에 따라 달라질 수 있음", "~여부는 추가 검토 필요")
- “~이다”, “~해야 한다”, “~대상이다”와 같은 단정 표현을 사용하지 않습니다.
- 법령은 결론의 근거가 아니라 ‘검토 기준’으로만 인용합니다.
- 근거가 부족한 경우 반드시 “근거 부족”을 명시하고,
  추가로 확인이 필요한 정보를 질문합니다.
- 모든 요구사항·검토 사항에는 출처 번호([1], [2] …)를 붙입니다.

"""


def build_user_prompt(question: str, context_blocks: str) -> str:
    return f"""질문: {question}

요청 사항:

1) 컨텍스트에 근거하여,
   가설울타리 계획변경 시 안전관리계획 수립 여부를
   검토할 때 고려해야 할 기준·조건을
   조건부 표현으로 bullet 형태로 정리하세요.

2) 각 기준 또는 조건마다
   근거가 되는 문서명과 조항·페이지를 함께 제시하세요.
   (법령 / 시행령 / 시행규칙 / 질의·회신 / 사례집 구분)

3) 단독 가설울타리 설치와
   다른 가설공사·고소작업·해체공사와 결합되는 경우를
   구분하여 검토 포인트를 정리하세요.

4) 종합 검토 의견(1~2문장):
   - 본 사안이 안전관리계획 또는 관련 계획의
     검토 대상으로 전환될 수 있는 조건을 요약하고,
   - 추가로 확인이 필요한 핵심 정보를 제시하세요.
   
※ 주의:
- 법적 판단 또는 행정 결론을 내리지 마세요.
- 컨텍스트에 없는 내용은 포함하지 마세요.



[컨텍스트]
{context_blocks}
"""


# -----------------------------
# Tokenize / Normalize helpers
# -----------------------------
_RE_WORD = re.compile(r"[A-Za-z0-9]+|[가-힣]+")


def tokenize_ko_en(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    return _RE_WORD.findall(text.lower())


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# -----------------------------
# Config
# -----------------------------
@dataclass
class RAGConfig:
    # Paths
    vectordb_dir: str = "VectorDB"
    data_dir: str = "data"

    # Embeddings / Retrieval
    embedding_model: str = "BAAI/bge-m3"
    alpha: float = 0.5      # 0~1 (vec:alpha, bm25:1-alpha)
    top_k: int = 6
    fetch_k: int = 60       # 후보 풀

    # Chunking (build_vectordb에서 사용)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # LLM (llama.cpp)
    gguf_path: str = "/home/nami/Desktop/rag_project_v2/models/llama8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"
    chat_format: str = "llama-3"

    # Prompt mode: 'prod' (템플릿) or 'eval' (짧은 QA, 지표 비교용)
    prompt_mode: str = "prod"
    n_ctx: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.95
    n_gpu_layers: int = -1   # -1이면 가능한 만큼 GPU offload
    n_threads: int = 0       # 0이면 llama.cpp 기본값
    n_batch: int = 512


# -----------------------------
# VectorDB IO
# -----------------------------
def load_vectordb(vectordb_dir: str) -> Tuple[Any, List[Document], Any]:
    """Return (faiss_index, docs, bm25_obj)."""
    docs_path = os.path.join(vectordb_dir, "docs.pkl")
    bm25_path = os.path.join(vectordb_dir, "bm25.pkl")
    index_path = os.path.join(vectordb_dir, "faiss.index")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"docs.pkl not found: {docs_path}")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"bm25.pkl not found: {bm25_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"faiss.index not found: {index_path}")

    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    # build_vectordb.py / evaluate.py 포맷 호환: bm25.pkl이 dict로 저장된 경우
    if isinstance(bm25, dict) and "bm25" in bm25:
        bm25 = bm25["bm25"]
    index = faiss.read_index(index_path)
    return index, docs, bm25


# -----------------------------
# Retriever
# -----------------------------
class HybridRetriever:
    def __init__(self, docs: List[Document], bm25: BM25Okapi, index: Any, embed_model: SentenceTransformer):
        self.docs = docs
        self.bm25 = bm25
        self.index = index
        self.embed_model = embed_model

    def _vector_search(self, query: str, fetch_k: int) -> Tuple[np.ndarray, np.ndarray]:
        qv = self.embed_model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)
        scores, idxs = self.index.search(qv, fetch_k)
        return scores[0], idxs[0]

    def _bm25_search(self, query: str, fetch_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q_tokens = tokenize_ko_en(query)
        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)
        # top fetch_k indices
        if fetch_k >= scores.shape[0]:
            idxs = np.argsort(-scores)
        else:
            idxs = np.argpartition(-scores, fetch_k)[:fetch_k]
            idxs = idxs[np.argsort(-scores[idxs])]
        return scores[idxs], idxs.astype(np.int64)

    def retrieve(
        self,
        query: str,
        alpha: float,
        top_k: int,
        fetch_k: int,
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))
        top_k = int(top_k)
        fetch_k = int(fetch_k)

        vec_scores, vec_idxs = self._vector_search(query, fetch_k)
        bm_scores, bm_idxs = self._bm25_search(query, fetch_k)

        # Candidate union
        cand = {}
        for s, i in zip(vec_scores, vec_idxs):
            if i < 0:
                continue
            cand[int(i)] = {"vec": float(s), "bm25": 0.0}
        for s, i in zip(bm_scores, bm_idxs):
            i = int(i)
            if i not in cand:
                cand[i] = {"vec": 0.0, "bm25": float(s)}
            else:
                cand[i]["bm25"] = float(s)

        # Normalize within candidates
        vec_arr = np.array([cand[i]["vec"] for i in cand.keys()], dtype=np.float32)
        bm_arr = np.array([cand[i]["bm25"] for i in cand.keys()], dtype=np.float32)
        vec_n = minmax_norm(vec_arr)
        bm_n = minmax_norm(bm_arr)

        keys = list(cand.keys())
        out = []
        for j, i in enumerate(keys):
            hybrid = alpha * float(vec_n[j]) + (1.0 - alpha) * float(bm_n[j])
            cand[i]["hybrid"] = hybrid
            cand[i]["vec_n"] = float(vec_n[j])
            cand[i]["bm25_n"] = float(bm_n[j])

        # sort by hybrid
        keys.sort(key=lambda i: cand[i]["hybrid"], reverse=True)
        keys = keys[:top_k]

        for rank, i in enumerate(keys, start=1):
            d = self.docs[i]
            meta = d.metadata or {}
            dbg = {
                "rank": rank,
                "doc_index": i,
                "chunk_id": meta.get("chunk_id"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "source_basename": os.path.basename(meta.get("source", "")) if meta.get("source") else meta.get("source_basename"),
                "hybrid": cand[i]["hybrid"],
                "bm25": cand[i]["bm25"],
                "vec": cand[i]["vec"],
                "bm25_n": cand[i]["bm25_n"],
                "vec_n": cand[i]["vec_n"],
            }
            out.append((d, cand[i]["hybrid"], dbg))
        return out


def make_context_blocks(retrieved: Sequence[Tuple[Document, float, Dict[str, Any]]]) -> str:
    blocks = []
    for i, (doc, _score, dbg) in enumerate(retrieved, start=1):
        src = dbg.get("source_basename") or dbg.get("source") or "UNKNOWN"
        page = dbg.get("page")
        chunk_id = dbg.get("chunk_id")
        txt = doc.page_content.strip()
        blocks.append(f"[{i}] ({src} p.{page}, chunk_id={chunk_id})\n{txt}\n")
    return "\n".join(blocks).strip()


# -----------------------------
# LLM (llama.cpp direct)
# -----------------------------
def build_llama(cfg: RAGConfig) -> Llama:
    if not os.path.exists(cfg.gguf_path):
        raise FileNotFoundError(f"GGUF 모델 파일을 찾을 수 없습니다: {cfg.gguf_path}")
    kwargs = dict(
        model_path=cfg.gguf_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        n_batch=cfg.n_batch,
        chat_format=cfg.chat_format,
    )
    if cfg.n_threads and int(cfg.n_threads) > 0:
        kwargs["n_threads"] = int(cfg.n_threads)
    return Llama(**kwargs)


def build_user_input(question: str, context_blocks: str) -> str:
    """Evaluation-friendly short QA prompt (used in 통합 전 evaluate.py).
    - keeps answers short
    - encourages citation like [1],[2]
    """
    return f"""질문:\n{question}\n\n컨텍스트:\n{context_blocks}\n\n요청:\n- 반드시 컨텍스트 근거로만 답변\n- 가능하면 [1],[2] 출처 표시\n"""



def llama_chat(
    llm: Llama,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.95,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
    )
    return resp["choices"][0]["message"]["content"]


# -----------------------------
# Resources + high-level API
# -----------------------------
@dataclass
class RAGResources:
    cfg: RAGConfig
    index: Any
    docs: List[Document]
    bm25: Any
    embed_model: SentenceTransformer
    llm: Llama
    retriever: HybridRetriever


def load_resources(cfg: Optional[RAGConfig] = None) -> RAGResources:
    cfg = cfg or RAGConfig()
    index, docs, bm25 = load_vectordb(cfg.vectordb_dir)
    embed_model = SentenceTransformer(cfg.embedding_model)
    llm = build_llama(cfg)
    retriever = HybridRetriever(docs, bm25, index, embed_model)
    return RAGResources(cfg, index, docs, bm25, embed_model, llm, retriever)


def retrieve(resources: RAGResources, query: str, alpha: Optional[float] = None, top_k: Optional[int] = None, fetch_k: Optional[int] = None):
    cfg = resources.cfg
    return resources.retriever.retrieve(
        query,
        cfg.alpha if alpha is None else float(alpha),
        cfg.top_k if top_k is None else int(top_k),
        cfg.fetch_k if fetch_k is None else int(fetch_k),
    )


def generate(
    resources: RAGResources,
    query: str,
    retrieved,
    system_prompt: Optional[str] = None,
    prompt_mode: Optional[str] = None,
):
    """Generate answer.

    prompt_mode:
      - 'prod': 운영 템플릿(build_user_prompt)
      - 'eval': 통합 전 evaluate.py 스타일(build_user_input)
    """
    cfg = resources.cfg
    ctx = make_context_blocks(retrieved)
    mode = (prompt_mode or getattr(cfg, 'prompt_mode', 'prod')).lower()
    if mode == 'eval':
        user_prompt = build_user_input(query, ctx)
    else:
        user_prompt = build_user_prompt(query, ctx)
    answer = llama_chat(
        resources.llm,
        system_prompt or SYSTEM_PROMPT,
        user_prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    return answer, ctx, user_prompt


def answer_query(
    resources: RAGResources,
    query: str,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    system_prompt: Optional[str] = None,
    prompt_mode: Optional[str] = None,
):
    retrieved = retrieve(resources, query, alpha=alpha, top_k=top_k, fetch_k=fetch_k)
    answer, ctx, user_prompt = generate(resources, query, retrieved, system_prompt=system_prompt, prompt_mode=prompt_mode)
    return answer, retrieved, ctx, user_prompt