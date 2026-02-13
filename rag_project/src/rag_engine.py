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


def build_user_prompt(
    question: str,
    context_blocks: str,
    report_type: str = "review_report",
    prompt_mode: str = "prod",
) -> str:
    """
    Build a user prompt for the LLM.

    report_type:
      - review_report: 검토의견서(리포트)
      - official_letter: 공문(안)
      - qa_short: 간단 QA
      - checklist: 체크리스트
    prompt_mode:
      - prod: 운영 리포트
      - eval: 평가용(짧은 답)
    """

    report_type = (report_type or "review_report").strip().lower()
    prompt_mode = (prompt_mode or "prod").strip().lower()

    # 평가 모드에서는 짧은 QA로 강제, 바꿔주면 평가모드에서도 다르게 가능
    if prompt_mode == "eval":
        #report_type = "qa_short"
        report_type = "checklist"

    # =========================================================
    #  1. QA SHORT (평가/디버그용)
    # =========================================================
    if report_type == "qa_short":
        return f"""[질의]
{question}

[컨텍스트]
{context_blocks}

[지시]
- 컨텍스트 내용만 사용하세요.
- 2~4문장으로 간결하게 답하세요.
- 모든 주장 문장 끝에 반드시 출처 번호([1],[2]...)를 표시하세요.
- 근거가 부족하면 '근거 부족'이라 명시하고 추가 질문을 작성하세요.

[출력 형식]
답변:
근거:
추가 질문(있는 경우):
"""

    # =========================================================
    #  2. CHECKLIST (현장 실행형)
    # =========================================================
    if report_type == "checklist":
        return f"""[역할]
당신은 법령/기준 기반 현장 실행 체크리스트 작성자입니다.

[질의]
{question}

[컨텍스트]
{context_blocks}

[절대 규칙]
- 컨텍스트에 없는 사실을 단정하지 마세요.
- 모든 항목 끝에 반드시 출처([1],[2]...)를 붙이세요.

[작성 절차]
1) 관련 근거를 5~15개 추출 (요약 + 출처)
2) 필수(의무) / 권고(운영) 구분
3) 체크리스트 작성

[출력 형식]
## 0) 근거 목록
- ... [출처]

## 1) 필수 요구사항 체크리스트
- [ ] ... [출처]

## 2) 권고 체크리스트
- [ ] ... [출처]

## 3) 근거 부족 / 추가 확인 질문
- ...
"""

    # =========================================================
    #  3. OFFICIAL LETTER (공문 특화 - 법령 QA 고도화)
    # =========================================================
    if report_type == "official_letter":
        return f"""[역할]
당신은 법령/기준 기반 공식 문서 작성자입니다.
컨텍스트 근거만 사용하여 공문(안)을 작성하세요.

[질의]
{question}

[컨텍스트]
{context_blocks}

[절대 규칙]
- 컨텍스트에 없는 조항/수치/요건은 단정하지 마세요.
- 모든 주장 문장 끝에 반드시 출처([1],[2]...) 표시.
- 조항 인용이 필요한 경우 짧은 원문 발췌 포함.
- 근거 부족은 별도 섹션으로 분리.

[작성 절차]
1) 근거 목록 추출 (요약 + 짧은 원문 발췌 + 출처)
2) 필수/권고 요건 표 작성
3) 위 표에 있는 내용만 사용해 공문 작성

[출력 형식]
## 0) 근거 목록(발췌)
- (요약) "원문 일부..." [출처]

## 1) 요건 정리
| 구분 | 내용 | 근거 |
|---|---|---|
| 필수 | ... | [1] |
| 권고 | ... | [2] |

## 2) 공문(안)
문서번호: (비워두기)
시행일자: (비워두기)
수    신: (비워두기)
참    조: (비워두기)
제    목: (질의 반영 제목)

1. 배경 및 목적
- ... [출처]

2. 검토 결과
- (필수) ... [출처]
- (권고) ... [출처]

3. 조치 요청 사항
- (필수) ... [출처]
- (권고) ... [출처]

4. 유의사항 / 근거 부족
- ...

끝.
"""

    # =========================================================
    #  4. REVIEW REPORT (검토의견서 - 법령 QA 최적화)
    # =========================================================
    return f"""[역할]
당신은 법령/기준 준수 검토 의견서 작성자입니다.
컨텍스트 근거만 사용합니다.

[질의]
{question}

[컨텍스트]
{context_blocks}

[절대 규칙]
- 컨텍스트에 없는 사실은 절대 단정하지 마세요.
- 모든 판단/요건/리스크 문장 끝에 출처([1],[2]...) 표시.
- 조항 인용이 중요하면 짧은 원문 발췌 포함.
- 근거 부족은 별도 섹션에 명확히 기재.

[작성 절차]
1) 근거 5~15개 추출 (요약 + 발췌 + 출처)
2) A/B 구분
   A: 필수 의무사항
   B: 권고/운영 체크포인트
3) 리스크 도출 (법적/안전/품질/운영 구분)
4) 우선순위 조치 제시

[출력 형식]
## 0) 근거 목록
- (요약) "원문 일부..." [출처]

## 1) A/B 요약
### A. 필수(의무)
- ... [출처]

### B. 권고/운영
- ... [출처]

## 2) 결론
- ... [출처]

## 3) 적용 범위 및 전제
- ... [출처 또는 근거 부족]

## 4) 상세 검토 의견
- 쟁점 1:
  - 판단: ... [출처]
  - 근거: ... [출처]
- 쟁점 2:
  - 판단: ... [출처]
  - 근거: ... [출처]

## 5) 리스크 및 권고 조치
- (법적) ... [출처]
- (안전) ... [출처]
- (품질) ... [출처]
- (운영) ... [출처]
- P1: ... [출처]
- P2: ... [출처]
- P3: ... [출처]

## 6) 근거 부족 / 추가 확인 질문
- ...
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
    alpha: float = 0.55      # 0~1 (vec:alpha, bm25:1-alpha)
    top_k: int = 6
    fetch_k: int = 40       # 후보 풀

    # Chunking (build_vectordb에서 사용)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # LLM (llama.cpp)
    gguf_path: str = "/home/nami/Desktop/rag_project_v2/models/llama8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"
    chat_format: str = "llama-3"
    n_ctx: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.95
    n_gpu_layers: int = -1   # -1이면 가능한 만큼 GPU offload
    n_threads: int = 0       # 0이면 llama.cpp 기본값
    n_batch: int = 512



    # Prompt / Report
    prompt_mode: str = "prod"   # prod|eval
    report_type: str = "review_report"  # review_report|official_letter|qa_short|checklist

def get_retriever_signature(cfg: "RAGConfig") -> str:
    # “지표가 바뀌는 원인”이 되는 요소만 담기 (경로는 넣지 않기)
    parts = [
        "retriever=hybrid(bm25+faiss)",
        f"alpha={cfg.alpha}",
        f"top_k={cfg.top_k}",
        f"fetch_k={cfg.fetch_k}",
        f"embedding_model={cfg.embedding_model}",
        # 아래는 구현/정책을 명시 (너의 build_vectordb / rag_engine 정책에 맞게 고정)
        "faiss_index=IndexFlatIP",
        "doc_embed_norm=normalize_embeddings=True",
        "query_embed_norm=normalize_embeddings=True",  # 또는 faiss.normalize_L2
        "bm25_tokenizer=_RE_WORD:v1",
        "hybrid_norm=minmax",
        "hybrid_norm_scope=candidates_union",  # 또는 "all_docs"
    ]
    return "|".join(parts)

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

    def embed_query(self, q: str) -> np.ndarray:
        # 통합 전 evaluate.py와 동일: normalize_embeddings=False 후 L2 normalize
        v = self.embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def retrieve(
        self,
        query: str,
        alpha: float,
        top_k: int,
        fetch_k: int,
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """통합 전 evaluate.py와 동일한 하이브리드 검색.
        - BM25: 전체 문서 점수 -> minmax
        - FAISS: fetch_k 후보 점수 -> 전체 길이 벡터에 채움 -> minmax
        - hybrid = alpha*vec + (1-alpha)*bm25
        """
        alpha = float(max(0.0, min(1.0, float(alpha))))
        top_k = int(top_k)
        fetch_k = int(fetch_k)

        # BM25 over all docs
        q_tokens = tokenize_ko_en(query)
        bm25_scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)
        bm25_n = minmax_norm(bm25_scores)

        # FAISS
        qv = self.embed_query(query)
        sim, idx = self.index.search(qv, fetch_k)
        sim = sim[0].astype(np.float32)
        idx = idx[0].astype(np.int64)

        vec_scores = np.zeros(len(self.docs), dtype=np.float32)
        mask = idx >= 0
        vec_scores[idx[mask]] = sim[mask]
        vec_n = minmax_norm(vec_scores)

        # Hybrid
        hybrid = alpha * vec_n + (1.0 - alpha) * bm25_n
        top_idx = np.argsort(-hybrid)[:top_k]

        results: List[Tuple[Document, float, Dict[str, Any]]] = []
        for rank_i, i in enumerate(top_idx, start=1):
            d = self.docs[int(i)]
            meta = d.metadata or {}
            dbg = {
                "rank": rank_i,
                "doc_index": int(i),
                "chunk_id": meta.get("chunk_id"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "source_basename": os.path.basename(str(meta.get("source", ""))) if meta.get("source") else meta.get("source_basename"),
                "hybrid": float(hybrid[int(i)]),
                "bm25": float(bm25_scores[int(i)]),
                "vec": float(vec_scores[int(i)]),
            }
            results.append((d, float(hybrid[int(i)]), dbg))
        return results

def make_context_blocks(retrieved):
    # 완전 동일 텍스트만 제거
    seen_txt = set()
    uniq = []
    for doc, score, dbg in retrieved:
        txt = (doc.page_content or "").strip()
        if not txt:
            continue
        if txt in seen_txt:
            continue
        seen_txt.add(txt)
        uniq.append((doc, score, dbg))

    retrieved = uniq

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


def generate(resources: RAGResources, query: str, retrieved, system_prompt: Optional[str] = None):
    cfg = resources.cfg
    ctx = make_context_blocks(retrieved)
    user_prompt = build_user_prompt(query, ctx, report_type=getattr(cfg, 'report_type', 'review_report'), prompt_mode=getattr(cfg, 'prompt_mode', 'prod'))
    answer = llama_chat(
        resources.llm,
        system_prompt or SYSTEM_PROMPT,
        user_prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    return answer, ctx, user_prompt


def answer_query(resources: RAGResources, query: str, alpha: Optional[float] = None, top_k: Optional[int] = None, fetch_k: Optional[int] = None, system_prompt: Optional[str] = None):
    retrieved = retrieve(resources, query, alpha=alpha, top_k=top_k, fetch_k=fetch_k)
    answer, ctx, user_prompt = generate(resources, query, retrieved, system_prompt=system_prompt)
    return answer, retrieved, ctx, user_prompt