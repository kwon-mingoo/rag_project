# rag_project --- Hybrid RAG (BM25 + FAISS) + llama.cpp (GGUF)

문서(PDF) 기반 Retrieval-Augmented Generation(RAG) 시스템입니다.

-   Retrieval: BM25 + FAISS Hybrid Search
-   Generation: llama-cpp-python (로컬 GGUF 모델)
-   Evaluation: BLEU / Semantic Similarity + LLM Judge 기반 정량
    평가
-   목표: 법령·가이드·사내 문서 기반 고신뢰 QA 시스템 구축

------------------------------------------------------------------------

# Repository Structure

    rag_project/
    ├─ data/                # RAG 대상 원문 데이터 (PDF 등) (github 미포함)
    ├─ eval_data/           # 평가용 질문-정답 CSV (github 미포함)
    ├─ models/              # 로컬 GGUF 모델 파일 (github 미포함)
    ├─ VectorDB/            # 생성된 벡터DB 파일 (자동 생성)
    ├─ main.py/             # 챗봇 시스템
    ├─ src/
    │  ├─ build_vectordb.py # VectorDB 생성
    │  ├─ rag_engine.py     # RAG 통합 엔진 (검색 + 생성)
    │  ├─ rag_run.py        # 단일 질의 실행 CLI
    │  └─ evaluate.py       # 배치 평가 실행
    └─ requirements.txt

------------------------------------------------------------------------

# System Architecture

1.  User Query 입력\
2.  Hybrid Retrieval (BM25 + FAISS)\
3.  Top-k Chunk 선택\
4.  llama.cpp (GGUF) 기반 답변 생성

------------------------------------------------------------------------

# Quick Start

## 환경 설정

``` bash
RTX 5090 기준
python -m venv .venv
source .venv/bin/activate
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python llama-cpp-python
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
```

------------------------------------------------------------------------

## VectorDB 생성

``` bash
python src/build_vectordb.py
```

생성 결과:

    VectorDB/
     ├─ docs.pkl
     ├─ bm25.pkl
     ├─ faiss.index
     └─ config.json

------------------------------------------------------------------------

## 단일 질문 실행

``` bash
python src/rag_run.py --query "산업안전보건기준에 관한 규칙 제291조는?"
```

------------------------------------------------------------------------

## 평가 실행

``` bash
python src/evaluate.py
```

결과 파일:

    eval_data/eval_report.json

------------------------------------------------------------------------

# Evaluation Metrics

## Generation

-   BLEU\
-   Semantic Similarity\
-   Faithfulness (Judge)\
-   Truthfulness (Judge)

## Retrieval

-   Recall@k\
-   MRR\
-   Context Recall

------------------------------------------------------------------------

# 주요 설정 (RAGConfig)

-   vectordb_dir/
-   embedding_model/
-   alpha (Hybrid 가중치)/
-   top_k / fetch_k/
-   gguf_path/
-   prompt_mode (prod / eval)

------------------------------------------------------------------------
# Troubleshooting
# 1. VectorDB 파일이 없음

→ python src/build_vectordb.py 먼저 실행

# 2. Faithfulness가 0.0

→ judge_max_tokens 증가 (1024 이상)

# 3. BLEU 낮음

→ prompt_mode=eval 사용
→ 답변 길이 제한

# 4. 속도가 느림

→ n_ctx 줄이기
→ fetch_k 조정
→ GPU 활성화 확인

# 5. 지표 평가

------------------------------------------------------------------------

# 목표

-   문서 기반 근거 제시
-   환각 최소화
-   재현 가능한 평가
-   협업 가능한 구조
