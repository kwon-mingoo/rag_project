# Construction Safety Hybrid RAG (BM25 + FAISS + llama.cpp)

건설 안전/법령 PDF들을 대상으로 **BM25 + FAISS(Vector) 하이브리드 검색**을 수행하고,  
로컬 GGUF 모델(Meta-Llama-3.1-8B-Instruct.Q8_0.gguf)을 llama.cpp(=llama-cpp-python)로 직접 구동해 답변을 생성하는 프로젝트입니다.

 `models/` 폴더에 모델을 내려받아 넣고, `src/evaluate.py`의 `gguf_path`만 맞추면 바로 실행됩니다.

---

## 1) 폴더 구조 (상세)

```
rag_project_v1/
├─ main.py
├─ requirements.txt
├─ .env
├─ data/
│  └─ *.pdf
├─ src/
│  ├─ build_vectordb.py
│  ├─ rag_run.py
│  └─ evaluate.py
├─ VectorDB/
│  ├─ docs.pkl
│  ├─ bm25.pkl
│  ├─ faiss.index
│  └─ config.json
├─ eval_data/
│  ├─ eval_dataset_rag_queries.csv
│  ├─ retrieval_debug.jsonl
│  └─ eval_report.json
└─ models/
   └─ (여기에 GGUF 모델 파일을 직접 넣습니다. GitHub 미포함)
```

### 각 폴더/파일 역할

- **`data/`**  
  RAG 대상 원문 PDF 저장 폴더. `build_vectordb.py`가 이 폴더를 읽어 인덱스를 만듭니다.

- **`VectorDB/`** *(자동 생성/캐시)*  
  검색에 필요한 인덱스/문서 메타데이터 저장.
  - `docs.pkl`: chunk 단위 문서 리스트(텍스트 + source/page/chunk_id 메타)
  - `bm25.pkl`: BM25 인덱스
  - `faiss.index`: FAISS 벡터 인덱스
  - `config.json`: 생성 당시 설정(임베딩 모델명, chunk 설정 등)

- **`eval_data/`**  
  평가용 데이터와 산출물
  - `eval_dataset_rag_queries.csv`: 평가 질의/정답/근거 메타
  - `retrieval_debug.jsonl`: retrieval 디버깅 로그(옵션 저장)
  - `eval_report.json`: 평가 결과 리포트(JSON)

- **`src/build_vectordb.py`**  
  PDF → 텍스트 로드 → chunking → 임베딩 생성 → FAISS 저장 + BM25 저장.

- **`src/rag_run.py`**  
  단발성 질의 실행(검색 + 생성). `--query`로 한 번 질의하고 종료합니다.

- **`src/evaluate.py`**  
  평가 스크립트(검색 성능 + 생성 품질).  
  내부에 **운영용 프롬프트(SYSTEM_PROMPT/build_user_prompt)** 와 **llama.cpp 호출** 로직이 포함되어 있습니다.

- **`main.py`**  
  운영용 진입점(대화형). VectorDB가 없으면 `build_vectordb.py`를 먼저 실행한 뒤, 계속 질의할 수 있습니다.

---

## 2) 모델(GGUF) 준비 (GitHub에 포함하지 않음)

예시 디렉토리(권장):

```
models/llama8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf
```

그리고 `src/evaluate.py`(및 `src/build_vectordb.py`)의 설정에서 다음 값을 맞춥니다.

```python
# src/evaluate.py (EvalConfig)
gguf_path: str = "/home/nami/Desktop/rag_project_v1/models/llama8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"
```

> 로컬 경로가 다르면 **반드시 수정**하세요.  
> (운영/평가 모두 동일 파일을 참조합니다.)

---

## 3) 설치

### (1) Python 환경
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### (2) FAISS (GPU 권장)
- conda 사용 시:
```bash
conda install -c pytorch faiss-gpu
```

### (3) llama-cpp-python (CUDA/GPU 사용)
본 프로젝트는 `llama_cpp.Llama`를 사용합니다.  
GPU를 쓰려면 **CUDA 지원 빌드/휠**이 필요합니다.

- 이미 GPU가 정상 동작한다면(현재 상태) 추가 설정 없이 진행하면 됩니다.
- 만약 CPU로만 잡히면, `llama-cpp-python`을 CUDA 지원으로 재설치해야 합니다.

> 참고: HuggingFace에서 임베딩 모델을 내려받기 때문에 첫 실행 시 다운로드 시간이 걸릴 수 있습니다.  
> rate limit/속도 문제를 줄이려면 환경변수 `HF_TOKEN`을 설정하세요.

---

## 4) VectorDB 생성 (BM25 + FAISS 저장)

```bash
python src/build_vectordb.py
```

정상 완료 후 `VectorDB/` 폴더에 아래 파일들이 생성됩니다:

- `VectorDB/docs.pkl`
- `VectorDB/bm25.pkl`
- `VectorDB/faiss.index`
- `VectorDB/config.json`

---

## 5) 운영 실행 (대화형)

```bash
python main.py
```

옵션:
- `--show_topk` : 검색 Top-k 메타(문서/페이지/스코어) 출력
- `--show_context` : 컨텍스트 전문 출력
- `--alpha 0.6 --top_k 8 --fetch_k 60` : 하이브리드/검색 파라미터 오버라이드

예:
```bash
python main.py --show_topk
```

---

## 6) 단발 실행 (rag_run.py)

```bash
python src/rag_run.py --query "가설울타리 계획변경 시 안전관리계획 수립 대상인가?"
```

디버그 옵션:
```bash
python src/rag_run.py --query "산업안전보건기준에 관한 규칙 제291조는?" --show_topk --show_context
```

---

## 7) 평가 실행 (evaluate.py)

```bash
python src/evaluate.py --max_rows 50 --debug_n 3
```

- `eval_data/eval_dataset_rag_queries.csv`를 입력으로 사용하고,
- `eval_data/eval_report.json`에 리포트가 저장됩니다.

---

## 8) GitHub 업로드 권장 설정 (.gitignore)

모델과 인덱스는 용량이 커질 수 있어 GitHub에는 보통 제외합니다.

```gitignore
# 모델(매우 큼)
models/**/*.gguf

# VectorDB 캐시(필요시 제외)
VectorDB/
*.index
*.pkl

# 파이썬 캐시
__pycache__/
*.pyc
.venv/
```


---

## 9) 자주 발생하는 이슈

### (1) HF Hub 경고 (unauthenticated)
```
Warning: You are sending unauthenticated requests to the HF Hub...
```
- `HF_TOKEN` 환경변수를 설정하면 다운로드 속도/제한이 개선됩니다.

### (2) 모델 출력이 깨지는 경우(한글 난독/반복)
- `gguf_path`가 실제 파일과 맞는지 확인
- `n_ctx`, `temperature`, `max_tokens` 조정
- 너무 긴 컨텍스트를 넣지 않도록 `top_k`, chunk 길이 조절
- Q8_0/quant 방식 변경(Q4_K_M 등)도 품질에 영향

---

## 10) 라이선스/주의
- 입력 PDF 및 생성 결과의 사용/배포는 각 문서의 저작권/공개 범위를 준수하세요.
- 본 프로젝트는 “문서 근거 기반 답변”을 지향하지만, 최종 법적 판단은 전문가 검토가 필요합니다.
