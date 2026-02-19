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

from collections import defaultdict

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
        report_type = "qa_short"
        #report_type = "checklist"

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
당신은 법령/기준에 근거하여 대외 발송용 공식 공문(안)을 작성하는 문서 담당자입니다.
본 문서는 "법률 검토 보고서"가 아니라 "행정 통보 문서"입니다.
과도한 법령 해석·형사처벌 분석은 하지 않습니다.

[질의]
{question}

[컨텍스트]
{context_blocks}

━━━━━━━━━━━━━━━━━━━━━━━━━━
[최우선 원칙 – 행정 문서 모드]
━━━━━━━━━━━━━━━━━━━━━━━━━━
1) 공문은 "사실 통보 + 계획 보고 + 협조 요청" 중심으로 작성합니다.
2) 법령은 “의무 존재” 수준까지만 언급합니다.
3) 조문번호·형벌·벌금·과태료 액수는
   컨텍스트에 법령 원문이 직접 인용되어 있지 않으면 작성하지 않습니다.
4) 사례집·가이드라인은 “운영상 참고 근거”로만 사용합니다.
5) 컨텍스트에 없는 날짜·기관명·수신처·금액·조문번호는 절대 생성하지 않습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[법령 언급 통제 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━
- 단순히 "법률명 + 제○조"가 텍스트에 등장했다는 이유만으로
  조문번호를 반복 확정하지 않습니다.
- 법령 원문이 직접 인용되어 있지 않으면:
  → "관련 법령에 따라"
  → "관련 기준에 근거하여"
  와 같이 포괄 표현을 사용합니다.
- 형사처벌·벌금·과태료 수치는 원문이 없으면 작성하지 않습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[작성 절차]
━━━━━━━━━━━━━━━━━━━━━━━━━━
1) 근거 목록 추출 (요약 + 짧은 원문 발췌 + 출처)
   - 근거 유형을 함께 표시:
     (법령 원문 / 시행령·시행규칙 / 가이드라인·사례집)

2) 필수/권고 요건 표 작성
   - "법적 의무"와 "운영상 권고"를 구분
   - 제재 수위는 포함하지 않음 (원문 없으면 금지)

3) 위 표의 내용만 사용하여 공문 작성
   - 표에 없는 내용은 공문에 쓰지 않음
   - 법령 분석 문장은 최소화

━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 형식]
━━━━━━━━━━━━━━━━━━━━━━━━━━

## 0) 근거 목록(발췌)
- (근거 유형) (요약) "원문 일부..." [출처]

## 1) 요건 정리
| 구분 | 내용 | 근거 |
|------|------|------|
| 필수 | ... | [1] |
| 권고 | ... | [2] |

## 2) 공문(안)

문서번호: (비워두기)
시행일자: (비워두기)
수    신: (비워두기)
참    조: (비워두기)
제    목: (질의 반영 제목)

1. 배경 및 목적
- 관련 법령 및 기준에 따라 … 실시하고자 하며, 이에 그 계획을 보고드립니다. [출처]

2. 계획 내용
- (필수) … 이행 예정입니다. [출처]
- (권고) … 반영할 계획입니다. [출처]

3. 협조 요청 사항
- …에 대한 확인 및 협조를 요청드립니다. [출처]

4. 유의사항 / 근거 부족
- 구체 제재 수위·조문번호 등은 법령 원문 확인 필요. [근거 부족 시 기재]

끝.

"""

    if report_type == "doc_draft":
        return f"""[역할]
당신은 건설 현장 문서 초안을 작성하는 실무자입니다.
이 문서는 "불완전한 초안"이며,
정보가 부족한 항목은 보완하지 않고 그대로 비워둡니다.
추정·보완·상식적 추론 작성은 허용되지 않습니다.

[요청]
{question}

[컨텍스트]
{context_blocks}

━━━━━━━━━━━━━━━━━━━━━━━━━━
[허용 정보 선언]
━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 본 문서에서 사용 가능한 정보는 다음 두 출처뿐이다:
   - 사용자 질문에 명시된 정보
   - 컨텍스트에 명시된 정보

2. 위 두 출처에 없는 정보는 생성하지 않는다.

3. 정보가 없을 경우 반드시 정확히 다음 문자열만 사용한다:
   → [현장 입력 필요]

4. 다음 정보는 사용자 또는 컨텍스트에 정확히 있을 때만 작성 가능:
   - 날짜 / 기간
   - 인원 수
   - 층수 / 높이
   - 장비 모델명
   - 발주처
   - 법령명 및 조항번호
   - 과태료·벌금·형량 등 구체 수치

5. 다음은 절대 생성하지 않는다:
   - 사용자 입력에 없는 추가 인원 수
   - 사고 발생 사실 (사망, 중상, 사고 사례)
   - 과거 판례·행정처분 사례
   - 구체 금액·형량
   - 법령명 추론 생성

6. 인원 수는 사용자 입력값만 사용한다.
   총 인원을 추론하여 보완하지 않는다.
   명시되지 않은 경우:
   → [현장 입력 필요]

━━━━━━━━━━━━━━━━━━━━━━━━━━
[위험도 작성 규칙 - 숫자 생성 차단]
━━━━━━━━━━━━━━━━━━━━━━━━━━

- 위험도는 구체 숫자를 생성하지 않는다.
- 반드시 다음 형식 중 하나만 사용한다:

  F? / S? / R? [현장 확인 필요]
  또는
  [현장 확인 필요]

- 계산값 생성 금지.
- 1~5 범위 숫자 생성 금지.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[법적 의무 작성 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 법적 의무/기준은 컨텍스트에 동일 문구가 있을 때만 작성한다.
2. 정확한 법령명 또는 조항이 확인되지 않으면:
   → [근거 부족]

3. 구체 금액·형량은 컨텍스트에 동일 수치가 있을 때만 작성한다.
4. 수치가 없으면 반드시 다음 문장만 사용한다:
   → [구체 제재 수위/금액은 법령 원문 확인 필요]

━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 구조 - 수정 금지]
━━━━━━━━━━━━━━━━━━━━━━━━━━

# 위험성평가 및 작업계획서(초안)

## 1. 문서 기본정보
- 현장명: [현장 입력 필요]
- 공사명/발주처: [현장 입력 필요]
- 작업명(공종): [현장 입력 필요]
- 작업일시/구간: [현장 입력 필요]
- 작업장소(세부): [현장 입력 필요]
- 작성/검토/승인: [현장 입력 필요]

## 2. 작업 개요 및 절차
- 작업 목적/배경: ...
- 작업 범위: ...
- 작업 절차:
  1) ...
  2) ...
  3) ...

## 3. 투입 인원/장비/자재
- 인원(역할): ...
- 장비: ...
- 자재/공구: ...
- 주변 작업/간섭: [현장 입력 필요]

## 4. 위험요인 식별 및 위험성평가

| 작업단계 | 위험요인 | 원인/상황 | 현재 통제 | 위험도(전) | 저감대책(공학/관리/PPE) | 위험도(후) | 담당 |
|----------|----------|-----------|-----------|------------|-------------------------|------------|------|

※ 위험도는 반드시 위 규칙에 따른다.

## 5. 관련 법적 의무 요약
- 적용 법령/기준: ...
- 필수 이행사항: ...
- 기록/관리 의무: ...
- 미이행 시 책임 범주: ...

(출처 필수 / 없으면 [근거 부족])

## 6. 작업 전 점검 체크리스트
- [ ] 출입통제
- [ ] 교육 실시
- [ ] 장비 점검
- [ ] 위험요인 제거 확인
- [ ] 기타: [현장 입력 필요]

## 7. 작업중지 기준
- 즉시 중지: 중대한 위험 발생 시
- 재개 조건: 위험요인 제거 및 확인 후

## 8. 근거(인용)
- [1] ...
- [2] ...
- [3] ...

## 9. 추가 확인 필요
- [현장 입력 필요]

"""


    # =========================================================
    #  4. REVIEW REPORT (검토의견서 - 법령 QA 최적화)
    # =========================================================
    return f"""[역할]
당신은 법령/기준 준수 검토 의견서 작성자입니다.
오직 컨텍스트 근거만 사용합니다. 추정/상식/암기 기반 작성 금지.

[질의]
{question}

[컨텍스트]
{context_blocks}

━━━━━━━━━━━━━━━━━━━━━━━━━━
[최우선 원칙: 법령 근거 게이트]
━━━━━━━━━━━━━━━━━━━━━━━━━━
1) "법령/시행령/시행규칙/고시/조문" 수준의 원문 근거가 컨텍스트에 포함되어 있지 않으면,
   - 조문번호(제X조/항/호)
   - 형사처벌(징역/벌금)
   - 과태료/과징금 액수
   - 행정처분 종류(영업정지 등)의 법정 근거
   를 절대 작성하지 말고 반드시 다음으로 대체한다:
   → [근거 부족: 법령 원문/시행령/시행규칙 확인 필요]

2) 컨텍스트가 "가이드라인/매뉴얼/사례집/Q&A" 뿐이라면
   - 법적 결론(‘위반 시 ○○처벌’)을 단정하지 않는다.
   - 이 경우에는 “운영상 권고/체크포인트”로만 기술한다.
   - 제재는 반드시:
     → [구체 제재 수위/금액은 법령 원문 확인 필요]
     로 표기한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[절대 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━
- 컨텍스트에 없는 법령명/조문번호/수치(징역·벌금·과태료)를 생성하지 말 것.
- 질문에 법률명이 들어 있어도, 컨텍스트에 동일 법률명이 없으면 “법률명”을 그대로 반복하여 단정하지 말 것.
  (예: 컨텍스트에 '산업안전보건법'이 없으면 해당 명칭을 근거로 단정 금지 → [근거 부족])
- 모든 의무/요건/금지/제재 문장 끝에는 출처([1],[2]...)를 붙일 것.
- 출처를 붙일 수 없으면 그 문장은 작성하지 말고 [근거 부족]으로 남길 것.
- 가이드라인/매뉴얼/사례집은 “법적 근거”가 아니라 “운영상 참고”로만 사용한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[질의 처리 방식]
━━━━━━━━━━━━━━━━━━━━━━━━━━
- 질문에 특정 법률/기준이 명시된 경우:
  ① 컨텍스트에서 해당 법률/기준의 "원문(법령/시행령/시행규칙)"이 실제로 존재하는지 먼저 확인한다.
  ② 존재하면: 법령별로 의무/제재 체계를 정리한다.
  ③ 존재하지 않으면: 법률명/조문/제재 단정 없이, 컨텍스트 기반 “안전조치/관리체계/권고”로만 답하고,
     법률 원문 확인 필요성을 명시한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[작성 절차]
━━━━━━━━━━━━━━━━━━━━━━━━━━
0) 컨텍스트 근거의 "유형"을 먼저 분류해 1줄로 선언:
   - (법령 원문 포함 / 법령 원문 없음-가이드/사례만 있음)

1) 근거 5~15개 추출: (요약 + 짧은 원문 발췌 + 출처)
   - 발췌는 1~2문장 이내로 짧게.

2) A/B 요약
   A: 필수(의무/금지/요건)  ← 근거가 법령 원문이면 우선 여기에
   B: 권고/운영 체크포인트 ← 가이드/사례집/매뉴얼은 주로 여기에

3) (가능할 때만) 법령별 의무·제재 체계 정리
   - 법령 원문이 있을 때만 조문/제재를 구체화
   - 없으면 해당 섹션은 "[근거 부족]"으로 처리

4) 결론(P1~P3): 실행 가능한 조치만(근거 필수)

5) 근거 부족/추가 확인 질문: “무엇을 확인해야 결론이 확정되는지” 질문으로 작성

━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 형식]
━━━━━━━━━━━━━━━━━━━━━━━━━━
## -1) 근거 유형 선언
- ...

## 0) 근거 목록 (5~15개)
- (요약) "원문 일부..." [출처]

## 1) A/B 요약
### A. 필수(의무/금지/요건)
- ... [출처]
### B. 권고/운영 체크포인트
- ... [출처]

## 2) 법령별 의무·제재 체계 정리 (가능할 때만)
### 2-1) (법령/기준 A)
- 관련 근거(조문/문서): ... [출처]  (법령 원문 없으면 [근거 부족])
- 핵심 의무(요건): ... [출처 또는 근거 부족]
- 적용 범위/전제: ... [출처 또는 근거 부족]
- 위반 시 제재(구분):
  - (형사) ... [출처 또는 "근거 부족: 법령 원문 확인 필요"]
  - (과태료) ... [출처 또는 "근거 부족: 법령 원문 확인 필요"]
  - (행정처분/기타) ... [출처 또는 "근거 부족: 법령 원문 확인 필요"]

### 2-2) (법령/기준 B)
(동일 형식)

## 3) 차이점·중복·우선순위(실무 관점)
- 중복되는 의무: ... [출처 또는 근거 부족]
- 차이점: ... [출처 또는 근거 부족]
- 우선순위(현장 즉시조치 vs 관리체계): ... [출처 또는 근거 부족]

## 4) 결론(P1~P3)
- P1: ... [출처]
- P2: ... [출처]
- P3: ... [출처 또는 근거 부족]

## 5) 근거 부족 / 추가 확인 질문
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
    top_k: int = 4
    fetch_k: int = 150 #40       # 후보 풀

    # Chunking (build_vectordb에서 사용)
    chunk_size: int = 1400 #800
    chunk_overlap: int = 250 # 120

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

    mmr_enabled: bool = True
    mmr_lambda: float = 0.7          # 0.6~0.8 추천
    mmr_candidates: int = 80         # 후보에서 MMR 적용할 개수 

    # Prompt / Report
    prompt_mode: str = "prod"   # prod|eval
    report_type: str = "doc_draft"  # review_report|official_letter|qa_short|checklist

def get_retriever_signature(cfg: "RAGConfig") -> str:
    # “지표가 바뀌는 원인”이 되는 요소만 담기 (경로는 넣지 않기)
    parts = [
        "retriever=hybrid(bm25+faiss)",
        f"alpha={cfg.alpha}",
        f"top_k={cfg.top_k}",
        f"fetch_k={cfg.fetch_k}",
        f"embedding_model={cfg.embedding_model}",
        # 아래는 구현/정책을 명시 (build_vectordb / rag_engine 정책에 맞게 고정)
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

#MMR
def mmr_select(
    candidate_idxs,
    relevance_scores,   # dict: idx -> rel_score (float)
    get_vec,            # function idx -> vector (np.ndarray) or None
    top_k: int,
    lambda_: float = 0.7,
):
    selected = []
    selected_vecs = []

    # 미리 후보 벡터 확보 (없으면 그 후보는 다양성 계산에서 불리/불가)
    cand_vecs = {}
    for i in candidate_idxs:
        v = get_vec(i)
        if v is not None:
            cand_vecs[i] = v

    while len(selected) < top_k and candidate_idxs:
        best = None
        best_score = -1e9

        for i in candidate_idxs:
            rel = relevance_scores.get(i, 0.0)

            # 다양성 항: 선택된 것들과의 최대 유사도(없으면 0)
            div = 0.0
            vi = cand_vecs.get(i)
            if vi is not None and selected_vecs:
                # cosine (IndexFlatIP + 정규화면 dot = cosine)
                sims = [float(np.dot(vi, vj)) for vj in selected_vecs]
                div = max(sims) if sims else 0.0

            score = lambda_ * rel - (1.0 - lambda_) * div
            if score > best_score:
                best_score = score
                best = i

        if best is None:
            break

        selected.append(best)
        if best in cand_vecs:
            selected_vecs.append(cand_vecs[best])
        candidate_idxs.remove(best)

    return selected




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
        mmr_enabled: bool = False,
        mmr_candidates: int = 80,
        mmr_lambda: float = 0.7,
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

        # 후보를 넉넉히 확보
        cand_n = max(top_k, int(mmr_candidates))
        top_idx = np.argsort(-hybrid)[:cand_n]

        # MMR rerank 적용 (중복 억제 + 다양성)
        if bool(mmr_enabled) and len(top_idx) > top_k:
            rel_map = {int(i): float(hybrid[int(i)]) for i in top_idx}

            def _get_vec(doc_i: int):
                try:
                    v = self.index.reconstruct(int(doc_i))
                    v = np.asarray(v, dtype=np.float32)
                    n = np.linalg.norm(v) + 1e-12
                    return v / n
                except Exception:
                    return None

            cand = [int(i) for i in top_idx.tolist()]
            picked = mmr_select(
                cand, rel_map, _get_vec,
                top_k=top_k, lambda_=float(mmr_lambda)
            )
            top_idx = np.asarray(picked, dtype=np.int64)
        else:
            top_idx = top_idx[:top_k]

        # retrieved 구성 + dbg 생성 + return
        retrieved: List[Tuple[Document, float, Dict[str, Any]]] = []
        for i in top_idx:
            ii = int(i)
            doc = self.docs[ii]
            score = float(hybrid[ii])

            meta = getattr(doc, "metadata", {}) or {}
            dbg = {
                "doc_i": ii,
                "source": meta.get("source", ""),
                "source_basename": meta.get("source_basename", meta.get("source", "")),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id", ii),
                "bm25_n": float(bm25_n[ii]),
                "vec_n": float(vec_n[ii]),
                "hybrid": score,
            }
            retrieved.append((doc, score, dbg))

        return retrieved



def make_context_blocks(retrieved, max_per_page: int = 2):
    """
    Build context blocks from retrieved [(doc, score, dbg), ...]
    while limiting redundancy per (source,page).
    """
    per_page = defaultdict(int)
    seen_txt = set()
    kept = []

    for doc, score, dbg in retrieved:
        txt = (doc.page_content or "").strip()
        if not txt:
            continue
        # exact-dup 제거
        if txt in seen_txt:
            continue

        src = dbg.get("source_basename") or dbg.get("source") or ""
        page = dbg.get("page")
        key = (src, page)

        # 같은 (문서,페이지)에서 너무 많이 뽑히는 것 제한
        if per_page[key] >= max_per_page:
            continue

        per_page[key] += 1
        seen_txt.add(txt)
        kept.append((doc, score, dbg))

    blocks = []
    for doc, score, dbg in kept:
        src = dbg.get("source_basename") or dbg.get("source") or "unknown"
        page = dbg.get("page")
        blocks.append(
            f"[source={src} | page={page} | score={score:.4f}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)



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
        mmr_enabled=cfg.mmr_enabled,
        mmr_candidates=cfg.mmr_candidates,
        mmr_lambda=cfg.mmr_lambda,
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