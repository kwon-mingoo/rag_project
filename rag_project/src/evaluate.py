import os
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# 기존 RAG 엔진 가져오기
from rag_engine import get_rag_chain, get_llm

# NLTK 데이터 다운로드 (최초 1회 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================================
# 1. 평가용 데이터셋 (Ground Truth)
# ==========================================
# 실제 평가할 때는 이 내용을 엑셀에서 불러오거나 더 늘려야 합니다.
EVAL_DATASET = [
    {
        "question": "건설 현장에서 추락 사고 예방을 위한 조치는?",
        "ground_truth": "추락 사고 예방을 위해 안전난간 설치, 추락 방호망 설치, 그리고 작업자 안전대 착용이 필수적입니다."
    },
    {
        "question": "안전관리비의 사용 가능 항목은 무엇인가요?",
        "ground_truth": "안전관리자 인건비, 안전 시설비, 개인 보호구 구매비, 안전 교육비 등이 포함됩니다."
    }
]

# ==========================================
# 2. 평가 지표 계산 함수들
# ==========================================

class RAGEvaluator:
    def __init__(self):
        print("평가 시스템 초기화 중...")
        self.rag_chain = get_rag_chain()
        self.llm = get_llm() # 심사위원(Judge)으로 사용할 LLM
        
        # 의미적 유사도 계산을 위한 임베딩 모델 (기존과 동일한 것 사용)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda'} # GPU 사용
        )

    def calculate_bleu(self, generated_text, reference_text):
        """BLEU 점수 계산 (단어 일치도)"""
        ref_tokens = nltk.word_tokenize(reference_text)
        gen_tokens = nltk.word_tokenize(generated_text)
        # SmoothingFunction은 짧은 문장에서 점수가 0이 되는 것을 방지
        score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=SmoothingFunction().method1)
        return score

    def calculate_semantic_similarity(self, generated_text, reference_text):
        """의미적 유사도 (코사인 유사도)"""
        embeddings = self.embedding_model.embed_documents([generated_text, reference_text])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return score

    def evaluate_with_llm(self, prompt_text):
        """LLM에게 점수를 매기게 하는 함수"""
        try:
            response = self.llm.invoke(prompt_text)
            # 답변에서 숫자만 추출 (예: "점수는 5점입니다" -> 5)
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                return int(numbers[0])
            else:
                return 0 # 파싱 실패 시 0점 처리
        except Exception as e:
            print(f" LLM 평가 중 오류: {e}")
            return 0

    def calculate_faithfulness(self, context, answer):
        """Faithfulness: 답변이 문서에 있는 내용으로만 작성되었는가?"""
        prompt = f"""
        당신은 공정한 평가자입니다. 아래 [참고 문서]와 [AI 답변]을 읽고 평가하세요.
        
        [참고 문서]:
        {context}
        
        [AI 답변]:
        {answer}
        
        평가 기준:
        - AI 답변이 [참고 문서]에 있는 내용에 기반하고 있으면 1점
        - 문서에 없는 내용을 지어냈으면 0점
        
        결과를 0 또는 1 숫자 하나로만 답하세요.
        """
        return self.evaluate_with_llm(prompt)

    def calculate_truthfulness(self, ground_truth, answer):
        """Truthfulness: 정답(Ground Truth)과 비교했을 때 사실 관계가 맞는가?"""
        prompt = f"""
        당신은 건설 전문가입니다. [정답]과 [AI 답변]을 비교하여 정확성을 평가하세요.
        
        [정답]:
        {ground_truth}
        
        [AI 답변]:
        {answer}
        
        평가 기준:
        - AI 답변이 정답의 핵심 내용을 모두 포함하고 정확하면 1점
        - 틀린 내용이 있거나 핵심이 빠졌으면 0점
        
        결과를 0 또는 1 숫자 하나로만 답하세요.
        """
        return self.evaluate_with_llm(prompt)

    def calculate_context_recall(self, context, ground_truth):
        """Context Recall: 검색된 문서 안에 정답을 맞힐 수 있는 정보가 있었는가?"""
        prompt = f"""
        [참고 문서] 안에 [정답]을 유추할 수 있는 정보가 포함되어 있는지 확인하세요.
        
        [참고 문서]:
        {context}
        
        [정답]:
        {ground_truth}
        
        평가 기준:
        - 문서만 보고 정답을 맞힐 수 있으면 1점
        - 문서에 관련 내용이 없으면 0점
        
        결과를 0 또는 1 숫자 하나로만 답하세요.
        """
        return self.evaluate_with_llm(prompt)

    # ==========================================
    # 3. 전체 평가 실행
    # ==========================================
    def run_evaluation(self, dataset):
        print(f"\n총 {len(dataset)}개의 질문에 대해 평가를 시작합니다...\n")
        
        results = {
            "bleu": [],
            "semantic_similarity": [],
            "faithfulness": [],
            "truthfulness": [],
            "context_recall": []
        }

        for i, data in enumerate(dataset):
            question = data["question"]
            ground_truth = data["ground_truth"]
            
            print(f"[{i+1}/{len(dataset)}] 질문 처리 중: {question}")

            # 1. RAG 실행
            response = self.rag_chain.invoke(question)
            generated_answer = response['result']
            
            # 검색된 문서 내용 합치기
            retrieved_docs = response['source_documents']
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])

            # 2. 지표 계산
            # (1) BLEU
            bleu = self.calculate_bleu(generated_answer, ground_truth)
            
            # (2) Semantic Similarity
            sem_sim = self.calculate_semantic_similarity(generated_answer, ground_truth)
            
            # (3) Faithfulness (환각 여부)
            faith = self.calculate_faithfulness(context_text, generated_answer)
            
            # (4) Truthfulness (정답 일치 여부)
            truth = self.calculate_truthfulness(ground_truth, generated_answer)
            
            # (5) Context Recall (검색 품질)
            recall = self.calculate_context_recall(context_text, ground_truth)

            # 결과 저장
            results["bleu"].append(bleu)
            results["semantic_similarity"].append(sem_sim)
            results["faithfulness"].append(faith)
            results["truthfulness"].append(truth)
            results["context_recall"].append(recall)

            print(f"   -> BLEU: {bleu:.4f} | Sim: {sem_sim:.4f} | Faith: {faith} | Truth: {truth} | Recall: {recall}")

        # 평균 점수 출력
        print("\n" + "="*40)
        print("최종 평가 리포트")
        print("="*40)
        print(f"1. BLEU (단어 일치도)      : {np.mean(results['bleu']):.4f}")
        print(f"2. Sem Sim (의미 유사도)   : {np.mean(results['semantic_similarity']):.4f}")
        print(f"3. Faithfulness (신뢰성)   : {np.mean(results['faithfulness']):.2f} / 1.0")
        print(f"4. Truthfulness (정확성)   : {np.mean(results['truthfulness']):.2f} / 1.0")
        print(f"5. Context Recall (검색력) : {np.mean(results['context_recall']):.2f} / 1.0")
        print("="*40)

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation(EVAL_DATASET)