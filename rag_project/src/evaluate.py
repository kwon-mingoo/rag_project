import os
import re
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from rag_engine import get_rag_chain, get_llm

# NLTK 데이터 확인
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ==========================================
# 평가 데이터셋 설정
# 주의: Recall@k와 MRR을 측정하려면 'relevant_docs'에 정답 파일명을 적어야 합니다.
# ==========================================
EVAL_DATASET = [
    {
        "question": "건설 현장에서 추락 사고 예방을 위한 조치는?",
        "ground_truth": "안전난간 설치, 추락 방호망 설치, 안전대 착용이 필수적입니다.",
        "relevant_docs": ["소규모 건설공사 안전관리 매뉴얼(2023).pdf"] # 정답이 포함된 실제 파일명
    },
    {
        "question": "안전관리비의 사용 가능 항목은?",
        "ground_truth": "안전관리자 인건비, 안전 시설비, 개인 보호구 구매비 등이 포함됩니다.",
        "relevant_docs": ["건설공사 안전관리계획서 작성 매뉴얼.pdf"]
    }
]

class RAGEvaluator:
    def __init__(self):
        print("평가 시스템 초기화 중...")
        self.rag_chain = get_rag_chain()
        self.llm = get_llm()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda'}
        )

    def parse_score(self, response_text):
        """LLM 답변에서 0 또는 1만 추출"""
        numbers = re.findall(r'\b[0-1]\b', response_text)
        if numbers:
            return int(numbers[0])
        return 0

    def calculate_metrics(self, generated_text, reference_text):
        # 1. BLEU
        ref_tokens = nltk.word_tokenize(reference_text)
        gen_tokens = nltk.word_tokenize(generated_text)
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=SmoothingFunction().method1)
        
        # 2. Semantic Similarity
        emb = self.embedding_model.embed_documents([generated_text, reference_text])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        
        return bleu, sim

    def evaluate_llm_metric(self, prompt_text):
        try:
            response = self.llm.invoke(prompt_text)
            return self.parse_score(response)
        except:
            return 0

    # ==========================================
    # 신규 추가: 검색 성능 지표 (Recall@k, MRR)
    # ==========================================
    def calculate_retrieval_metrics(self, retrieved_docs, relevant_docs_list):
        if not relevant_docs_list:
            return 0.0, 0.0

        # 검색된 문서들의 파일명(source) 추출
        # 경로가 포함되어 있을 수 있으므로 파일명만 추출해서 비교하거나 부분 일치 확인
        retrieved_sources = [os.path.basename(doc.metadata.get('source', '')) for doc in retrieved_docs]
        
        # 1. Recall@k 계산
        hits = 0
        for true_doc in relevant_docs_list:
            for ret_doc in retrieved_sources:
                if true_doc in ret_doc: # 부분 일치 허용
                    hits += 1
                    break
        recall_at_k = hits / len(relevant_docs_list)

        # 2. MRR 계산
        mrr = 0.0
        for rank, ret_doc in enumerate(retrieved_sources):
            for true_doc in relevant_docs_list:
                if true_doc in ret_doc:
                    mrr = 1.0 / (rank + 1)
                    return recall_at_k, mrr # 첫 번째 정답 발견 시 종료
        
        return recall_at_k, mrr

    def run_evaluation(self, dataset):
        print(f"\n총 {len(dataset)}개의 질문 평가 시작...\n")
        
        # 결과 저장소
        results = {
            "bleu": [], "sem_sim": [], 
            "faith": [], "truth": [], "ctx_recall_llm": [],
            "recall_k": [], "mrr": []
        }

        for i, data in enumerate(dataset):
            q = data["question"]
            gt = data["ground_truth"]
            targets = data.get("relevant_docs", []) # 정답 파일명 리스트
            
            print(f"[{i+1}] 질문: {q}")
            
            # RAG 실행
            res = self.rag_chain.invoke(q)
            ans = res['result']
            docs = res['source_documents'] # 검색된 문서들
            
            # 검색된 문서 텍스트 합치기 (LLM 평가용)
            ctx_text = "\n".join([d.page_content[:200] for d in docs])

            # ---------------------------
            # 1. 생성 성능 평가 (답변 품질)
            # ---------------------------
            bleu, sim = self.calculate_metrics(ans, gt)
            
            # Faithfulness
            p_faith = f"""
            [문서]를 보고 [답변]이 사실인지 판단하세요.
            문서에 있는 내용이면 1, 없는 내용을 지어냈으면 0을 출력하세요. 설명은 하지 마세요.
            [문서]: {ctx_text}
            [답변]: {ans}
            점수(0 또는 1):"""
            faith = self.evaluate_llm_metric(p_faith)

            # Truthfulness
            p_truth = f"""
            [정답]과 비교하여 [답변]이 정확한지 판단하세요.
            핵심 내용이 맞으면 1, 틀리면 0을 출력하세요. 설명은 하지 마세요.
            [정답]: {gt}
            [답변]: {ans}
            점수(0 또는 1):"""
            truth = self.evaluate_llm_metric(p_truth)

            # LLM Context Recall (답변 가능 여부)
            p_ctx_recall = f"""
            [문서] 안에 [정답]에 대한 내용이 포함되어 있나요?
            있으면 1, 없으면 0을 출력하세요.
            [문서]: {ctx_text}
            [정답]: {gt}
            점수(0 또는 1):"""
            ctx_recall = self.evaluate_llm_metric(p_ctx_recall)

            # ---------------------------
            # 2. 검색 성능 평가 (Recall@k, MRR)
            # ---------------------------
            recall_k, mrr = self.calculate_retrieval_metrics(docs, targets)

            # 결과 저장
            results["bleu"].append(bleu)
            results["sem_sim"].append(sim)
            results["faith"].append(faith)
            results["truth"].append(truth)
            results["ctx_recall_llm"].append(ctx_recall)
            results["recall_k"].append(recall_k)
            results["mrr"].append(mrr)

            # 개별 결과 출력 (이모지 제거)
            print(f"   -> 답변: {ans[:30]}...")
            print(f"   -> 생성지표: Sim={sim:.2f}, Faith={faith}, Truth={truth}")
            print(f"   -> 검색지표: Recall@k={recall_k:.2f}, MRR={mrr:.2f}")

        # 최종 리포트 출력
        print("\n" + "="*40)
        print("최종 평가 리포트")
        print("="*40)
        print("[생성 품질 지표]")
        print(f"BLEU Score        : {np.mean(results['bleu']):.4f} (참고용)")
        print(f"Semantic Sim      : {np.mean(results['sem_sim']):.4f} (목표: 0.7 이상)")
        print(f"Faithfulness      : {np.mean(results['faith']):.2f} (목표: 0.9 이상)")
        print(f"Truthfulness      : {np.mean(results['truth']):.2f} (목표: 0.8 이상)")
        print("-" * 40)
        print("[검색 품질 지표]")
        print(f"Context Recall(LLM): {np.mean(results['ctx_recall_llm']):.2f} (답변 가능성)")
        print(f"Recall@k (Retrieval): {np.mean(results['recall_k']):.2f} (문서 발견율)")
        print(f"MRR                 : {np.mean(results['mrr']):.4f} (상위 노출도)")
        print("="*40)

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation(EVAL_DATASET)