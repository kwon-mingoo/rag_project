import os
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 모델 경로 (직접 다운로드 받은 GGUF 파일 경로)
# HuggingFace에서 'Qwen/Qwen2.5-72B-Instruct-GGUF' 검색 후 q4_k_m.gguf 다운로드 추천
MODEL_PATH = "./models/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
DB_PATH = "./vector_db"

def get_llm():
    # RTX 5090을 위한 LlamaCpp 설정
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=50,        # -1: 가능한 모든 레이어를 GPU에 올림 (필수!), LLM이 크다면 어느정도 수치를 잡아야 함
        n_batch=512,
        n_ctx=4096,             # 문맥 길이 (필요에 따라 8192 등으로 증가)
        f16_kv=True,            # 메모리 절약
        verbose=True,
        temperature=0.1,        # 정답이 중요한 업무이므로 창의성 낮춤
        max_tokens=2048,        # 답변 길이 제한
    )
    return llm

def get_rag_chain():
    # 1. 벡터 DB 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'}
    )
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Retriever 설정 (유사도 높은 3개 문서 참조)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3. 프롬프트 템플릿 (Qwen/Construction DX 최적화)
    template = """
    당신은 건설 회사의 DX(디지털 전환) 및 행정 업무를 지원하는 유능한 AI 어시스턴트입니다.
    아래의 [참고 문서]를 바탕으로 질문에 대해 정확하고 전문적으로 답변하세요.
    문서에 없는 내용은 지어내지 말고 "정보가 없습니다"라고 답하세요.

    [참고 문서]:
    {context}

    [질문]:
    {question}

    [답변]:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4. LLM 로드
    llm = get_llm()

    # 5. 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, # 출처 확인용
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain