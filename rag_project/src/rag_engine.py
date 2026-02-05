import os
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 주의: 이 코드는 'rag_project' 폴더 안에서 실행해야 작동합니다.
MODEL_PATH = "./models/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"
DB_PATH = "./vector_db"

def get_llm():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        # 성능
        n_gpu_layers=-1,        # 모든 연산을 GPU로 처리 (속도 최적화)
        n_batch=512,            # 한 번에 처리할 데이터 양
        n_ctx=8192,             # 최대 문맥 길이 (질문+문서+답변)
        f16_kv=True,            # 메모리 절약 (성능 유지)
        verbose=True,           # 실행 로그 출력

        # 답변 스타일
        temperature=0.1,        # 창의성 억제 (사실 위주 답변)
        top_p=0.9,              # 뚱딴지같은 단어 방지
        repeat_penalty=1.2,     # 같은 말 반복 금지
        stop=["[질문]:", "Question:", "\n\n", "User:", "---"] # 답변 끝나면 즉시 멈춤
    )
    return llm

# src/rag_engine.py (수정본)

def get_rag_chain():
    # 1. 벡터 DB 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'}
    )
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"벡터 DB가 안 보임.")

    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 2. 프롬프트
    template = """
    당신은 건설 안전 및 DX 전문가입니다. [참고 문서]를 보고 질문에 핵심만 답하세요.
    문서에 없으면 모른다고 하고, 설명을 마치면 즉시 멈추세요.

    [참고 문서]:
    {context}

    [질문]:
    {question}

    [답변]:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 3. 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        
        # [핵심 수정] 이 줄이 빠져서 에러가 났던 겁니다! 다시 추가했습니다.
        return_source_documents=True, 
        
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain