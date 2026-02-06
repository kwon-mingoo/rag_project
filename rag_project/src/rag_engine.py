import os
import glob
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# [신규 추가] Hybrid Search용 라이브러리
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "models","Llama","Meta-Llama-3.1-8B-Instruct.Q8_0.gguf")
DB_PATH = os.path.join(project_root, "vector_db")
DATA_PATH = os.path.join(project_root, "data") # 원본 문서 경로

def get_llm():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=8192,
        f16_kv=True,
        verbose=True,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.2,
        stop=["[질문]:", "Question:", "\n\n", "User:", "---"]
    )
    return llm

def load_documents_for_bm25():
    """
    BM25 검색을 위해 data 폴더의 원본 문서를 다시 로드합니다.
    (FAISS는 벡터만 저장하므로, 키워드 검색을 위해 원문 텍스트가 필요합니다)
    """
    print("BM25(키워드 검색) 인덱스 생성 중...")
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    docs = []
    
    if not pdf_files:
        print("경고: data 폴더에 PDF 파일이 없습니다. BM25가 제대로 작동하지 않을 수 있습니다.")
        return []

    # 1. 문서 로드
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # 2. 텍스트 분할 (Ingest와 동일한 설정 권장)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

def get_rag_chain():
    # 1. 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'}
    )
    
    # 2. FAISS (벡터 검색기) 로드
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("벡터 DB가 없습니다. 먼저 ingest.py를 실행하세요.")
    
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 3. BM25 (키워드 검색기) 생성
    # 문서를 로드해서 인덱스를 만듭니다.
    bm25_docs = load_documents_for_bm25()
    
    if bm25_docs:
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 5  # BM25도 5개 검색
        
        # 4. 앙상블 (Hybrid) 검색기 생성
        # weights=[0.5, 0.5] -> 키워드 검색 50%, 의미 검색 50% 반영
        # 건설 용어가 중요하다면 BM25 비중을 0.6~0.7로 높여도 좋습니다.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        final_retriever = ensemble_retriever
        print("Hybrid Search(BM25 + FAISS)가 적용되었습니다.")
    else:
        # 문서 로드 실패 시 FAISS만 사용
        final_retriever = faiss_retriever
        print("경고: BM25 생성 실패. FAISS만 사용합니다.")

    # 5. 프롬프트 설정
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

    # 6. 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=final_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain