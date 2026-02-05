import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 경로 설정
DATA_PATH = "./data"
DB_PATH = "./vector_db"

def run_ingest():
    print("📂 문서를 로드 중입니다...")
    
    # PDF 및 텍스트 파일 로드
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    
    documents = []
    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())

    if not documents:
        print("❌ 데이터 폴더에 문서가 없습니다.")
        return

    print(f"✅ 총 {len(documents)}개의 문서를 로드했습니다. 청킹(Chunking)을 시작합니다...")

    # 문서 분할 (건설 문서는 문맥이 중요하므로 청크 사이즈를 넉넉하게 잡음)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 임베딩 모델 로드 (한국어 성능 우수: ko-sroberta)
    print("🧠 임베딩 모델을 로드 중입니다 (jhgan/ko-sroberta-multitask)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'} # 5090 사용
    )

    # 벡터 DB 생성 및 저장
    print("💾 벡터 DB를 생성하고 저장합니다...")
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(DB_PATH)
    print("🎉 완료! 벡터 DB가 저장되었습니다.")

if __name__ == "__main__":
    run_ingest()