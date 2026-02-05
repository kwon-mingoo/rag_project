import os
import time
from src.ingest import run_ingest
from src.rag_engine import get_rag_chain

def main():
    print("🏗️  건설 DX AI 시스템을 부팅합니다...")
    
    # DB가 없으면 생성 (최초 1회)
    if not os.path.exists("./vector_db"):
        print("⚠️  벡터 DB가 없습니다. 문서를 먼저 처리합니다.")
        run_ingest()

    # 체인 로드
    try:
        qa_chain = get_rag_chain()
        print("🚀 시스템 준비 완료! (종료하려면 'exit' 입력)")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("모델 파일 경로가 정확한지, models 폴더에 .gguf 파일이 있는지 확인하세요.")
        return

    while True:
        query = input("\n💬 질문을 입력하세요: ")
        if query.lower() in ["exit", "quit", "종료"]:
            break
        
        start_time = time.time()
        
        # 답변 생성
        result = qa_chain.invoke({"query": query})
        
        end_time = time.time()
        
        print(f"\n🤖 [답변] ({end_time - start_time:.2f}초 소요):")
        print(result['result'])
        
        print("\n📄 [참고 문서]:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']} (내용 일부: {doc.page_content[:50]}...)")

if __name__ == "__main__":
    main()