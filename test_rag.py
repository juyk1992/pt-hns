import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def test_kcg_rag_search():
    persist_directory = "./kcg_guide_chromadb"
    
    print("🧠 로컬 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("💾 저장된 Chroma Vector DB 로드 중...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # 테스트 검색어 (가이드북에 있을법한 키워드)
    test_query = "황산 유출 시 초동 대응 조치 및 개인 보호구"
    print(f"\n🔍 테스트 검색어: '{test_query}'\n" + "="*50)

    # 유사도 기반 상위 3개 검색 결과 추출
    docs = vectorstore.similarity_search(test_query, k=3)

    if not docs:
        print("❌ DB에서 검색된 결과가 없습니다. DB 생성을 확인하세요.")
        return

    print(f"🎉 성공! 총 {len(docs)}개의 관련 가이드 조각을 찾았습니다:\n")
    for idx, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 0) + 1
        print(f"[{idx}] 📄 페이지: 약 {page}쪽")
        print(f" 내용 요약: {doc.page_content[:150]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_kcg_rag_search()
