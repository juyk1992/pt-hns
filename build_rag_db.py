import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_kcg_vector_db():
    pdf_path = "위험유해물질(HNS) 해양사고 대응 가이드.pdf"
    persist_directory = "./kcg_guide_chromadb"

    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF 파일('{pdf_path}')을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"📄 '{pdf_path}' PDF 로드 중...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️ 텍스트 Chunking 진행 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    print("🧠 로컬 한국어 임베딩 모델(ko-sroberta-multitask) 로드 및 Vector DB 생성 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"🎉 성공! 총 {len(docs)}개 조각의 Vector DB가 '{persist_directory}'에 저장되었습니다.")

if __name__ == "__main__":
    build_kcg_vector_db()
