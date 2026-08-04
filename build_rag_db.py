import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

def build_kcg_vector_db():
    pdf_path = "위험유해물질(HNS) 해양사고 대응 가이드.pdf"
    persist_directory = "./kcg_guide_chromadb"

    print(f"📄 '{pdf_path}' PDF 로드 중...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️ 텍스트 Chunking 진행 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    print("🧠 Gemini Embedding 변환 및 Chroma DB 생성 중...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"🎉 성공! '{persist_directory}'에 Vector DB 저장이 완료되었습니다.")

if __name__ == "__main__":
    build_kcg_vector_db()
