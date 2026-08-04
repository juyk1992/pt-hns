import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# GitHub Secrets에서 등록된 GEMINI_API_KEY 로드
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

# LangChain 전용 구글 API 키 환경변수 설정
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

def build_kcg_vector_db():
    pdf_path = "위험유해물질(HNS) 해양사고 대응 가이드.pdf"
    persist_directory = "./kcg_guide_chromadb"

    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF 파일('{pdf_path}')을 찾을 수 없습니다. 루트 경로에 업로드했는지 확인하세요.")
        sys.exit(1)

    print(f"📄 '{pdf_path}' PDF 로드 중...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️ 텍스트 Chunking (700자 단위 분할) 진행 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)

    print("🧠 Gemini Embedding (text-embedding-004) 변환 및 Chroma DB 생성 중...")
    # LangChain 공식 Google Embeddings 적용
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"🎉 성공! 총 {len(docs)}개 조각의 Vector DB가 '{persist_directory}'에 저장되었습니다.")

if __name__ == "__main__":
    build_kcg_vector_db()
