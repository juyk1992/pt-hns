import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai import types

# GitHub Secrets에서 등록된 GEMINI_API_KEY 로드
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

# Google 공식 최신 SDK 기반 Custom Embeddings 클래스
class GeminiEmbeddings:
    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        # 💡 [핵심 수정] 404 에러 방지를 위해 v1 정식 API 버전 명시
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version='v1')
        )
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 10  # 안정적인 API 처리를 위한 배치 분할
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            for emb in response.embeddings:
                embeddings.append(emb.values)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return response.embeddings[0].values

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

    print("🧠 Gemini Embedding (v1 API) 변환 및 Chroma DB 생성 중...")
    embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY, model_name="text-embedding-004")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"🎉 성공! 총 {len(docs)}개 조각의 Vector DB가 '{persist_directory}'에 저장되었습니다.")

if __name__ == "__main__":
    build_kcg_vector_db()
