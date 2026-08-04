import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import json
import base64
import urllib3
from datetime import datetime, timedelta
from google import genai

# RAG Vector DB 연동 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# SSL 경고창 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. 로컬 이미지 경로 및 Base64 변환 함수
# ==========================================
KCG_LOGO_PATH = "kcg_logo.png"

@st.cache_data
def get_base64_logo(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

kcg_logo_b64 = get_base64_logo(KCG_LOGO_PATH)

# 페이지 설정
st.set_page_config(
    page_title="평택해양경찰서 HNS AI 대응 시스템",
    page_icon=KCG_LOGO_PATH if os.path.exists(KCG_LOGO_PATH) else "🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Reflex.dev 감성 Light UI + 모바일 완벽 가시성 보장 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-sub: #F1F5F9;
        --text-main: #0F172A;
        --text-sub: #64748B;
        --border-color: #E2E8F0;
        --accent-blue: #3B82F6;
        --accent-blue-hover: #2563EB;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-main: #0F172A;
            --bg-card: #1E293B;
            --bg-sub: #334155;
            --text-main: #F8FAFC;
            --text-sub: #94A3B8;
            --border-color: #334155;
            --accent-blue: #60A5FA;
            --accent-blue-hover: #3B82F6;
        }
    }

    [data-theme="dark"] {
        --bg-main: #0F172A;
        --bg-card: #1E293B;
        --bg-sub: #334155;
        --text-main: #F8FAFC;
        --text-sub: #94A3B8;
        --border-color: #334155;
        --accent-blue: #60A5FA;
        --accent-blue-hover: #3B82F6;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }

    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
    }

    .hero-container {
        padding: 2rem;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 1.5rem;
    }
    
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--text-main) !important;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    
    .sub-header {
        color: var(--text-sub) !important;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .badge-unno {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        padding: 3px 9px;
        border-radius: 8px;
        font-family: monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }

    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadio"] > div {
        display: grid !important;
        grid-template-rows: repeat(2, 38px) !important;
        grid-auto-flow: column !important;
        grid-auto-columns: max-content !important;
        gap: 8px 8px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        padding-bottom: 8px !important;
        -webkit-overflow-scrolling: touch !important;
    }

    div[data-testid="stRadio"] > div::-webkit-scrollbar {
        height: 5px !important;
    }
    div[data-testid="stRadio"] > div::-webkit-scrollbar-track {
        background: var(--bg-sub) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stRadio"] > div::-webkit-scrollbar-thumb {
        background: var(--border-color) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stRadio"] > div > label {
        height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 20px !important;
        padding: 0 16px !important;
        color: var(--text-main) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        cursor: pointer !important;
        white-space: nowrap !important;
    }

    div[data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background-color: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stRadio"] > div > label[data-checked="true"] *,
    div[data-testid="stRadio"] > div > label:has(input:checked) * {
        color: #FFFFFF !important;
    }

    .stTextInput input {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        background-color: var(--accent-blue) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover, .stButton > button:active {
        background-color: var(--accent-blue-hover) !important;
        color: #FFFFFF !important;
    }

    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        border-radius: 14px !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        font-weight: 700 !important;
        padding: 0.8rem 1rem !important;
    }

    .streamlit-expanderContent {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-bottom-left-radius: 14px !important;
        border-bottom-right-radius: 14px !important;
        padding: 1rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent !important;
        border-bottom: 2px solid var(--border-color) !important;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        background-color: var(--bg-sub) !important;
        border-radius: 12px !important;
        color: var(--text-sub) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0 18px !important;
        border: 1px solid var(--border-color) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card) !important;
        color: var(--accent-blue) !important;
        border: 1.5px solid var(--accent-blue) !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent-blue) !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

PUBLIC_API_KEY = st.secrets.get("PUBLIC_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# ==========================================
# 1. HNS 정보집 원본 텍스트 DB & RAG Vector DB 로드
# ==========================================
@st.cache_data
def load_full_hns_db():
    file_path = "hns_full_text_database.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

full_hns_db = load_full_hns_db()

def find_hns_raw_text(query):
    if not full_hns_db or not query:
        return None
        
    q = query.strip().upper()
    for item in full_hns_db:
        unno = str(item.get('unno', '')).strip()
        title = item.get('title_header', '').upper()
        synonyms = item.get('synonyms', '').upper()
        
        if (q and q == unno) or (q in title) or (q in synonyms):
            return item.get('raw_full_text', '')
    return None

# 🧠 RAG Vector DB 로드 모듈 (ChromaDB + HuggingFace)
@st.cache_resource
def load_kcg_vectorstore():
    persist_dir = "./kcg_guide_chromadb"
    if os.path.exists(persist_dir):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            print(f"RAG Vector DB 로드 실패: {e}")
    return None

kcg_vectorstore = load_kcg_vectorstore()

def fetch_rag_context(query, k=5):
    """해양사고 대응 가이드 PDF에서 연관 지침 RAG 검색"""
    if not kcg_vectorstore or not query:
        return "RAG 가이드 데이터베이스 미생성"
    
    try:
        docs = kcg_vectorstore.similarity_search(query, k=k)
        if not docs:
            return "관련 가이드 지침 검색 결과 없음"
        
        context_items = []
        for doc in docs:
            page = doc.metadata.get('page', 0) + 1
            context_items.append(f"[대응가이드 {page}쪽 지침]\n{doc.page_content}")
            
        return "\n\n".join(context_items)
    except Exception as e:
        return f"RAG 검색 오류: {e}"

# ==========================================
# 2. 공공 API 연동 모듈 (HTTPS 및 SSL 세션 적용)
# ==========================================
def fetch_dgst_info(unno):
    """[해양수산부 위험물정보 API]"""
    if not unno or unno in ["0000", "-", ""]:
        return {
            "imdgNm": "", "imdgEngNm": "", "kndNm": "-", "kndPrdlstNm": "-",
            "imdgGradCd": "-", "emergManagtCd": "-", "ldadngMth": "-", "catinMatter": "-"
        }

    clean_unno = str(unno).strip().zfill(4)
    # 💡 HTTPS 적용 및 인증키 URL 결합
    url = f"https://apis.data.go.kr/1192000/DgstInqire3/Info?serviceKey={PUBLIC_API_KEY}"
    params = {'unno': clean_unno, 'numOfRows': '1', 'pageNo': '1'}
    info = {
        "imdgNm": "", "imdgEngNm": "", "kndNm": "-", "kndPrdlstNm": "-",
        "imdgGradCd": "-", "emergManagtCd": "-", "ldadngMth": "-", "catinMatter": "-"
    }
    try:
        session = requests.Session()
        session.verify = False
        res = session.get(url, params=params, timeout=8)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            info['imdgNm'] = item.findtext('imdgNm') or ""
            info['imdgEngNm'] = item.findtext('imdgEngNm') or ""
            info['kndNm'] = item.findtext('kndNm') or "-"
            info['kndPrdlstNm'] = item.findtext('kndPrdlstNm') or "-"
            info['imdgGradCd'] = item.findtext('imdgGradCd') or "-"
            info['emergManagtCd'] = item.findtext('emergManagtCd') or "-"
            info['ldadngMth'] = item.findtext('ldadngMth') or "-"
            info['catinMatter'] = item.findtext('catinMatter') or "-"
    except Exception as e:
        print(f"위험물정보 API 에러: {e}")
    return info

def fetch_chem_safety_info(cas_no):
    """[화학물질안전원 화학물질 안전관리정보 API]"""
    if not cas_no or cas_no in ["-", "0000", "없음", ""]:
        return {
            "symptom": "자료 없음", "inhale": "자료 없음", "skin": "자료 없음",
            "eyeball": "자료 없음", "oral": "자료 없음", "etc": "자료 없음"
        }

    clean_cas = str(cas_no).strip()
    # 💡 HTTPS 적용 및 인증키 URL 결합
    url = f"https://apis.data.go.kr/1480802/iciskischem/kischemlist?serviceKey={PUBLIC_API_KEY}"
    params = {'numOfRows': '3', 'pageNo': '1', 'casNo': clean_cas}
    safety_data = {
        "symptom": "자료 없음", "inhale": "자료 없음", "skin": "자료 없음", 
        "eyeball": "자료 없음", "oral": "자료 없음", "etc": "자료 없음"
    }
    try:
        session = requests.Session()
        session.verify = False
        res = session.get(url, params=params, timeout=8)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            safety_data['symptom'] = item.findtext('symptom') or "자료 없음"
            safety_data['inhale'] = item.findtext('inhale') or "자료 없음"
            safety_data['skin'] = item.findtext('skin') or "자료 없음"
            safety_data['eyeball'] = item.findtext('eyeball') or "자료 없음"
            safety_data['oral'] = item.findtext('oral') or "자료 없음"
            safety_data['etc'] = item.findtext('etc') or "자료 없음"
    except Exception as e:
        print(f"화학물질 안전관리정보 API 에러: {e}")
    return safety_data

def fetch_kosha_msds_info(chem_name, cas_no, unno):
    """[안전보건공단 MSDS OPEN API 연동]"""
    base_url = "https://msds.kosha.or.kr/openapi/service/msdschem"
    chem_id = None

    search_trials = [
        (chem_name, "0"),
        (cas_no, "1"),
        (unno, "2")
    ]

    for search_wrd, search_cnd in search_trials:
        if not search_wrd or search_wrd in ["-", "0000", "없음"]:
            continue
        
        list_url = f"{base_url}/getChemList"
        params = {
            'serviceKey': PUBLIC_API_KEY,
            'searchWrd': search_wrd.strip(),
            'searchCnd': search_cnd,
            'numOfRows': '1',
            'pageNo': '1'
        }
        try:
            res = requests.get(list_url, params=params, timeout=5)
            root = ET.fromstring(res.content)
            item = root.find('.//item')
            if item is not None:
                found_id = item.findtext('chemId') or item.findtext('chemId'.lower())
                if found_id:
                    chem_id = found_id.strip()
                    break
        except Exception as e:
            print(f"KOSHA getChemList (cnd={search_cnd}) 에러: {e}")

    if not chem_id:
        return "안전보건공단 MSDS 연동 데이터 없음 (chemId 미발급)"

    msds_details = []
    for i in range(1, 17):
        op_name = f"getChemDetail{i:02d}"
        detail_url = f"{base_url}/{op_name}"
        params = {'serviceKey': PUBLIC_API_KEY, 'chemId': chem_id}
        try:
            res = requests.get(detail_url, params=params, timeout=4)
            root = ET.fromstring(res.content)
            items = root.findall('.//item')
            for item in items:
                name_kor = (item.findtext('msdsItemNameKor') or '').strip()
                detail_val = (item.findtext('itemDetail') or '').strip()
                if detail_val and detail_val != "자료없음":
                    msds_details.append(f"[{name_kor}] {detail_val}")
        except Exception:
            continue

    if not msds_details:
        return f"안전보건공단 MSDS 기본 정보 등록 (chemId: {chem_id})"

    return f"[KOSHA MSDS chemId: {chem_id}]\n" + "\n".join(msds_details[:30])

# ==========================================
# 3. Gemini 자연어 매핑 및 AI 요약
# ==========================================
@st.cache_data(ttl=3600)
def map_search_query_with_gemini(query_text):
    default_res = {
        "chem_ko": query_text, 
        "chem_eng": query_text, 
        "unno": "0000", 
        "cas_no": "-", 
        "accident_context": ""
    }
    if not GEMINI_API_KEY or not query_text:
        return default_res

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        사용자 입력 텍스트: "{query_text}"
        
        이 문장 또는 검색어에서:
        1. 포함되어 있거나 가장 대표적인 위험물/HNS 화학물질의 표준 정보(한글명, 영문명, UN번호 4자리, CAS번호)를 분석하세요.
        2. 만약 특정 사고 상황(예: 좌초, 충돌, 유출, 화재, 침수, 특정 장소 등)이 언급되어 있다면 해당 핵심 사고 상황 요약 문장을 "accident_context"에 작성하세요. (단순 물질명 검색일 경우 빈 문자열 "")

        아래 JSON 포맷으로만 답변하세요. 다른 설명은 금지합니다:
        {{
            "chem_ko": "공식 한글명", 
            "chem_eng": "공식 영문명", 
            "unno": "4자리 UN번호", 
            "cas_no": "CAS번호(예: 7664-93-9)",
            "accident_context": "사고 상황 요약 문장 또는 빈값"
        }}
        """
        
        candidate_models = ['gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
        for model_id in candidate_models:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                text = response.text.replace('```json', '').replace('```', '').strip()
                res_json = json.loads(text)
                if "accident_context" not in res_json:
                    res_json["accident_context"] = ""
                return res_json
            except Exception:
                continue

        return default_res
    except Exception:
        return default_res

def generate_gemini_summary(chem_name, unno, cas_no, dgst_info, safety_info, kosha_msds_text, rag_text, accident_context=""):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API 키가 설정되지 않았습니다."
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        hns_raw_text = find_hns_raw_text(unno) or find_hns_raw_text(chem_name)
        hns_context = f"[해양경찰청 HNS 정보집 원본 데이터]\n{hns_raw_text}\n" if hns_raw_text else ""

        rag_context = f"[위험유해물질(HNS) 해양사고 대응 가이드 PDF RAG 검색 결과]\n{rag_text}\n"
        
        accident_info = f"\n🚨 [현장 사고 상황 조건]: {accident_context}\n" if accident_context else ""

        # 💡 [줄바꿈 및 개별 불릿 적용 고도화 프롬프트 + 문구 교체]
        prompt = f"""
        당신은 해양경찰청 및 항만 HNS 비상대응 상황실 관제관입니다.
        수집된 다중 데이터(공공 API, HNS DB, 해경 대응가이드 RAG) 및 [해상화학사고 상황실 대응절차 가이드]를 종합 분석하여, 관제관이 현장 세력(OSC, 함정, 구조대 등)에 바로 지시/전파할 수 있는 비상대응 가이드를 작성하세요.

        {accident_info}
        {hns_context}
        {rag_context}

        [해양수산부 위험물정보 API 수집 데이터]
        - 물질명: {chem_name} (UN NO: {unno})
        - IMDG 명칭: {dgst_info.get('imdgNm')} ({dgst_info.get('imdgEngNm')})
        - IMDG 등급코드 / 종류명: {dgst_info.get('imdgGradCd', '-')} / {dgst_info.get('kndNm', '-')}
        - 종류품목명: {dgst_info.get('kndPrdlstNm', '-')}
        - 비상조치코드(EmS): {dgst_info.get('emergManagtCd', '-')}
        - 선박 적재방법: {dgst_info.get('ldadngMth', '-')}
        - 주의사항: {dgst_info.get('catinMatter', '-')}

        [화학물질안전원 안전관리정보 API 수집 데이터 (CAS NO: {cas_no})]
        - 일반 증상 및 표적장기: {safety_info.get('symptom', '-')}
        - 흡입/피부/안구/경구 영향: {safety_info.get('inhale', '-')}, {safety_info.get('skin', '-')}, {safety_info.get('eyeball', '-')}, {safety_info.get('oral', '-')}
        - 기타 유의사항: {safety_info.get('etc', '-')}

        [안전보건공단 MSDS 1~16번 종합 수집 데이터]
        {kosha_msds_text}

        [상황실 지침 반영 엄격 작성 규칙]
        1. [초동대응 핵심요약]: 각 항목의 시작은 `* **항목명**:` 포맷을 사용하고, 현장 실행 위주의 명확한 개조식 문장으로 작성하세요.
        2. [수치 및 안전 기준]: 
           - 이격거리 및 보호구 등 핵심 수치는 MSDS/물질정보집 참조 및 해경 HNS 대응가이드(RAG) 수치를 최우선 반영하세요.
           - 물질명 미확인 시 기본 유출 100m / 화재 800m 이격 조치를 지정하세요.
           - 물 반응성 물질 확인 시 직사주수 절대 금지를 명시하세요.
        3. [사고 상황 맞춤 지침]: [현장 사고 상황 조건]이 존재할 경우(화재·폭발, 유출, 충돌·침수, 좌초 등), 해당 사고 유형별 비상조치 지원 지침을 최우선 지시사항으로 포함하세요.
        4. [세부 지침 1~4번]: 모든 항목을 현장 실행 위주의 명확한 개조식 문장으로 작성하세요.

        --- 출력 형식을 엄격히 준수하세요 (항목 간 반드시 엔터로 줄바꿈) ---

        ### 🚨 [초동대응 핵심요약]

        * **사고물질 및 위험성 판단**: [IMDG 등급 및 핵심위험성(인화성/독성/수반응성 등) 전파 및 위험성 평가 지시]
        * **통제 및 이격거리 지시**: [초기이격, 화재대피, 유출방호 M단위 수치 명시 및 해역/현장 통제 조치 지시]
        * **출동세력 보호구 지정**: [필수 레벨(Level A/B/C/D) 및 필수 장비(공기호흡기, 내화학복, 가스탐지기 등) 착용 지시]
        * **현장 초동 행동 수칙**: [풍상위치 확보, 사고유형 맞춤 행동, 직수금지/소화약제 및 사고선 비상조치 확인 지시]

        ---
        ### 1. ⚠️ 물리·화학적 성상 및 주요 위험성
        ### 2. 🛡️ 현장 개인 보호구 및 초동 방제/소화 요령
        ### 3. ⛔ 절대 금지 행동 (금기 사항)
        ### 4. 🏥 인체 노출 시 신체 영향 및 긴급 응급조치
        """

        candidate_models = ['gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
        for model_id in candidate_models:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                return response.text
            except Exception:
                continue
                
        return "⚠️ Gemini API 호출에 실패했습니다."

    except Exception as e:
        return f"Gemini API 클라이언트 생성 오류: {e}"

# ==========================================
# 4. 항구별 RPA 데이터 로드 (30초 캐시)
# ==========================================
@st.cache_data(ttl=30)
def load_integrated_hns_data(port_code):
    filename_map = {
        "031": "hns_pyeongtaek_report.csv",
        "300": "hns_daesan_report.csv"
    }
    
    file_path = filename_map.get(port_code, "hns_pyeongtaek_report.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['호출부호'] = df['호출부호'].astype(str).str.strip()
        df['선박명(선택)'] = df['선박명(선택)'].astype(str).str.strip()
        return df
    return None

# ==========================================
# 5. 항구별 데이터 렌더링 헬퍼 함수
# ==========================================
def render_port_dashboard(port_name, port_code):
    kst_now = datetime.utcnow() + timedelta(hours=9)
    today_str = kst_now.strftime("%Y-%m-%d")
    from_str = (kst_now - timedelta(days=3)).strftime("%Y-%m-%d")
    
    df = load_integrated_hns_data(port_code)

    col_title, col_metric = st.columns([3, 1])
    with col_title:
        st.markdown(f"#### 📊 {port_name} 위험물 반입 현황")
        st.caption(f"조회기간: {from_str} ~ {today_str}")

    if df is None or df.empty:
        st.warning(f"⚠️ {port_name}의 {from_str} ~ {today_str} 기준 수집 데이터가 없습니다. RPA 봇을 가동해 주세요.")
    else:
        ship_column = "선박명(선택)" if "선박명(선택)" in df.columns else df.columns[0]
        unique_ships = sorted([str(s) for s in df[ship_column].dropna().unique()])
        ship_options = ["전체 보기"] + unique_ships
        
        st.markdown(f"**🚢 [{port_name}] 조회할 선박 선택**")
        selected_ship = st.radio(
            label=f"ship_radio_{port_code}",
            options=ship_options,
            index=0,
            horizontal=True,
            key=f"select_ship_radio_{port_code}"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        if selected_ship != "전체 보기":
            filtered_df = df[df[ship_column] == selected_ship]
        else:
            filtered_df = df

        with col_metric:
            st.metric(label="반입 신고 건수", value=f"{len(filtered_df)} 건")

        display_cols = [
            '선박명(선택)', '호출부호', '사용목적', '운송형태', '화물명', 
            '하역업체', '하역기간', '사용장소', '전출항지', 'UNNO', 'IMDG', '품명', '중량', '단위'
        ]
        
        with st.expander(f"📋 {port_name} 위험물 반입 신고 목록 데이터 보기", expanded=False):
            st.dataframe(filtered_df[[c for c in display_cols if c in filtered_df.columns]], use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"##### 🚢 {port_name} 선박별 상세 운송 정보 및 AI 비상 대응")

        for ship in filtered_df['선박명(선택)'].unique():
            ship_data = filtered_df[filtered_df['선박명(선택)'] == ship]
            call_sign = ship_data['호출부호'].iloc[0]
            location = ship_data['사용장소'].iloc[0] if pd.notna(ship_data['사용장소'].iloc[0]) else "장소 미상"
            work_period = ship_data['하역기간'].iloc[0]
            use_purpose = ship_data['사용목적'].iloc[0] if '사용목적' in ship_data.columns else "-"
            transport_type = ship_data['운송형태'].iloc[0] if '운송형태' in ship_data.columns else "-"
            prev_port = ship_data['전출항지'].iloc[0] if '전출항지' in ship_data.columns else "-"
            
            with st.expander(f"⚓ [{ship}] (호출부호: {call_sign}) ｜ 하역장소: {location} ｜ 기간: {work_period}"):
                st.markdown(f"**🏢 하역업체:** {ship_data['하역업체'].iloc[0]} &nbsp;\|&nbsp; **사용목적:** {use_purpose} &nbsp;\|&nbsp; **운송형태:** {transport_type} &nbsp;\|&nbsp; **전출항지:** {prev_port}")
                st.markdown("---")
                st.markdown("###### 📦 적재 위험물 목록")
                
                for idx, row in ship_data.iterrows():
                    unno = str(row['UNNO']).zfill(4) if pd.notna(row['UNNO']) else "0000"
                    chem_name = str(row['품명']) if pd.notna(row['품명']) else "정보 없음"
                    weight = str(row['중량']) if pd.notna(row['중량']) else "-"
                    unit = str(row['단위']) if pd.notna(row['단위']) else ""
                    imdg = str(row['IMDG']) if 'IMDG' in row and pd.notna(row['IMDG']) else "-"
                    
                    if unno == '0000' or not unno.strip():
                        continue
                    
                    c_info, c_btn = st.columns([4, 1])
                    with c_info:
                        st.markdown(f"• <span class='badge-unno'>UN {unno}</span> &nbsp; **{chem_name}** &nbsp; <span style='color:#64748B;'>(IMDG: {imdg} / 수량: {weight} {unit})</span>", unsafe_allow_html=True)
                    with c_btn:
                        button_key = f"btn_{port_code}_{ship}_{idx}_{unno}"
                        if st.button("🤖 AI 가이드 생성", key=button_key, use_container_width=True):
                            mapped_info = map_search_query_with_gemini(chem_name)
                            st.session_state['active_chem'] = mapped_info.get("chem_ko", chem_name)
                            st.session_state['active_unno'] = unno
                            st.session_state['active_cas'] = mapped_info.get("cas_no", "-")
                            st.session_state['active_ship'] = f"[{port_name}] {ship}"
                            st.session_state['active_accident_context'] = mapped_info.get("accident_context", "")
                            st.session_state['active_summary'] = ""
                            st.session_state['active_key_changed'] = True
                            st.rerun()

# ==========================================
# 6. 메인 화면 구성 (Hero Section & 로고 정렬)
# ==========================================
if kcg_logo_b64:
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 6px;">
            <img src="data:image/png;base64,{kcg_logo_b64}" style="width: 58px; height: auto; object-fit: contain;" alt="해양경찰 로고" />
            <div class="main-header" style="margin: 0;">평택해양경찰서 HNS AI 대응 시스템</div>
        </div>
        <div class="sub-header">포트미스(PORT-MIS) + 공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 정보집, HNS 대응가이드) + Gemini AI</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-container">
        <div class="main-header">🚢 평택해양경찰서 HNS AI 대응 솔루션</div>
        <div class="sub-header">포트미스(PORT-MIS) + 공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 정보집, HNS 대응가이드) + Gemini AI</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# 🔥 [고도화] HNS AI 통합 검색창 (물질명 및 사고 상황 자유 입력)
# ------------------------------------------
st.markdown("### 🔎 AI 통합검색 (화학물질 또는 사고 상황 입력)")
search_input = st.text_input(
    "화학물질명, 화학식, 관용명 또는 사고 상황을 자유롭게 입력하세요 (예: 황산, H2SO4, LNG / 평택호 좌초로 질산 유출 중)", 
    key="global_search_box"
)

if search_input:
    with st.spinner("Gemini AI가 입력 내용을 지능형 분석 중..."):
        mapped_result = map_search_query_with_gemini(search_input)
        mapped_ko = mapped_result.get("chem_ko", search_input)
        mapped_eng = mapped_result.get("chem_eng", search_input)
        mapped_unno = str(mapped_result.get("unno", "0000")).zfill(4)
        mapped_cas = str(mapped_result.get("cas_no", "-"))
        accident_ctx = mapped_result.get("accident_context", "")
        
        c1, c2 = st.columns([4, 1])
        with c1:
            info_msg = f"💡 **AI 매핑 결과:** 물질명: **{mapped_ko}** ({mapped_eng}) ｜ UN NO: `{mapped_unno}` ｜ CAS NO: `{mapped_cas}`"
            if accident_ctx:
                info_msg += f"\n ｜ 🚨 **사고 상황 식별:** `{accident_ctx}`"
            st.info(info_msg)
        with c2:
            if st.button("🤖 AI 가이드 생성", key="btn_global_search", use_container_width=True):
                st.session_state['active_chem'] = mapped_ko
                st.session_state['active_unno'] = mapped_unno
                st.session_state['active_cas'] = mapped_cas
                st.session_state['active_ship'] = f"자유 통합 검색 ('{search_input}')"
                st.session_state['active_accident_context'] = accident_ctx
                st.session_state['active_summary'] = ""
                st.session_state['active_key_changed'] = True
                st.rerun()

# ------------------------------------------
# ⚡ AI 대응 가이드 출력 모달/컨테이너
# ------------------------------------------
if 'active_chem' in st.session_state:
    st.divider()
    chem = st.session_state['active_chem']
    unno = st.session_state['active_unno']
    cas = st.session_state.get('active_cas', '-')
    ship_info = st.session_state['active_ship']
    accident_ctx = st.session_state.get('active_accident_context', '')
    
    status_header = f"⚡ [지능형 비상대응 가이드] 대상: {ship_info} ｜ 물질: {chem} (UN NO: {unno} / CAS NO: {cas})"
    if accident_ctx:
        status_header += f" ｜ 상황: {accident_ctx}"
    st.error(status_header)
    
    st.caption("⚠️ **[할루시네이션 주의]** 본 대응 가이드는 공공 API 3종 및 해경 HNS 정보집 DB, HNS 대응가이드를 통합한 RAG(검색증강생성) 모델로 AI 환각(Hallucination) 현상을 최소화했습니다. 단, 현장 상황에 따라 다를 수 있으므로 재확인을 권장합니다.")
    
    if 'active_summary' not in st.session_state or st.session_state.get('active_key_changed', False) or not st.session_state['active_summary']:
        with st.spinner('공공 API + 해경 HNS DB + Gemini AI 종합 분석 중...'):
            dgst_info = fetch_dgst_info(unno)
            safety_info = fetch_chem_safety_info(cas)
            kosha_msds_text = fetch_kosha_msds_info(chem, cas, unno)
            
            # RAG Vector DB 검색 (검색어 + 사고 상황 포함 조합)
            rag_search_query = f"{chem} {unno} {accident_ctx} 사고 대응 방제 조치"
            rag_text = fetch_rag_context(rag_search_query)
            
            # 활용 원본 데이터 세션 저장
            st.session_state['active_source_data'] = {
                "dgst": dgst_info,
                "safety": safety_info,
                "hns_raw": find_hns_raw_text(unno) or find_hns_raw_text(chem),
                "rag_text": rag_text,
                "kosha": kosha_msds_text
            }
            
            st.session_state['active_summary'] = generate_gemini_summary(
                chem, unno, cas, dgst_info, safety_info, kosha_msds_text, rag_text, accident_context=accident_ctx
            )
            st.session_state['active_key_changed'] = False
        
    st.markdown(st.session_state['active_summary'])
    
    # ------------------------------------------
    # 📚 활용 원본 자료 확인 탭 (접이식 Expander)
    # ------------------------------------------
    if 'active_source_data' in st.session_state:
        src = st.session_state['active_source_data']
        with st.expander("📚 생성 정보 출처 및 활용 원본 데이터 검증/보기", expanded=False):
            t1, t2, t3, t4, t5 = st.tabs([
                "🚢 해수부 위험물정보", 
                "🛡️ 화학물질안전원", 
                "📄 해경 HNS 정보집", 
                "🧠 해경 HNS 대응가이드",
                "🏥 안전보건공단 MSDS"
            ])
            
            with t1:
                st.markdown("**[해양수산부 위험물정보 API 수집 데이터]**")
                d = src.get("dgst", {})
                st.write(f"- **IMDG 한글/영문명:** {d.get('imdgNm', '-')} ({d.get('imdgEngNm', '-')})")
                st.write(f"- **IMDG 등급 / 종류:** {d.get('imdgGradCd', '-')} / {d.get('kndNm', '-')}")
                st.write(f"- **비상조치코드(EmS):** {d.get('emergManagtCd', '-')}")
                st.write(f"- **선박 적재방법:** {d.get('ldadngMth', '-')}")
                st.write(f"- **주의사항:** {d.get('catinMatter', '-')}")

            with t2:
                st.markdown("**[화학물질안전원 화학물질 안전관리정보 API 수집 데이터]**")
                s = src.get("safety", {})
                st.write(f"- **표적장기 및 주요증상:** {s.get('symptom', '-')}")
                st.write(f"- **흡입 영향:** {s.get('inhale', '-')}")
                st.write(f"- **피부 노출:** {s.get('skin', '-')}")
                st.write(f"- **안구 노출:** {s.get('eyeball', '-')}")
                st.write(f"- **기타 유의사항:** {s.get('etc', '-')}")

            with t3:
                st.markdown("**[해양경찰청 HNS 정보집]**")
                hns_t = src.get("hns_raw")
                if hns_t:
                    st.text_area("HNS 정보집 원본 텍스트", value=hns_t[:1500] + ("..." if len(hns_t) > 1500 else ""), height=200, disabled=True)
                else:
                    st.info("해당 물질의 HNS 정보집 단일 텍스트 DB 매칭 내역 없음 (API 및 RAG 대체)")

            with t4:
                st.markdown("**[위험유해물질(HNS) 해양사고 대응 가이드]**")
                rag_t = src.get("rag_text", "")
                st.text_area("Vector DB 추출 지침 (k=5)", value=rag_t, height=200, disabled=True)

            with t5:
                st.markdown("**[안전보건공단 MSDS API 수집 데이터]**")
                k_t = src.get("kosha", "")
                st.text_area("MSDS 세부 수집 정보", value=k_t, height=200, disabled=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("❌ 가이드 창 닫기", key="close_global_guide", use_container_width=True):
        for key in ['active_chem', 'active_unno', 'active_cas', 'active_ship', 'active_accident_context', 'active_summary', 'active_source_data', 'active_key_changed']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.divider()

# ------------------------------------------
# ⚓ 항구별 위험물 반입 현황 탭 (평택항 / 대산항)
# ------------------------------------------
tab_pyeongtaek, tab_daesan = st.tabs(["⚓ 평택항 현황 및 대응 (청코드: 031)", "⚓ 대산항 현황 및 대응 (청코드: 300)"])

with tab_pyeongtaek:
    render_port_dashboard("평택항", "031")

with tab_daesan:
    render_port_dashboard("대산항", "300")
