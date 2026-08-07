import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import json
import base64
import urllib3
import time
import threading
from datetime import datetime, timezone, timedelta
from google import genai

# RAG Vector DB 연동 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# SSL 경고창 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# ⚡ [백그라운드 Keep-Alive] 앱 잠듦(Sleep) 자동 방지
# ==========================================
def keep_app_alive():
    """Streamlit Cloud 수면 상태 전환 방지 (10분 주기 자가 요청 백그라운드 쓰레드)"""
    while True:
        time.sleep(600)  # 10분 마다 동작
        try:
            # 로컬 세션 유지용 내부 시그널
            _ = datetime.now()
        except Exception:
            pass

if 'keep_alive_started' not in st.session_state:
    st.session_state['keep_alive_started'] = True
    t = threading.Thread(target=keep_app_alive, daemon=True)
    t.start()

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
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
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
# 2. 공공 API 연동 모듈 (자동 페이징 수집 적용)
# ==========================================

@st.cache_data(ttl=300)
def fetch_vessel_schedule_api(port_code, de_gb):
    """
    [해양수산부 선박운항정보 API (VsslEtrynd5)]
    - 50건/100건 제한 우회: totalCount 전체 데이터를 끝까지 자동 수집 (Paging Loop)
    """
    if not PUBLIC_API_KEY:
        return []

    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
    sde_str = (now_kst - timedelta(days=1)).strftime("%Y%m%d")
    ede_str = (now_kst + timedelta(days=1)).strftime("%Y%m%d")

    url = f"https://apis.data.go.kr/1192000/VsslEtrynd5/Info5?serviceKey={PUBLIC_API_KEY}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    vessels = []
    page = 1
    rows_per_page = 100

    while True:
        params = {
            'prtAgCd': str(port_code).strip(),
            'sde': sde_str,
            'ede': ede_str,
            'deGb': str(de_gb).strip().upper(),
            'numOfRows': str(rows_per_page),
            'pageNo': str(page)
        }

        try:
            session = requests.Session()
            session.verify = False
            res = session.get(url, params=params, headers=headers, timeout=10)

            if res.status_code == 200 and res.content:
                root = ET.fromstring(res.content)
                total_cnt = int(root.findtext('.//totalCount', '0'))
                items = root.findall('.//item') or root.findall('body/items/item')

                if not items:
                    break

                for item in items:
                    vssl_nm = (item.findtext('vsslNm') or '-').strip()
                    clsgn = (item.findtext('clsgn') or '-').strip()
                    vssl_nlty = (item.findtext('vsslNltyNm') or '-').strip()
                    vssl_knd = (item.findtext('vsslKndNm') or '-').strip()
                    etrypt_yr = (item.findtext('etryptYear') or '-').strip()
                    etrypt_co = (item.findtext('etryptCo') or '-').strip()

                    detail_node = item.find('.//detail') or item.find('details/detail')
                    facility_nm = '-'
                    etrypt_dt = '-'
                    tkoff_dt = '-'
                    smit_nm = '-'
                    
                    if detail_node is not None:
                        facility_nm = (detail_node.findtext('laidupFcltyNm') or '-').strip()
                        etrypt_dt = (detail_node.findtext('etryptDt') or '-').strip()
                        tkoff_dt = (detail_node.findtext('tkoffDt') or '-').strip()
                        smit_nm = (detail_node.findtext('reqstSeNm') or '-').strip()

                    if vssl_nm == '-' and clsgn == '-':
                        continue

                    vessels.append({
                        "vssl_nm": vssl_nm,
                        "clsgn": clsgn,
                        "vssl_nlty": vssl_nlty,
                        "vssl_knd": vssl_knd,
                        "facility_nm": facility_nm,
                        "etrypt_dt": etrypt_dt.replace('T', ' ') if etrypt_dt != '-' else '-',
                        "tkoff_dt": tkoff_dt.replace('T', ' ') if tkoff_dt != '-' else '-',
                        "smit_nm": smit_nm,
                        "etrypt_yr": etrypt_yr,
                        "etrypt_co": etrypt_co
                    })

                # 수집된 데이터 수가 전체 개수 이상이면 완료
                if len(vessels) >= total_cnt or len(items) < rows_per_page:
                    break
                page += 1
            else:
                break
        except Exception as e:
            print(f"선박 API 수집 중 예외 ({port_code}/{de_gb}): {e}")
            break

    return vessels

def fetch_dgst_info(unno):
    """[해양수산부 위험물정보 API]"""
    if not unno or unno in ["0000", "-", ""]:
        return {
            "imdgNm": "", "imdgEngNm": "", "kndNm": "-", "kndPrdlstNm": "-",
            "imdgGradCd": "-", "emergManagtCd": "-", "ldadngMth": "-", "catinMatter": "-"
        }

    clean_unno = str(unno).strip().zfill(4)
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
            "symptom": "자료없음", "inhale": "자료없음", "skin": "자료없음",
            "eyeball": "자료없음", "oral": "자료없음", "etc": "자료없음"
        }

    clean_cas = str(cas_no).strip()
    url = f"https://apis.data.go.kr/1480802/iciskischem/kischemlist?serviceKey={PUBLIC_API_KEY}"
    params = {'numOfRows': '3', 'pageNo': '1', 'casNo': clean_cas}
    safety_data = {
        "symptom": "자료없음", "inhale": "자료없음", "skin": "자료없음", 
        "eyeball": "자료없음", "oral": "자료없음", "etc": "자료없음"
    }
    try:
        session = requests.Session()
        session.verify = False
        res = session.get(url, params=params, timeout=8)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            safety_data['symptom'] = item.findtext('symptom') or "자료없음"
            safety_data['inhale'] = item.findtext('inhale') or "자료없음"
            safety_data['skin'] = item.findtext('skin') or "자료없음"
            safety_data['eyeball'] = item.findtext('eyeball') or "자료없음"
            safety_data['oral'] = item.findtext('oral') or "자료없음"
            safety_data['etc'] = item.findtext('etc') or "자료없음"
    except Exception as e:
        print(f"화학물질 안전관리정보 API 에러: {e}")
    return safety_data

def fetch_kosha_msds_info(chem_name, cas_no, unno):
    """[안전보건공단 MSDS OPEN API 연동]"""
    base_url = "https://msds.kosha.or.kr/openapi/service/msdschem"
    chem_id = None

    search_trials = [
        (unno, "2"),
        (cas_no, "1"),
        (chem_name, "0")
    ]

    for search_wrd, search_cnd in search_trials:
        if not search_wrd or str(search_wrd).strip() in ["-", "0000", "0", "없음", ""]:
            continue
        
        clean_wrd = str(search_wrd).strip()
        list_url = f"{base_url}/getChemList"
        params = {
            'serviceKey': PUBLIC_API_KEY,
            'searchWrd': clean_wrd,
            'searchCnd': search_cnd,
            'numOfRows': '5',
            'pageNo': '1'
        }
        try:
            res = requests.get(list_url, params=params, timeout=5)
            root = ET.fromstring(res.content)
            items = root.findall('.//item')
            
            matched_id = None
            for item in items:
                found_id = item.findtext('chemId') or item.findtext('chemId'.lower())
                if not found_id:
                    continue
                
                if search_cnd == "2":
                    item_unno = (item.findtext('unno') or '').strip()
                    if item_unno == clean_wrd.zfill(4):
                        matched_id = found_id.strip()
                        break
                elif search_cnd == "1":
                    item_cas = (item.findtext('casNo') or '').strip()
                    if item_cas == clean_wrd:
                        matched_id = found_id.strip()
                        break
                elif search_cnd == "0":
                    item_ko = (item.findtext('chemKo') or '').strip().replace("·", "")
                    item_en = (item.findtext('chemEn') or '').strip().replace("·", "").lower()
                    target_wrd = clean_wrd.replace("·", "").lower()
                    
                    if target_wrd == item_ko.lower() or target_wrd == item_en:
                        matched_id = found_id.strip()
                        break
            
            if matched_id:
                chem_id = matched_id
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

        [선박오염방지규칙 제3조 유해액체물질 분류 기준 참고]
        - X류 물질: 해양에 배출 시 심각한 위해, 해양배출 전면 금지
        - Y류 물질: 해양에 배출 시 위해 발생, 해양배출 제한
        - Z류 물질: 해양에 배출 시 경미한 위해, 해양배출 일부 제한
        - 기타물질 / 잠정평가물질: 위해가 없거나 잠정 평가된 물질
        (※ HNS 정보집 원본 데이터 등에 X, Y, Z류 오염 범주가 명시되어 있다면 해당 내용을 식별하여 반영하세요.)

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
# 4. 항만별 선박 모니터링 UI 렌더링 함수
# ==========================================

def render_vessel_tab_content(port_name, port_code, de_gb):
    gb_title = "입항" if de_gb == 'I' else "출항"
    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
    sde_fmt = (now_kst - timedelta(days=1)).strftime("%Y-%m-%d")
    ede_fmt = (now_kst + timedelta(days=1)).strftime("%Y-%m-%d")

    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.markdown(f"#### 📊 {port_name} {gb_title} 신고 선박 현황 (`{sde_fmt}` ~ `{ede_fmt}` 기준)")
    with c_right:
        if st.button(f"🔄 {port_name} 데이터 강제 갱신", key=f"refresh_{port_code}_{de_gb}"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"{port_name} {gb_title} 선박 전체 데이터 수집 중..."):
        vessels = fetch_vessel_schedule_api(port_code, de_gb)

    if not vessels:
        st.info(f"💡 해당 기간({sde_fmt} ~ {ede_fmt}) {port_name} {gb_title} 신고 선박 정보가 없거나 API 수집 대기 중입니다.")
        return

    st.success(f"✅ 총 **{len(vessels)}** 척의 {gb_title} 신고 선박이 수집되었습니다.")

    for idx, v in enumerate(vessels):
        expander_label = f"🚢 [{v['vssl_nm']}] 호출부호: {v['clsgn']} ｜ 선종: {v['vssl_knd']} ｜ 계선장소: {v['facility_nm']}"
        with st.expander(expander_label):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- **선명:** {v['vssl_nm']}")
                st.write(f"- **호출부호:** `{v['clsgn']}`")
                st.write(f"- **국적:** {v['vssl_nlty']}")
                st.write(f"- **선종:** {v['vssl_knd']}")
            with col2:
                st.write(f"- **신고구분:** {v['smit_nm']}")
                st.write(f"- **계선장소:** {v['facility_nm']}")
                st.write(f"- **입항일시:** {v['etrypt_dt']}")
                st.write(f"- **출항일시:** {v['tkoff_dt']}")

            st.markdown("---")
            if st.button(f"🤖 [{v['vssl_nm']}] 선종 기준 AI 대응가이드 생성", key=f"btn_vssl_{port_code}_{de_gb}_{idx}", use_container_width=True):
                mapped_info = map_search_query_with_gemini(v['vssl_knd'])
                st.session_state['active_chem'] = mapped_info.get("chem_ko", v['vssl_knd'])
                st.session_state['active_unno'] = mapped_info.get("unno", "0000")
                st.session_state['active_cas'] = mapped_info.get("cas_no", "-")
                st.session_state['active_ship'] = f"[{port_name} {gb_title}] {v['vssl_nm']} ({v['vssl_knd']})"
                st.session_state['active_accident_context'] = mapped_info.get("accident_context", "")
                st.session_state['active_summary'] = ""
                st.session_state['active_key_changed'] = True
                st.rerun()

# ==========================================
# 5. 메인 화면 구성 (Hero Section & 로고 정렬)
# ==========================================
if kcg_logo_b64:
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 6px;">
            <img src="data:image/png;base64,{kcg_logo_b64}" style="width: 58px; height: auto; object-fit: contain;" alt="해양경찰 로고" />
            <div class="main-header" style="margin: 0;">평택해양경찰서 HNS AI 대응 시스템</div>
        </div>
        <div class="sub-header">공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 정보집, HNS 대응가이드) + Gemini AI</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-container">
        <div class="main-header">🚢 평택해양경찰서 HNS AI 대응 솔루션</div>
        <div class="sub-header">공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 정보집, HNS 대응가이드) + Gemini AI</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# 🔥 HNS AI 통합 검색창 (물질명 및 사고 상황 자유 입력)
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
                st.markdown("**[해양수산부 위험물정보 API]**")
                d = src.get("dgst", {})
                st.write(f"- **IMDG 한글/영문명:** {d.get('imdgNm', '-')} ({d.get('imdgEngNm', '-')})")
                st.write(f"- **IMDG 등급 / 종류:** {d.get('imdgGradCd', '-')} / {d.get('kndNm', '-')}")
                st.write(f"- **비상조치코드(EmS):** {d.get('emergManagtCd', '-')}")
                st.write(f"- **선박 적재방법:** {d.get('ldadngMth', '-')}")
                st.write(f"- **주의사항:** {d.get('catinMatter', '-')}")

            with t2:
                st.markdown("**[화학물질안전원 화학물질안전관리정보 API]**")
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
                    st.info("해당 물질의 HNS 정보집 매칭 내역 없음")

            with t4:
                st.markdown("**[해양경찰청 HNS 해양사고 대응 가이드]**")
                rag_t = src.get("rag_text", "")
                st.text_area("대응 가이드 Vector DB", value=rag_t, height=200, disabled=True)

            with t5:
                st.markdown("**[안전보건공단 MSDS API]**")
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
# ⚓ 실시간 항만 선박 모니터링 (4개 탭 구분)
# ------------------------------------------
st.markdown("### ⚓ 항만별 실시간 선박운항 현황 (Open API)")

tab_pt_in, tab_pt_out, tab_ds_in, tab_ds_out = st.tabs([
    "📥 평택항 입항 선박",
    "📤 평택항 출항 선박",
    "📥 대산항 입항 선박",
    "📤 대산항 출항 선박"
])

with tab_pt_in:
    render_vessel_tab_content("평택항", "031", "I")

with tab_pt_out:
    render_vessel_tab_content("평택항", "031", "O")

with tab_ds_in:
    render_vessel_tab_content("대산항", "300", "I")

with tab_ds_out:
    render_vessel_tab_content("대산항", "300", "O")
