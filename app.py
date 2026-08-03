import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import json
import base64
from datetime import datetime, timedelta
from google import genai

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

    /* ==========================================
       1. 기본 라이트 모드 테마 변수 정의
       ========================================== */
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

    /* ==========================================
       2. 다크 모드 자동 감지 및 변수 재정의
       ========================================== */
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

    /* Streamlit 테마 설정이 다크일 때도 강제 반영 */
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

    /* ==========================================
       3. 동적 변수 기반 글로벌 스타일 적용
       ========================================== */
    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }

    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
    }

    /* Hero 컨테이너 */
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

    /* UNNO 배지 */
    .badge-unno {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        padding: 3px 9px;
        border-radius: 8px;
        font-family: monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* 2줄 고정 가로 스크롤 선박 칩(Radio) */
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

    /* 선택된 칩 */
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

    /* 입력창 */
    .stTextInput input {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    /* 버튼 */
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

    /* Expander(아코디언) */
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

    /* 탭(Tabs) */
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
# 1. HNS 정보집 원본 텍스트 DB 로드
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

# ==========================================
# 2. 공공 API 연동 모듈 (수정 완료)
# ==========================================
def fetch_dgst_info(unno):
    """
    [해양수산부 위험물정보 API]
    수집 항목: imdgNm, imdgEngNm, kndNm, kndPrdlstNm, imdgGradCd, emergManagtCd, ldadngMth, catinMatter
    """
    url = "http://apis.data.go.kr/1192000/DgstInqire3/Info"
    params = {'serviceKey': PUBLIC_API_KEY, 'unno': unno, 'numOfRows': '1', 'pageNo': '1'}
    info = {
        "imdgNm": "", "imdgEngNm": "", "kndNm": "-", "kndPrdlstNm": "-",
        "imdgGradCd": "-", "emergManagtCd": "-", "ldadngMth": "-", "catinMatter": "-"
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            info['imdgNm'] = item.findtext('imdgNm', '')
            info['imdgEngNm'] = item.findtext('imdgEngNm', '')
            info['kndNm'] = item.findtext('kndNm', '-')
            info['kndPrdlstNm'] = item.findtext('kndPrdlstNm', '-')
            info['imdgGradCd'] = item.findtext('imdgGradCd', '-')
            info['emergManagtCd'] = item.findtext('emergManagtCd', '-')
            info['ldadngMth'] = item.findtext('ldadngMth', '-')
            info['catinMatter'] = item.findtext('catinMatter', '-')
    except Exception as e:
        print(f"위험물정보 API 에러: {e}")
    return info

def fetch_chem_safety_info(cas_no):
    """
    [화학물질안전원 화학물질 안전관리정보 API]
    요청 파라미터: serviceKey, numOfRows, pageNo, casNo
    응답 수집 항목: symptom, inhale, skin, eyeball, oral, etc
    """
    url = "http://apis.data.go.kr/1480802/iciskischem/kischemlist"
    
    # casNo가 올바르지 않거나 없을 경우 예외 처리
    if not cas_no or cas_no in ["-", "0000", "없음"]:
        return {
            "symptom": "자료 없음", "inhale": "자료 없음", "skin": "자료 없음",
            "eyeball": "자료 없음", "oral": "자료 없음", "etc": "자료 없음"
        }

    clean_cas = str(cas_no).strip()
    params = {'serviceKey': PUBLIC_API_KEY, 'numOfRows': '3', 'pageNo': '1', 'casNo': clean_cas}
    safety_data = {
        "symptom": "자료 없음", "inhale": "자료 없음", "skin": "자료 없음", 
        "eyeball": "자료 없음", "oral": "자료 없음", "etc": "자료 없음"
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            safety_data['symptom'] = item.findtext('symptom', '자료 없음')
            safety_data['inhale'] = item.findtext('inhale', '자료 없음')
            safety_data['skin'] = item.findtext('skin', '자료 없음')
            safety_data['eyeball'] = item.findtext('eyeball', '자료 없음')
            safety_data['oral'] = item.findtext('oral', '자료 없음')
            safety_data['etc'] = item.findtext('etc', '자료 없음')
    except Exception as e:
        print(f"화학물질 안전관리정보 API 에러: {e}")
    return safety_data

# ==========================================
# 3. Gemini 자연어 매핑 및 AI 요약 (CAS 번호 추가)
# ==========================================
@st.cache_data(ttl=3600)
def map_search_query_with_gemini(query_text):
    if not GEMINI_API_KEY or not query_text:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000", "cas_no": "-"}

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        사용자 입력 검색어: "{query_text}"
        이 화학물질/관용명/화학식에 해당하는 가장 대표적인 위험물/HNS 화학물질의 표준 정보를 찾아 아래 JSON 포맷으로만 답변하세요. 다른 설명은 금지합니다:
        {{"chem_ko": "공식 한글명", "chem_eng": "공식 영문명", "unno": "4자리 UN번호", "cas_no": "CAS번호(예: 7664-93-9)"}}
        """
        
        candidate_models = ['gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
        for model_id in candidate_models:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(text)
            except Exception:
                continue

        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000", "cas_no": "-"}
    except Exception:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000", "cas_no": "-"}

def generate_gemini_summary(chem_name, unno, cas_no, dgst_info, safety_info):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API 키가 설정되지 않았습니다."
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        hns_raw_text = find_hns_raw_text(unno) or find_hns_raw_text(chem_name)
        hns_context = f"[해양경찰청 HNS 정보집 원본 문서 정보]\n{hns_raw_text}\n" if hns_raw_text else "[해양경찰청 HNS 정보집 정보]\n매칭 데이터 참조\n"

        prompt = f"""
        당신은 해양경찰청 및 항만 HNS 비상대응 상황실 관제관입니다.
        제공된 데이터들을 종합 분석하여 현장 대응 가이드를 작성하세요.

        {hns_context}

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
        - 흡입 영향: {safety_info.get('inhale', '-')}
        - 피부 노출 영향: {safety_info.get('skin', '-')}
        - 안구 노출 영향: {safety_info.get('eyeball', '-')}
        - 경구 섭취 영향: {safety_info.get('oral', '-')}
        - 기타 유의사항: {safety_info.get('etc', '-')}

        [작성 핵심 규칙]
        ★ 최상단 핵심요약문은 **장황한 설명글을 절대 금지**하며, 현장 요원이 보고 1초만에 지시/전파할 수 있도록 **핵심 키워드, 수치(M단위), 구체적 단어 위주로 극도로 간결하게 표 형태로 작성**하세요.

        --- 출력 형식을 엄격히 준수하세요 ---

        ### 🚨 [초동대응 핵심요약]
        | 구분 | 핵심 대응 내용 |
        |---|---|
        | **사고물질/위험성** | [IMDG 등급] + [핵심위험: 예) 인화성/독성가스/수반응성] |
        | **대피/이격거리** | **초기이격:** OOm / **화재대피:** OOm / **유출방호:** OOm (HNS 정보집 수치 필수) |
        | **필수 보호구** | **[Level A/C]** + [공기호흡기/내화학복/복합가스탐지기 등 필수장비] |
        | **초동 행동수칙** | **[풍상위치]** + [핵심금지사항 및 소화약제 요약] |

        ---
        ### 1. ⚠️ 물리·화학적 성상 및 주요 위험성 (EmS/적재방법/주의사항)
        ### 2. 🛡️ 현장 개인 보호구 및 초동 방제/소화 요령
        ### 3. ⛔ 절대 금지 행동 (금기 사항 - 물 접촉 금지, 직사주수 금지 등)
        ### 4. 🏥 인체 노출 시 신체 영향 및 긴급 응급조치 (흡입/피부/안구/경구/기타)
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
# 5. 항구별 데이터 렌더링 헬퍼 함수 (CAS_NO 적용)
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
                            # AI가 CAS 번호까지 정밀 추출하도록 바인딩
                            mapped_info = map_search_query_with_gemini(chem_name)
                            st.session_state['active_chem'] = mapped_info.get("chem_ko", chem_name)
                            st.session_state['active_unno'] = unno
                            st.session_state['active_cas'] = mapped_info.get("cas_no", "-")
                            st.session_state['active_ship'] = f"[{port_name}] {ship}"

# ==========================================
# 6. 메인 화면 구성 (Hero Section & 로고 정렬)
# ==========================================
if kcg_logo_b64:
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 6px;">
            <img src="data:image/png;base64,{kcg_logo_b64}" style="width: 58px; height: auto; object-fit: contain;" alt="해양경찰 로고" />
            <div class="main-header" style="margin: 0;">평택해양경찰서 HNS AI 대응 솔루션</div>
        </div>
        <div class="sub-header">포트미스(PORT-MIS) + 공공 API + 해경 HNS DB + Gemini AI 지능형 관제 시스템</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-container">
        <div class="main-header">🚢 평택해양경찰서 HNS AI 대응 솔루션</div>
        <div class="sub-header">포트미스(PORT-MIS) + 공공 API + 해경 HNS DB + Gemini AI 지능형 관제 시스템</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# 🔥 [통합] HNS 화학물질 AI 스마트 매핑 검색창
# ------------------------------------------
st.markdown("### 🔎 화학물질 AI 스마트 검색")
search_input = st.text_input(
    "화학물질명, 화학식, 관용명을 입력하세요 (예: H2SO4, 황산, 가성소다, LNG, 수산화나트륨)", 
    key="global_search_box"
)

if search_input:
    with st.spinner("Gemini AI가 화학물질을 정밀 분석 중..."):
        mapped_result = map_search_query_with_gemini(search_input)
        mapped_ko = mapped_result.get("chem_ko", search_input)
        mapped_eng = mapped_result.get("chem_eng", search_input)
        mapped_unno = str(mapped_result.get("unno", "0000")).zfill(4)
        mapped_cas = str(mapped_result.get("cas_no", "-"))
        
        c1, c2 = st.columns([4, 1])
        with c1:
            st.info(f"💡 **AI 매핑 결과:** 물질명: **{mapped_ko}** ({mapped_eng}) ｜ UN NO: `{mapped_unno}` ｜ CAS NO: `{mapped_cas}`")
        with c2:
            if st.button("🤖 AI 가이드 생성", key="btn_global_search", use_container_width=True):
                st.session_state['active_chem'] = mapped_ko
                st.session_state['active_unno'] = mapped_unno
                st.session_state['active_cas'] = mapped_cas
                st.session_state['active_ship'] = f"자유 통합 검색 ('{search_input}')"

# AI 대응 가이드 출력 모달/컨테이너
if 'active_chem' in st.session_state:
    st.divider()
    chem = st.session_state['active_chem']
    unno = st.session_state['active_unno']
    cas = st.session_state.get('active_cas', '-')
    ship_info = st.session_state['active_ship']
    
    st.error(f"⚡ [지능형 비상대응 가이드] 대상: {ship_info} ｜ 물질: {chem} (UN NO: {unno} / CAS NO: {cas})")
    
    with st.spinner('공공 API + 해경 HNS 정보집 + Gemini AI 가이드 생성 중...'):
        dgst_info = fetch_dgst_info(unno)
        safety_info = fetch_chem_safety_info(cas)  # casNo 파라미터를 정상 활용!
        ai_summary = generate_gemini_summary(chem, unno, cas, dgst_info, safety_info)
        
    st.markdown(ai_summary)
    
    if st.button("❌ 가이드 창 닫기", key="close_global_guide", use_container_width=True):
        del st.session_state['active_chem']
        del st.session_state['active_unno']
        if 'active_cas' in st.session_state:
            del st.session_state['active_cas']
        del st.session_state['active_ship']
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
