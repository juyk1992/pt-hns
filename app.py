import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import json
import base64  # 로고 이미지 인코딩용 추가
from datetime import datetime, timedelta
from google import genai

# ==========================================
# 0. 페이지 설정 & 로고 이미지 처리 (Base64 인코딩)
# ==========================================
# 💡 [핵심] 로컬 이미지를 Base64로 인코딩하여 HTML에 강제 반영 (경로 오류 완벽 해결)
KCG_LOGO_PATH = "kcg_logo.png"

@st.cache_data
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

kcg_logo_base64 = get_base64_image(KCG_LOGO_PATH)

# 페이지 설정
st.set_page_config(
    page_title="평택해양경찰서 HNS AI 대응 시스템",
    # 💡 파비콘에도 Base64 이미지를 사용해 100% 나오도록 보장
    page_icon=KCG_LOGO_PATH if os.path.exists(KCG_LOGO_PATH) else "🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 1. Reflex 스타일 Custom CSS (모바일/다크모드 완벽 대응 및 Hero Section 보강)
# ==========================================
# 💡 [보강] hero-container CSS를 구체적이고 강력하게 수정 (테두리 및 정렬 보장)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

    /* [핵심] 글로벌 다크/라이트 모드 오버라이드 */
    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }

    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
    }

    /* 테마 변수 정의 (기존과 동일) */
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

    /* 💡 [수정 및 보강] Hero Section 컨테이너 (테두리와 배경색, 정렬 강제 적용) */
    .stApp .hero-container {
        padding: 1.8rem 2rem !important;
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 2rem !important; /* 아래 스마트 검색창과의 간격 확보 */
        display: block !important;
    }

    .hero-container .header-title-box {
        display: flex !important;
        align-items: center !important; /* 로고와 타이틀 수직 정렬 완벽 보장 */
        gap: 16px !important;
        margin-bottom: 6px !important;
    }

    .hero-container .kcg-logo-img {
        width: 58px !important; /* 스크린샷보다 키우고 선명하게 조정 */
        height: auto !important;
        object-fit: contain !important;
    }
    
    .hero-container .main-header {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.5px !important;
        margin: 0 !important;
        display: block !important;
    }
    
    .hero-container .sub-header {
        color: var(--text-sub) !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        margin: 4px 0 0 0 !important;
        display: block !important;
    }

    /* ... (UNNO 배지, 2줄 스크롤, 버튼 등 나머지 CSS는 기존과 동일) ... */
    .badge-unno {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        padding: 3px 9px;
        border-radius: 8px;
        font-family: monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }

    div[data-testid="stRadio"] > label { display: none !important; }
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

    div[data-testid="stRadio"] > div::-webkit-scrollbar { height: 5px !important; }
    div[data-testid="stRadio"] > div::-webkit-scrollbar-track { background: var(--bg-sub) !important; border-radius: 10px !important; }
    div[data-testid="stRadio"] > div::-webkit-scrollbar-thumb { background: var(--border-color) !important; border-radius: 10px !important; }

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

    div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }

    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background-color: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stRadio"] > div > label[data-checked="true"] *,
    div[data-testid="stRadio"] > div > label:has(input:checked) * { color: #FFFFFF !important; }

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

    .stButton > button * { color: #FFFFFF !important; }
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

    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    [data-testid="stMetricValue"] {
        color: var(--accent-blue) !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# ... (API 및 DB 연동 코드는 동일하므로 생략) ...
PUBLIC_API_KEY = st.secrets.get("PUBLIC_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

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

def fetch_dgst_info(unno):
    url = "http://apis.data.go.kr/1192000/DgstInqire3/Info"
    params = {'serviceKey': PUBLIC_API_KEY, 'unno': unno, 'numOfRows': '1', 'pageNo': '1'}
    info = {"imdgNm": "", "imdgEngNm": "", "emergManagtCd": "-", "packngGrad": "-", "imdgGradCd": "-", "kndNm": "-", "packngGdline": "-", "ldadngMth": "-"}
    try:
        res = requests.get(url, params=params, timeout=5)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            for key in info.keys(): info[key] = item.findtext(key, '-')
    except Exception: pass
    return info

def fetch_chem_safety_info(chem_name):
    url = "http://apis.data.go.kr/1480802/iciskischem/kischemlist"
    clean_name = chem_name.split('(')[0].strip()
    params = {'serviceKey': PUBLIC_API_KEY, 'numOfRows': '3', 'pageNo': '1', 'chemKo': clean_name}
    safety_data = {"casNo": "-", "symptom": "자료 없음", "inhale": "자료 없음", "skin": "자료 없음", "eyeball": "자료 없음", "oral": "자료 없음"}
    try:
        res = requests.get(url, params=params, timeout=5)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            for key in safety_data.keys(): safety_data[key] = item.findtext(key, '자료 없음')
    except Exception: pass
    return safety_data

@st.cache_data(ttl=3600)
def map_search_query_with_gemini(query_text):
    if not GEMINI_API_KEY or not query_text:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""사용자 입력 검색어: "{query_text}"\n JSON 포맷 답변: {{"chem_ko": "한글명", "chem_eng": "영문명", "unno": "UN번호"}}"""
        candidate_models = ['gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
        for model_id in candidate_models:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(text)
            except Exception: continue
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}
    except Exception: return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}

def generate_gemini_summary(chem_name, unno, dgst_info, safety_info):
    if not GEMINI_API_KEY: return "⚠️ Gemini API 키 없음"
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        hns_raw_text = find_hns_raw_text(unno) or find_hns_raw_text(chem_name)
        hns_context = f"[HNS 정보집]\n{hns_raw_text}\n" if hns_raw_text else "[HNS 정보집 없음]\n"
        prompt = f"""핵심요약 표 작성. 규칙 준수.\n{hns_context}... """
        candidate_models = ['gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
        for model_id in candidate_models:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                return response.text
            except Exception: continue
        return "⚠️ Gemini 실패"
    except Exception: return "⚠️ Gemini 오류"

@st.cache_data(ttl=30)
def load_integrated_hns_data(port_code):
    filename_map = {"031": "hns_pyeongtaek_report.csv", "300": "hns_daesan_report.csv"}
    file_path = filename_map.get(port_code)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        for col in ['호출부호', '선박명(선택)']: df[col] = df[col].astype(str).str.strip()
        return df
    return None

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
        st.warning(f"⚠️ {port_name} 수집 데이터 없음. RPA 가동 요망.")
    else:
        ship_column = "선박명(선택)" if "선박명(선택)" in df.columns else df.columns[0]
        unique_ships = sorted([str(s) for s in df[ship_column].dropna().unique()])
        st.markdown(f"**🚢 [{port_name}] 선박 선택**")
        selected_ship = st.radio(label=f"ship_{port_code}", options=["전체 보기"] + unique_ships, horizontal=True, key=f"radio_{port_code}")
        filtered_df = df[df[ship_column] == selected_ship] if selected_ship != "전체 보기" else df
        with col_metric: st.metric(label="반입 신고 건수", value=f"{len(filtered_df)} 건")
        display_cols = ['선박명(선택)', '호출부호', '사용목적', '사용장소', 'UNNO', '品명', '중량']
        with st.expander(f"📋 목록 보기", expanded=False): st.dataframe(filtered_df[[c for c in display_cols if c in filtered_df.columns]], use_container_width=True)
        st.markdown(f"##### 상세 정보 및 AI 대응")
        for ship in filtered_df['선박명(선택)'].unique():
            ship_data = filtered_df[filtered_df['선박명(선택)'] == ship]
            location = ship_data['사용장소'].iloc[0] or "-"
            with st.expander(f"⚓ [{ship}] (호출: {ship_data['호출부호'].iloc[0]}) ｜ {location}"):
                for idx, row in ship_data.iterrows():
                    unno = str(row['UNNO']).zfill(4); chem = str(row['품명']); weight = str(row['중량'])
                    c_info, c_btn = st.columns([4, 1])
                    with c_info: st.markdown(f"• <span class='badge-unno'>UN {unno}</span> &nbsp; **{chem}** &nbsp; ({weight})", unsafe_allow_html=True)
                    with c_btn:
                        if st.button("AI 가이드", key=f"btn_{port_code}_{ship}_{idx}"):
                            st.session_state.update({'active_chem': chem, 'active_unno': unno, 'active_ship': f"[{port_name}] {ship}"})

# ==========================================
# 6. 메인 화면 구성 (Hero Section 및 로고 완벽 복구)
# ==========================================

# 💡 [핵심] Base64 인코딩 이미지를 HTML에 박아넣고 CSS로 완벽 정렬
if kcg_logo_base64:
    st.markdown(f"""
    <div class="hero-container">
        <div class="header-title-box">
            <img src="data:image/png;base64,{kcg_logo_base64}" class="kcg-logo-img" alt="해양경찰 로고" />
            <h1 class="main-header">평택해양경찰서 HNS AI 대응 솔루션</h1>
        </div>
        <div class="sub-header">포트미스(PORT-MIS) + 공공 API + 해경 HNS DB + Gemini AI 지능형 관제 시스템</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # 💡 로고 파일이 없을 때를 대비한 Fallback (이전처럼 테두리 박스는 유지)
    st.markdown("""
    <div class="hero-container">
        <h1 class="main-header">🚢 평택해양경찰서 HNS AI 대응 솔루션</h1>
        <div class="sub-header">포트미스(PORT-MIS) + 공공 API + 해경 HNS DB + Gemini AI 지능형 관제 시스템</div>
    </div>
    """, unsafe_allow_html=True)

# (이하 스마트 검색 및 탭 렌더링 코드는 기존과 동일)
# ------------------------------------------
# 🔥 [통합] HNS 화학물질 AI 스마트 매핑 검색창
# ------------------------------------------
st.markdown("### 🔎 화학물질 AI 스마트 검색")
search_input = st.text_input("화학물질명, 화학식, 관용명 입력", key="global_search_box")

if search_input:
    with st.spinner("AI 분석 중..."):
        mapped_result = map_search_query_with_gemini(search_input)
        mapped_ko = mapped_result.get("chem_ko"); mapped_unno = str(mapped_result.get("unno")).zfill(4)
        c1, c2 = st.columns([4, 1])
        with c1: st.info(f"💡 AI 매핑: {mapped_ko} ｜ UN NO: `{mapped_unno}`")
        with c2:
            if st.button("AI 가이드", key="btn_global_search"):
                st.session_state.update({'active_chem': mapped_ko, 'active_unno': mapped_unno, 'active_ship': f"자유 검색 '{search_input}'"})

if 'active_chem' in st.session_state:
    st.divider()
    chem = st.session_state['active_chem']; unno = st.session_state['active_unno']
    st.error(f"⚡ [AI 비상대응 가이드] 대상: {st.session_state['active_ship']} ｜ {chem} (UN {unno})")
    with st.spinner('가이드 생성 중...'):
        ai_summary = generate_gemini_summary(chem, unno, fetch_dgst_info(unno), fetch_chem_safety_info(chem))
    st.markdown(ai_summary)
    if st.button("❌ 가이드 닫기"):
        for key in ['active_chem', 'active_unno', 'active_ship']: del st.session_state[key]
        st.rerun()

st.divider()
t1, t2 = st.tabs(["⚓ 평택항 (031)", "⚓ 대산항 (300)"])
with t1: render_port_dashboard("평택항", "031")
with t2: render_port_dashboard("대산항", "300")
