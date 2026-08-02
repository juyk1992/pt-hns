import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import json
from datetime import datetime, timedelta
from google import genai

# ==========================================
# 0. 페이지 설정 & Custom CSS (세련된 UI 적용)
# ==========================================
st.set_page_config(
    page_title="평택해양경찰서 HNS AI 대응 시스템",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS 적용
st.markdown("""
<style>
    /* 전체 배경 및 폰트 감성 적용 */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    /* 타이틀 및 헤더 */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-bottom: 25px;
    }
    /* 배지 스타일 */
    .badge-port {
        background-color: #0284C7;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-unno {
        background-color: #E11D48;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: bold;
    }
    /* 버튼 스타일 통일 */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
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
    """UN번호 또는 물질명으로 로컬 HNS 정보집 텍스트 추출"""
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
# 2. 공공 API 연동 모듈
# ==========================================
def fetch_dgst_info(unno):
    """해수부 위험물정보 API 상세 파싱"""
    url = "http://apis.data.go.kr/1192000/DgstInqire3/Info"
    params = {'serviceKey': PUBLIC_API_KEY, 'unno': unno, 'numOfRows': '1', 'pageNo': '1'}
    info = {
        "imdgNm": "", "imdgEngNm": "", "emergManagtCd": "-", "packngGrad": "-",
        "imdgGradCd": "-", "kndNm": "-", "packngGdline": "-", "ldadngMth": "-"
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            info['imdgNm'] = item.findtext('imdgNm', '')
            info['imdgEngNm'] = item.findtext('imdgEngNm', '')
            info['emergManagtCd'] = item.findtext('emergManagtCd', '-')
            info['packngGrad'] = item.findtext('packngGrad', '-')
            info['imdgGradCd'] = item.findtext('imdgGradCd', '-')
            info['kndNm'] = item.findtext('kndNm', '-')
            info['packngGdline'] = item.findtext('packngGdline', '-')
            info['ldadngMth'] = item.findtext('ldadngMth', '-')
    except Exception as e:
        print(f"위험물정보 API 에러: {e}")
    return info

def fetch_chem_safety_info(chem_name):
    """화학물질안전원 안전관리정보 API 상세 파싱"""
    url = "http://apis.data.go.kr/1480802/iciskischem/kischemlist"
    clean_name = chem_name.split('(')[0].strip()
    params = {'serviceKey': PUBLIC_API_KEY, 'numOfRows': '3', 'pageNo': '1', 'chemKo': clean_name}
    safety_data = {
        "casNo": "-", "symptom": "자료 없음", "inhale": "자료 없음", 
        "skin": "자료 없음", "eyeball": "자료 없음", "oral": "자료 없음"
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            safety_data['casNo'] = item.findtext('casNo', '-')
            safety_data['symptom'] = item.findtext('symptom', '자료 없음')
            safety_data['inhale'] = item.findtext('inhale', '자료 없음')
            safety_data['skin'] = item.findtext('skin', '자료 없음')
            safety_data['eyeball'] = item.findtext('eyeball', '자료 없음')
            safety_data['oral'] = item.findtext('oral', '자료 없음')
    except Exception as e:
        print(f"화학물질 안전관리정보 API 에러: {e}")
    return safety_data

# ==========================================
# 3. Gemini 자연어 매핑 및 고속 AI 요약 모듈
# ==========================================
@st.cache_data(ttl=3600)
def map_search_query_with_gemini(query_text):
    """Gemini API로 다이렉트 자연어 매핑 (캐싱 적용)"""
    if not GEMINI_API_KEY or not query_text:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        사용자 입력 검색어: "{query_text}"
        이 화학물질/관용명/화학식에 해당하는 가장 대표적인 위험물/HNS 화학물질의 표준 정보를 찾아 아래 JSON 포맷으로만 답변하세요. 다른 설명은 금지합니다:
        {{"chem_ko": "공식 한글명", "chem_eng": "공식 영문명", "unno": "4자리 UN번호"}}
        """
        
        candidate_models = ['gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
        for model_id in candidate_models:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(text)
            except Exception:
                continue

        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}
    except Exception:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}

def generate_gemini_summary(chem_name, unno, dgst_info, safety_info):
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
        - IMDG 등급코드: {dgst_info.get('imdgGradCd', '-')} ({dgst_info.get('kndNm', '-')})
        - 포장등급 / 포장지침: {dgst_info.get('packngGrad', '-')} / {dgst_info.get('packngGdline', '-')}
        - 비상조치코드(EmS): {dgst_info.get('emergManagtCd', '-')}
        - 선박 적재방법: {dgst_info.get('ldadngMth', '-')}

        [화학물질안전원 안전관리정보 API 수집 데이터]
        - CAS 번호: {safety_info.get('casNo', '-')}
        - 일반 증상 및 표적장기: {safety_info.get('symptom', '-')}
        - 흡입/피부/안구/경구 영향: {safety_info.get('inhale', '-')}, {safety_info.get('skin', '-')}, {safety_info.get('eyeball', '-')}, {safety_info.get('oral', '-')}

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
        ### 1. ⚠️ 물리·화학적 성상 및 주요 위험성 (EmS/적재방법)
        ### 2. 🛡️ 현장 개인 보호구 및 초동 방제/소화 요령
        ### 3. ⛔ 절대 금지 행동 (금기 사항 - 물 접촉 금지, 직사주수 금지 등)
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
# 4. 항구별 RPA 데이터 로드
# ==========================================
@st.cache_data
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
# 5. 항구별 데이터 렌더링 헬퍼 함수 (선박 선택 필터 박스 포함)
# ==========================================
def render_port_dashboard(port_name, port_code):
    kst_now = datetime.utcnow() + timedelta(hours=9)
    today_str = kst_now.strftime("%Y-%m-%d")
    
    # 데이터 로드
    df = load_integrated_hns_data(port_code)

    col_title, col_metric = st.columns([3, 1])
    with col_title:
        st.subheader(f"📊 {port_name} 위험물 반입 현황")
        st.caption(f"조회 기준일자: {today_str}")

    if df is None or df.empty:
        st.warning(f"⚠️ {port_name}의 {today_str} 기준 수집 데이터가 없습니다. RPA 봇을 작동시켜 주세요.")
    else:
        # 💡 [핵심 추가] 선박 선택 필터 박스 영역
        ship_column = "선박명(선택)" if "선박명(선택)" in df.columns else df.columns[0]
        unique_ships = sorted([str(s) for s in df[ship_column].dropna().unique()])
        
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            selected_ships = st.multiselect(
                f"🚢 [{port_name}] 조회할 선박을 선택하세요 (다중 선택 가능)",
                options=unique_ships,
                default=[],
                key=f"select_ship_{port_code}",
                placeholder="선박명을 선택하세요 (미선택 시 전체 표시)"
            )
        
        # 선택 여부에 따른 데이터 필터링
        if selected_ships:
            filtered_df = df[df[ship_column].isin(selected_ships)]
        else:
            filtered_df = df

        with col_metric:
            st.metric(label="조회된 위험물 신고 건수", value=f"{len(filtered_df)} 건")

        # 전체 반입 신고 목록 Expander
        display_cols = [
            '선박명(선택)', '호출부호', '사용목적', '운송형태', '화물명', 
            '하역업체', '하역기간', '사용장소', '전출항지', 'UNNO', 'IMDG', '품명', '중량', '단위'
        ]
        
        with st.expander(f"📋 {port_name} 위험물 반입 신고 목록 데이터 보기", expanded=False):
            st.dataframe(filtered_df[[c for c in display_cols if c in filtered_df.columns]], use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"🚢 {port_name} 선박별 상세 운송 정보 및 지능형 비상 대응")

        # 필터링된 선박별 카드 Expander UI
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
                st.markdown("##### 📦 적재 위험물 목록")
                
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
                        st.markdown(f"• <span class='badge-unno'>UN {unno}</span> &nbsp; **{chem_name}** &nbsp; (IMDG: {imdg} / 수량: {weight} {unit})", unsafe_allow_html=True)
                    with c_btn:
                        button_key = f"btn_{port_code}_{ship}_{idx}_{unno}"
                        if st.button("🤖 AI 가이드 생성", key=button_key, use_container_width=True):
                            st.session_state['active_chem'] = chem_name
                            st.session_state['active_unno'] = unno
                            st.session_state['active_ship'] = f"[{port_name}] {ship}"

# ==========================================
# 6. 메인 화면 구성
# ==========================================

# 메인 타이틀
st.markdown('<div class="main-header">🚢 평택해양경찰서 HNS AI 대응 솔루션</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">포트미스 + 공공 API + 해경 HNS DB + Gemini AI 지능형 관제 시스템</div>', unsafe_allow_html=True)

# ------------------------------------------
# 🔥 [통합] HNS 화학물질 AI 스마트 매핑 검색창 (항 구분 없음)
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
        
        c1, c2 = st.columns([4, 1])
        with c1:
            st.info(f"💡 **AI 매핑 결과:** 물질명: **{mapped_ko}** ({mapped_eng}) ｜ UN NO: `{mapped_unno}`")
        with c2:
            if st.button("🤖 AI 가이드 생성", key="btn_global_search", use_container_width=True):
                st.session_state['active_chem'] = mapped_ko
                st.session_state['active_unno'] = mapped_unno
                st.session_state['active_ship'] = f"자유 통합 검색 ('{search_input}')"

# AI 대응 가이드 출력 모달/컨테이너 (통합 검색 및 선박 클릭 공용)
if 'active_chem' in st.session_state:
    st.divider()
    chem = st.session_state['active_chem']
    unno = st.session_state['active_unno']
    ship_info = st.session_state['active_ship']
    
    st.error(f"⚡ [지능형 비상대응 가이드] 대상: {ship_info} ｜ 물질: {chem} (UN NO: {unno})")
    
    with st.spinner('공공 API + 해경 HNS 정보집 + Gemini AI 가이드 생성 중...'):
        dgst_info = fetch_dgst_info(unno)
        safety_info = fetch_chem_safety_info(chem)
        
        ai_summary = generate_gemini_summary(chem, unno, dgst_info, safety_info)
        
    st.markdown(ai_summary)
    
    if st.button("❌ 가이드 창 닫기", key="close_global_guide", use_container_width=True):
        del st.session_state['active_chem']
        del st.session_state['active_unno']
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
