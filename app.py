import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
import json
from datetime import datetime, timedelta
from google import genai

# ==========================================
# 0. 기본 화면 설정 및 API 키 설정
# ==========================================
st.set_page_config(page_title="평택항 HNS 안전관리 시스템", page_icon="🚢", layout="wide")

PUBLIC_API_KEY = st.secrets.get("PUBLIC_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# ==========================================
# 1. HNS 정보집 원본 텍스트 DB 로드 및 검색
# ==========================================
@st.cache_data
def load_full_hns_db():
    """hns_full_text_database.json 전체 원본 텍스트 데이터 로드"""
    file_path = "hns_full_text_database.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

full_hns_db = load_full_hns_db()

def find_hns_raw_text(query):
    """UN번호, 국문명, 영문명, 관용명 기반 로컬 HNS 원본 텍스트 매칭"""
    if not full_hns_db or not query:
        return None
        
    q = query.strip().upper()
    for item in full_hns_db:
        unno = item.get('unno', '')
        title = item.get('title_header', '').upper()
        synonyms = item.get('synonyms', '').upper()
        
        # UN번호 완전치, 제목/유사명 부분 일치 탐색
        if (q and q == unno) or (q in title) or (q in synonyms):
            return item.get('raw_full_text', '')
    return None

# ==========================================
# 2. 공공 API 연동 모듈 (확장 항목 파싱)
# ==========================================
def fetch_dgst_info(unno):
    """해수부 위험물정보 API 상세 항목 파싱"""
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
    """화학물질안전원 안전관리정보 API 상세 항목 파싱"""
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
# 3. Gemini 자연어 매핑 및 풍부화된 AI 요약 모듈
# ==========================================
def map_search_query_with_gemini(query_text):
    """자연어/화학식/관용명 검색어를 공식 물질명 및 UNNO로 정제"""
    # 1. 로컬 DB 텍스트에서 빠른 매칭 시도
    raw_match = find_hns_raw_text(query_text)
    if raw_match:
        # 제목 첫 줄 등에서 추출
        lines = [l.strip() for l in raw_match.split('\n') if l.strip()]
        title = lines[0] if lines else query_text
        return {"chem_ko": title, "chem_eng": title, "unno": "0000"}

    if not GEMINI_API_KEY:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        사용자 입력 검색어: "{query_text}"
        이 화학물질의 표준 정보를 찾아서 아래 JSON 포맷으로만 답변하세요. 다른 설명 금지:
        {{"chem_ko": "공식 한국어명", "chem_eng": "공식 영문명", "unno": "4자리 UN번호"}}
        """
        response = client.models.generate_content(model='gemini-3.5-flash', contents=prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception:
        return {"chem_ko": query_text, "chem_eng": query_text, "unno": "0000"}

def generate_gemini_summary(chem_name, unno, dgst_info, safety_info):
    """HNS 정보집 원본 텍스트 + 공공 API 전체 수집 항목 종합 요약"""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API 키가 설정되지 않았습니다."
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # HNS 원본 텍스트 추출
        hns_raw_text = find_hns_raw_text(unno) or find_hns_raw_text(chem_name)
        hns_context = f"[해양경찰청 HNS 정보집 원본 문서 정보]\n{hns_raw_text}\n" if hns_raw_text else "[해양경찰청 HNS 정보집 정보]\n매칭 데이터 참조\n"

        prompt = f"""
        당신은 해양경찰청 및 항만 HNS(위험유해물질) 비상대응 상황실 관제관입니다.
        아래 제공된 [해양경찰청 HNS 정보집 원본]과 [공공 API 확장 수집 데이터]를 종합적으로 완벽하게 분석하여 현장 대응가이드를 작성하세요.

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
        - 흡입 노출 영향: {safety_info.get('inhale', '-')}
        - 피부 접촉 영향: {safety_info.get('skin', '-')}
        - 안구 접촉 영향: {safety_info.get('eyeball', '-')}
        - 경구 섭취 영향: {safety_info.get('oral', '-')}

        [작성 규칙]
        1. 가이드 최상단에 상황실에서 3초 만에 파악하고 전파할 수 있는 핵심 요약을 작성하세요.
        2. 해경 HNS 정보집의 대피거리 수치(M단위), 초기이격거리, 개인보호구 레벨(Level A/C 등)을 우선 적용하세요.
        3. 아래 구조로 명확히 출력하세요:

        ### 🚨 [상황실 전파 및 초동대응 핵심 요약] (3초 파악용)
        * **사고물질 및 주요위험:** (IMDG 등급, 화재/폭발/독성가스/물반응성 위험성)
        * **초기 이격 및 대피거리:** (HNS 정보집 기준 화재대피거리 및 해상유출 이격거리/방호활동거리 M단위 정확히 명시)
        * **필수 개인보호구 및 장비:** (레벨 A/C 등급, 내화학복, 공기호흡기, 복합가스탐지기 등)
        * **초동 조치 핵심:** (풍상 위치, 오염원 접근 주의, 방제/소화 약제 수칙)

        ---
        ### 1. ⚠️ 물리·화학적 성상 및 주요 위험성 (EmS/적재방법 포함)
        ### 2. 🛡️ 현장 개인 보호구 및 초동 방제/소화 요령
        ### 3. ⛔ 절대 금지 행동 (금기 사항 - 직사주수 금지, 물 접촉 금지 등)
        ### 4. 🏥 인체 노출 시 신체 영향 및 긴급 응급조치 (흡입/피부/안구/경구)
        """

        candidate_models = ['gemini-3.5-flash', 'gemini-3.6-flash', 'gemini-2.5-flash']
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
# 4. RPA 통합 데이터 로드
# ==========================================
@st.cache_data
def load_integrated_hns_data():
    file_path = "hns_fully_integrated_report.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['호출부호'] = df['호출부호'].astype(str).str.strip()
        df['선박명(선택)'] = df['선박명(선택)'].astype(str).str.strip()
        return df
    return None

# ==========================================
# 5. Streamlit UI 화면 구성
# ==========================================
st.title("🚨 평택항 HNS 실시간 안전관리 및 지능형 대응 시스템")
st.markdown("**포트미스 RPA + 공공 API + 해경 HNS 정보집 DB + Gemini AI 통합 솔루션**")

kst_now = datetime.utcnow() + timedelta(hours=9)
today_str = kst_now.strftime("%Y-%m-%d")

# 1) 메인 검색창 (자연어/화학식 실시간 매핑)
st.markdown("### 🔎 HNS 물질 자연어/화학식 실시간 통합 검색")
search_input = st.text_input("화학물질명, 화학식, 관용명을 입력하세요 (예: NaOH, 수산화나트륨, 가성소다, LNG)", key="main_search_box")

if search_input:
    with st.spinner("HNS DB 및 AI 분석 중..."):
        mapped_result = map_search_query_with_gemini(search_input)
        mapped_ko = mapped_result.get("chem_ko", search_input)
        mapped_eng = mapped_result.get("chem_eng", search_input)
        mapped_unno = str(mapped_result.get("unno", "0000")).zfill(4)
        
        st.success(f"🔍 **AI/DB 매핑 완료:** 물질명: **{mapped_ko}** ({mapped_eng}) ｜ UN NO: `{mapped_unno}`")
        
        if st.button("🤖 검색 물질 AI 대응 가이드 생성", key="btn_custom_search"):
            st.session_state['active_chem'] = mapped_ko
            st.session_state['active_unno'] = mapped_unno
            st.session_state['active_ship'] = f"자유 검색 ('{search_input}')"

st.divider()

# 2) 반입 현황 데이터 표출 (기준일자 명시)
df = load_integrated_hns_data()
st.subheader(f"📊 평택항 위험물 반입 현황 (조회 기준일자: {today_str})")

if df is None or df.empty:
    st.warning(f"⚠️ {today_str} 기준 데이터가 없습니다. RPA 봇을 돌려주세요!")
else:
    st.sidebar.header("🔍 검색 및 필터")
    selected_ship = st.sidebar.selectbox("선박 선택", ["전체보기"] + list(df['선박명(선택)'].unique()))
    
    filtered_df = df[df['선박명(선택)'] == selected_ship] if selected_ship != "전체보기" else df

    st.write(f"총 **{len(filtered_df)}건**의 위험물 반입 신고가 등록되어 있습니다.")
    st.dataframe(filtered_df[['선박명(선택)', '호출부호', '하역업체', '하역기간', '사용장소', 'UNNO', '품명', '중량', '단위', '수하인', '송하인']], use_container_width=True)

    st.divider()
    st.subheader("🚢 선박별 상세 운송 정보 및 지능형 비상 대응")

    for ship in filtered_df['선박명(선택)'].unique():
        ship_data = filtered_df[filtered_df['선박명(선택)'] == ship]
        call_sign = ship_data['호출부호'].iloc[0]
        location = ship_data['사용장소'].iloc[0] if pd.notna(ship_data['사용장소'].iloc[0]) else "지정 장소 미상"
        work_period = ship_data['하역기간'].iloc[0]
        
        with st.expander(f"⚓ [{ship}] (호출부호: {call_sign}) ｜ 하역장소: {location} ｜ 기간: {work_period}"):
            st.markdown(f"**🏢 하역업체:** {ship_data['하역업체'].iloc[0]} ｜ **반입구분:** {ship_data['반입구분'].iloc[0]}")
            st.markdown("---")
            st.markdown("#### 📦 적재된 위험물 목록")
            
            for idx, row in ship_data.iterrows():
                unno = str(row['UNNO']).zfill(4)
                chem_name = str(row['품명'])
                weight = str(row['중량'])
                unit = str(row['단위'])
                
                if unno == 'nan' or not unno.strip():
                    continue
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🔹 **UN NO:** `{unno}` ｜ **품명:** **{chem_name}** ｜ **수량:** {weight} {unit}")
                with col2:
                    button_key = f"btn_{ship}_{idx}_{unno}"
                    if st.button("🤖 AI 대응 가이드 생성", key=button_key):
                        st.session_state['active_chem'] = chem_name
                        st.session_state['active_unno'] = unno
                        st.session_state['active_ship'] = ship

# ==========================================
# 6. AI 대응 가이드 출력 영역 (체크리스트 완전 제거)
# ==========================================
if 'active_chem' in st.session_state:
    st.divider()
    chem = st.session_state['active_chem']
    unno = st.session_state['active_unno']
    
    st.error(f"⚡ [지능형 비상대응 가이드] 대상 선박: {st.session_state['active_ship']} ｜ 물질: {chem} (UN NO: {unno})")
    
    with st.spinner('해수부 API + 화학물질안전원 API + 해경 HNS 정보집 DB 기반 AI 가이드 생성 중...'):
        dgst_info = fetch_dgst_info(unno)
        safety_info = fetch_chem_safety_info(chem)
        
        ai_summary = generate_gemini_summary(chem, unno, dgst_info, safety_info)
        
    st.markdown(ai_summary)
    
    if st.button("🔄 가이드 닫기"):
        del st.session_state['active_chem']
        del st.session_state['active_unno']
        del st.session_state['active_ship']
        st.rerun()
