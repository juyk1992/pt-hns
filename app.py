import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
from google import genai

# ==========================================
# 0. 기본 화면 설정 및 API 키 설정
# ==========================================
st.set_page_config(page_title="평택항 HNS 안전관리 시스템", page_icon="🚢", layout="wide")

# 공공데이터 포털 인증키 및 Gemini API 키 설정
PUBLIC_API_KEY = st.secrets["PUBLIC_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ==========================================
# 1. 공공 API 연동 모듈 (위험물정보 + 화학물질안전관리정보)
# ==========================================
def fetch_dgst_info(unno):
    """
    1단계: UN NO를 이용해 해양수산부 '위험물정보 서비스' 조회
    """
    url = "http://apis.data.go.kr/1192000/DgstInqire3/Info"
    params = {
        'serviceKey': PUBLIC_API_KEY,
        'unno': unno,
        'numOfRows': '1',
        'pageNo': '1'
    }
    
    info = {"imdgNm": "", "imdgEngNm": "", "emergManagtCd": "-", "packngGrad": "-"}
    try:
        res = requests.get(url, params=params)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            info['imdgNm'] = item.findtext('imdgNm', '')
            info['imdgEngNm'] = item.findtext('imdgEngNm', '')
            info['emergManagtCd'] = item.findtext('emergManagtCd', '-')
            info['packngGrad'] = item.findtext('packngGrad', '-')
    except Exception as e:
        print(f"위험물정보 API 에러: {e}")
    return info

def fetch_chem_safety_info(chem_name):
    """
    2단계: 물질명(국문/영문)을 이용해 화학물질안전원 '화학물질 안전관리정보' 조회
    """
    url = "http://apis.data.go.kr/1480802/iciskischem/kischemlist"
    
    # 검색어 정제 (특수문자 제거 등)
    clean_name = chem_name.split('(')[0].strip()
    
    params = {
        'serviceKey': PUBLIC_API_KEY,
        'numOfRows': '3',
        'pageNo': '1',
        'chemKo': clean_name  # 국문 물질명 검색 활용
    }
    
    safety_data = {"symptom": "자료 없음", "inhale": "자료 없음", "skin": "자료 없음", "eyeball": "자료 없음", "oral": "자료 없음"}
    
    try:
        res = requests.get(url, params=params)
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
            safety_data['symptom'] = item.findtext('symptom', '자료 없음')
            safety_data['inhale'] = item.findtext('inhale', '자료 없음')
            safety_data['skin'] = item.findtext('skin', '자료 없음')
            safety_data['eyeball'] = item.findtext('eyeball', '자료 없음')
            safety_data['oral'] = item.findtext('oral', '자료 없음')
    except Exception as e:
        print(f"화학물질 안전관리정보 API 에러: {e}")
        
    return safety_data

# ==========================================
# 2. Gemini API 활용 요약 모듈
# ==========================================
def generate_gemini_summary(chem_name, unno, dgst_info, safety_info):
    """
    두 API에서 가져온 방대한 데이터를 Gemini API를 통해 현장 맞춤형으로 요약
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "여러분의_GEMINI_API_KEY":
        return "⚠️ Gemini API 키가 설정되지 않았습니다. 코드를 확인해주세요."
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        당신은 항만 및 해양 위험물(HNS) 안전관리 전문가입니다.
        아래의 공공 데이터 정보를 바탕으로 현장 대응 소방대원과 관리자를 위한 '긴급 비상대응 가이드'를 마크다운 형식으로 요약해 주세요.

        [기본 정보]
        - 물질명: {chem_name}
        - UN 번호: {unno}
        - IMDG 명칭: {dgst_info.get('imdgNm')} ({dgst_info.get('imdgEngNm')})
        - 비상조치코드(EmS): {dgst_info.get('emergManagtCd')}
        - 포장등급: {dgst_info.get('packngGrad')}

        [안전관리원 상세 증상 및 노출 정보]
        - 일반 증상: {safety_info.get('symptom')}
        - 흡입 시: {safety_info.get('inhale')}
        - 피부 접촉 시: {safety_info.get('skin')}
        - 안구 접촉 시: {safety_info.get('eyeball')}
        - 경구 섭취 시: {safety_info.get('oral')}

        요청 사항:
        1. 핵심 위험성 요약 (인체 영향 중심)
        2. 필수 개인 보호구 및 초동 방제 요령
        3. 절대 금지해야 할 행동 (금기 사항)
        전문적이고 명확하게 요약해 주세요.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash', # 빠르고 효율적인 모델 활용
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini API 호출 중 오류 발생: {e}"

# ==========================================
# 3. RPA 통합 데이터 로드
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
    else:
        return None

# ==========================================
# 4. Streamlit 화면 UI
# ==========================================
st.title("🚨 평택항 HNS 실시간 안전관리 및 지능형 대응 시스템")
st.markdown("**포트미스 RPA 데이터 + 해수부 위험물정보 API + 화학물질안전원 API + Gemini AI 통합 솔루션**")

df = load_integrated_hns_data()

if df is None or df.empty:
    st.warning("⚠️ 수집된 통합 위험물 데이터(`hns_fully_integrated_report.csv`)가 없습니다. RPA 봇을 먼저 실행해 주세요!")
else:
    st.sidebar.header("🔍 검색 및 필터")
    selected_ship = st.sidebar.selectbox("선박 선택", ["전체보기"] + list(df['선박명(선택)'].unique()))
    
    if selected_ship != "전체보기":
        filtered_df = df[df['선박명(선택)'] == selected_ship]
    else:
        filtered_df = df

    st.subheader(f"📊 평택항 위험물 반입 현황 (총 {len(filtered_df)}건)")
    st.dataframe(filtered_df[['선박명(선택)', '호출부호', '하역업체', '하역기간', '사용장소', 'UNNO', '품명', '중량', '단위', '수하인', '송하인']], use_container_width=True)

    st.divider()
    st.subheader("🚢 선박별 상세 운송 정보 및 지능형 비상 대응")

    unique_ships = filtered_df['선박명(선택)'].unique()
    
    for ship in unique_ships:
        ship_data = filtered_df[filtered_df['선박명(선택)'] == ship]
        call_sign = ship_data['호출부호'].iloc[0]
        location = ship_data['사용장소'].iloc[0] if pd.notna(ship_data['사용장소'].iloc[0]) else "지정 장소 미상"
        work_period = ship_data['하역기간'].iloc[0]
        
        with st.expander(f"⚓ [{ship}] (호출부호: {call_sign}) ｜ 하역장소: {location} ｜ 기간: {work_period}"):
            st.markdown(f"**🏢 하역업체:** {ship_data['하역업체'].iloc[0]} ｜ **반입구분:** {ship_data['반입구분'].iloc[0]}")
            st.markdown("---")
            st.markdown("#### 📦 적재된 위험물 목록")
            
            for idx, row in ship_data.iterrows():
                unno = str(row['UNNO']).zfill(4) # 4자리 UN NO 포맷 맞춤 (예: 0005)
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

    # 선택된 물질의 API 조회 및 Gemini AI 요약 출력 영역
    if 'active_chem' in st.session_state:
        st.divider()
        chem = st.session_state['active_chem']
        unno = st.session_state['active_unno']
        
        st.error(f"⚡ [지능형 비상대응 가이드] 대상 선박: {st.session_state['active_ship']} ｜ 물질: {chem} (UN NO: {unno})")
        
        with st.spinner('해양수산부 위험물정보 및 화학물질안전원 데이터를 실시간 수집하고 Gemini AI가 분석 중입니다...'):
            # 1. 해수부 위험물정보 API 호출
            dgst_info = fetch_dgst_info(unno)
            # 2. 화학물질안전원 안전관리정보 API 호출
            safety_info = fetch_chem_safety_info(chem)
            # 3. Gemini AI 요약 생성
            ai_summary = generate_gemini_summary(chem, unno, dgst_info, safety_info)
            
        st.markdown(ai_summary)
        
        st.markdown("#### 📋 현장 초동 조치 체크리스트")
        st.checkbox("1. 풍상측(바람을 등지는 방향)에서 접근 및 대피로를 확보하였는가?", key="chk1")
        st.checkbox("2. 사고 반경 내 통제선(Hot/Warm Zone)을 즉시 설정하였는가?", key="chk2")
        st.checkbox("3. 해경, 소방서 및 화학방재센터에 즉시 상황을 전파하였는가?", key="chk3")
        
        if st.button("🔄 가이드 닫기"):
            del st.session_state['active_chem']
            del st.session_state['active_unno']
            del st.session_state['active_ship']
            st.rerun()