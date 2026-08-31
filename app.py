import base64
from datetime import datetime, timedelta, timezone
import json
import os
import re
import urllib3
import xml.etree.ElementTree as ET
import folium
from google import genai
# RAG Vector DB 연동 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
# PDF 및 이미지 처리 라이브러리
import pdfplumber
from PIL import Image
import requests
import streamlit as st
import streamlit.components.v1 as components
import websocket

# SSL 경고창 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. 로컬 이미지 및 PDF 파일 경로 설정
# ==========================================
KCG_LOGO_PATH = 'kcg_logo.png'
HNS_PDF_PATH = '해상운송 위험유해물질 정보집(HNS 정보집)2024.pdf'
KCG_GUIDE_PDF_PATH = '위험유해물질(HNS) 해양사고 대응 가이드.pdf'


@st.cache_data
def get_base64_logo(image_path):
  if not os.path.exists(image_path):
    return None
  try:
    with open(image_path, 'rb') as f:
      data = f.read()
    return base64.b64encode(data).decode()
  except Exception:
    return None


kcg_logo_b64 = get_base64_logo(KCG_LOGO_PATH)

# 페이지 설정
st.set_page_config(
    page_title='평택해양경찰서 HNS AI 대응 시스템',
    page_icon=KCG_LOGO_PATH if os.path.exists(KCG_LOGO_PATH) else '🚢',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# Light UI + 모바일 가시성 CSS
st.markdown(
    """
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
</style>
""",
    unsafe_allow_html=True,
)

PUBLIC_API_KEY = st.secrets.get('PUBLIC_API_KEY', '')
GEMINI_API_KEY = st.secrets.get('GEMINI_API_KEY', '')
AISSTREAM_API_KEY = st.secrets.get('AISSTREAM_API_KEY', '')

# ==========================================
# 1. 🖼️ PDF 인덱스 맵 생성 (JSON 파일 오프라인 저장으로 새로고침 시 0.01초 초고속 로딩)
# ==========================================

HNS_INDEX_JSON_PATH = 'hns_pdf_index.json'


@st.cache_data(show_spinner=False)
def get_hns_pdf_index(pdf_path):
  if os.path.exists(HNS_INDEX_JSON_PATH):
    try:
      with open(HNS_INDEX_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)
    except Exception as e:
      print(f'JSON 인덱스 로드 실패, PDF 재스캔 진행: {e}')

  if not os.path.exists(pdf_path):
    return []

  index_list = []
  try:
    with pdfplumber.open(pdf_path) as pdf:
      start_idx = 38
      end_idx = min(223, len(pdf.pages))

      for idx in range(start_idx, end_idx):
        page = pdf.pages[idx]
        text = page.extract_text() or ''

        if not text.strip():
          continue

        unno_match = (
            re.search(r'UN\s*번호\s*[:\s]*(\d{4})', text, re.IGNORECASE)
            or re.search(r'UN\s*(\d{4})', text, re.IGNORECASE)
            or re.search(r'(\d{4})', text)
        )
        unno = unno_match.group(1).strip() if unno_match else ''

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        title = lines[0] if lines else ''

        synonym_match = re.search(r'유사명\s*[:\s]*([^\n]+)', text)
        synonyms = synonym_match.group(1).strip() if synonym_match else ''

        index_list.append({
            'page_index': idx,
            'display_page_no': idx + 1,
            'unno': unno,
            'title': title,
            'synonyms': synonyms,
            'raw_text': text.upper(),
        })

    with open(HNS_INDEX_JSON_PATH, 'w', encoding='utf-8') as f:
      json.dump(index_list, f, ensure_ascii=False, indent=2)

  except Exception as e:
    print(f'PDF 인덱스 구축 중 에러: {e}')

  return index_list


hns_pdf_index = get_hns_pdf_index(HNS_PDF_PATH)
st.session_state['hns_pdf_index'] = hns_pdf_index


def get_hns_page_image(unno_or_query, cas_no='-'):
  idx_list = st.session_state.get('hns_pdf_index', [])
  if not idx_list or not os.path.exists(HNS_PDF_PATH):
    return None, None

  q = str(unno_or_query).strip().upper()
  q_cas = str(cas_no).strip()
  target_item = None

  if not target_item and q_cas and q_cas not in ['-', '0000', '없음']:
    for item in idx_list:
      if q_cas in item.get('raw_text', ''):
        target_item = item
        break

  if not target_item and q.isdigit() and len(q) == 4:
    for item in idx_list:
      if item['unno'] == q:
        target_item = item
        break

  if not target_item and q:
    for item in idx_list:
      if q in item['title'].upper() or q in item['synonyms'].upper():
        target_item = item
        break

  if not target_item and len(q) >= 2:
    for item in idx_list:
      if q in item['raw_text']:
        target_item = item
        break

  if target_item is None:
    return None, None

  try:
    with pdfplumber.open(HNS_PDF_PATH) as pdf:
      page = pdf.pages[target_item['page_index']]
      pix = page.to_image(resolution=300)
      return pix.original, target_item['display_page_no']
  except Exception as e:
    print(f'HNS 정보집 PDF 이미지 렌더링 에러: {e}')
    return None, None


# 🧠 RAG Vector DB 로드 모듈
@st.cache_resource
def load_kcg_vectorstore():
  persist_dir = './kcg_guide_chromadb'
  if os.path.exists(persist_dir):
    try:
      embeddings = HuggingFaceEmbeddings(
          model_name='jhgan/ko-sroberta-multitask',
          model_kwargs={'device': 'cpu'},
          encode_kwargs={'normalize_embeddings': True},
      )
      return Chroma(
          persist_directory=persist_dir, embedding_function=embeddings
      )
    except Exception as e:
      print(f'RAG Vector DB 로드 실패: {e}')
  return None


kcg_vectorstore = load_kcg_vectorstore()


# 💡 가이드 참고 페이지를 5페이지로 최적화 (시간 단축)
def fetch_rag_context_and_images(query, k=5):
  if not kcg_vectorstore or not query:
    return 'RAG 가이드 데이터베이스 미생성', []

  try:
    docs = kcg_vectorstore.similarity_search(query, k=k)
    if not docs:
      return '관련 가이드 지침 검색 결과 없음', []

    context_items = []
    page_numbers = []

    for doc in docs:
      page_no = doc.metadata.get('page', 0) + 1
      if page_no not in page_numbers:
        page_numbers.append(page_no)
      context_items.append(f'[대응가이드 {page_no}쪽 지침]\n{doc.page_content}')

    rag_images = []
    if os.path.exists(KCG_GUIDE_PDF_PATH):
      with pdfplumber.open(KCG_GUIDE_PDF_PATH) as pdf:
        for p_no in page_numbers:
          if 0 <= p_no - 1 < len(pdf.pages):
            page = pdf.pages[p_no - 1]
            pix = page.to_image(resolution=200)
            rag_images.append({'page_no': p_no, 'pil_img': pix.original})

    return '\n\n'.join(context_items), rag_images
  except Exception as e:
    return f'RAG 검색 오류: {e}', []


# ==========================================
# 2. 공공 API 연동 모듈
# ==========================================


@st.cache_data(ttl=300)
def fetch_vessel_schedule_api(port_code, de_gb, sde_str, ede_str):
  if not PUBLIC_API_KEY:
    return []
  url = f'https://apis.data.go.kr/1192000/VsslEtrynd5/Info5?serviceKey={PUBLIC_API_KEY}'
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
  vessels, page, rows_per_page = [], 1, 50

  while True:
    params = {
        'prtAgCd': str(port_code).strip(),
        'sde': sde_str,
        'ede': ede_str,
        'deGb': str(de_gb).strip().upper(),
        'numOfRows': str(rows_per_page),
        'pageNo': str(page),
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
          prt_ag_nm = (item.findtext('prtAgNm') or '-').strip()
          etrypt_year = (item.findtext('etryptYear') or '-').strip()
          etrypt_co = (item.findtext('etryptCo') or '-').strip()
          clsgn = (item.findtext('clsgn') or '-').strip()
          vssl_nm = (item.findtext('vsslNm') or '-').strip()
          vssl_nlty_nm = (item.findtext('vsslNltyNm') or '-').strip()
          vssl_knd_nm = (item.findtext('vsslKndNm') or '-').strip()
          etrypt_purps_nm = (item.findtext('etryptPurpsNm') or '-').strip()
          prvs_dpmprt_prt_nm = (
              item.findtext('prvsDpmprtPrtNm') or '-'
          ).strip()
          nxlnpt_prt_nm = (item.findtext('nxlnptPrtNm') or '-').strip()
          dstn_prt_nm = (item.findtext('dstnPrtNm') or '-').strip()

          detail_node = item.find('.//detail') or item.find('details/detail')
          reqst_se_nm = etrynd_nm = etrypt_dt = tkoff_dt = ibobprt_nm = (
              laidup_fclty_nm
          ) = ldadng_frght_cl_cd = ldadng_ton = trnpdt_ton = landng_frght_ton = (
              ld_frght_ton
          ) = grtg = satmnt_entrps_nm = crew_co = tkoff_prrrn_dt = (
              dstn_etrypt_dt
          ) = '-'

          if detail_node is not None:
            reqst_se_nm = (detail_node.findtext('reqstSeNm') or '-').strip()
            etrynd_nm = (detail_node.findtext('etryndNm') or '-').strip()
            etrypt_dt = (detail_node.findtext('etryptDt') or '-').strip()
            tkoff_dt = (detail_node.findtext('tkoffDt') or '-').strip()
            ibobprt_nm = (detail_node.findtext('ibobprtNm') or '-').strip()
            laidup_fclty_nm = (
                detail_node.findtext('laidupFcltyNm') or '-'
            ).strip()
            ldadng_frght_cl_cd = (
                detail_node.findtext('ldadngFrghtClCd') or '-'
            ).strip()
            ldadng_ton = (detail_node.findtext('ldadngTon') or '-').strip()
            trnpdt_ton = (detail_node.findtext('trnpdtTon') or '-').strip()
            landng_frght_ton = (
                detail_node.findtext('landngFrghtTon') or '-'
            ).strip()
            ld_frght_ton = (detail_node.findtext('ldFrghtTon') or '-').strip()
            grtg = (detail_node.findtext('grtg') or '-').strip()
            satmnt_entrps_nm = (
                detail_node.findtext('satmntEntrpsNm') or '-'
            ).strip()
            crew_co = (detail_node.findtext('crewCo') or '-').strip()
            tkoff_prrrn_dt = (
                detail_node.findtext('tkoffPrrrnDt') or '-'
            ).strip()
            dstn_etrypt_dt = (
                detail_node.findtext('dstnEtryptDt') or '-'
            ).strip()

          if vssl_nm == '-' and clsgn == '-':
            continue

          vessels.append({
              'prt_ag_nm': prt_ag_nm,
              'etrypt_year': etrypt_year,
              'etrypt_co': etrypt_co,
              'clsgn': clsgn,
              'vssl_nm': vssl_nm,
              'vssl_nlty_nm': vssl_nlty_nm,
              'vssl_knd_nm': vssl_knd_nm,
              'etrypt_purps_nm': etrypt_purps_nm,
              'prvs_dpmprt_prt_nm': prvs_dpmprt_prt_nm,
              'nxlnpt_prt_nm': nxlnpt_prt_nm,
              'dstn_prt_nm': dstn_prt_nm,
              'etrynd_nm': etrynd_nm,
              'ibobprt_nm': ibobprt_nm,
              'etrypt_dt': (
                  etrypt_dt.replace('T', ' ') if etrypt_dt != '-' else '-'
              ),
              'tkoff_dt': tkoff_dt.replace('T', ' ') if tkoff_dt != '-' else '-',
              'laidup_fclty_nm': laidup_fclty_nm,
              'ldadng_frght_cl_cd': ldadng_frght_cl_cd,
              'ldadng_ton': ldadng_ton,
              'trnpdt_ton': trnpdt_ton,
              'landng_frght_ton': landng_frght_ton,
              'ld_frght_ton': ld_frght_ton,
              'grtg': grtg,
              'satmnt_entrps_nm': satmnt_entrps_nm,
              'crew_co': crew_co,
              'tkoff_prrrn_dt': tkoff_prrrn_dt,
              'dstn_etrypt_dt': dstn_etrypt_dt,
              'reqst_se_nm': reqst_se_nm,
          })

        if len(vessels) >= total_cnt or len(items) < rows_per_page:
          break
        page += 1
      else:
        break
    except Exception as e:
      print(f'선박 API 예외 ({port_code}/{de_gb}): {e}')
      break

  return vessels


def fetch_dgst_info(unno):
  if not unno or unno in ['0000', '-', '']:
    return {
        'imdgNm': '',
        'imdgEngNm': '',
        'kndNm': '-',
        'kndPrdlstNm': '-',
        'imdgGradCd': '-',
        'emergManagtCd': '-',
        'ldadngMth': '-',
        'catinMatter': '-',
    }

  clean_unno = str(unno).strip().zfill(4)
  url = f'https://apis.data.go.kr/1192000/DgstInqire3/Info?serviceKey={PUBLIC_API_KEY}'
  params = {'unno': clean_unno, 'numOfRows': '1', 'pageNo': '1'}
  info = {
      'imdgNm': '',
      'imdgEngNm': '',
      'kndNm': '-',
      'kndPrdlstNm': '-',
      'imdgGradCd': '-',
      'emergManagtCd': '-',
      'ldadngMth': '-',
      'catinMatter': '-',
  }
  try:
    session = requests.Session()
    session.verify = False
    res = session.get(url, params=params, timeout=8)
    root = ET.fromstring(res.content)
    item = root.find('.//item')
    if item is not None:
      info['imdgNm'] = item.findtext('imdgNm') or ''
      info['imdgEngNm'] = item.findtext('imdgEngNm') or ''
      info['kndNm'] = item.findtext('kndNm') or '-'
      info['kndPrdlstNm'] = item.findtext('kndPrdlstNm') or '-'
      info['imdgGradCd'] = item.findtext('imdgGradCd') or '-'
      info['emergManagtCd'] = item.findtext('emergManagtCd') or '-'
      info['ldadngMth'] = item.findtext('ldadngMth') or '-'
      info['catinMatter'] = item.findtext('catinMatter') or '-'
  except Exception as e:
    print(f'위험물정보 API 에러: {e}')
  return info


def fetch_chem_safety_info(cas_no):
  if not cas_no or cas_no in ['-', '0000', '없음', '']:
    return {
        'symptom': '자료없음',
        'inhale': '자료없음',
        'skin': '자료없음',
        'eyeball': '자료없음',
        'oral': '자료없음',
        'etc': '자료없음',
    }

  clean_cas = str(cas_no).strip()
  url = f'https://apis.data.go.kr/1480802/iciskischem/kischemlist?serviceKey={PUBLIC_API_KEY}'
  params = {'numOfRows': '3', 'pageNo': '1', 'casNo': clean_cas}
  safety_data = {
      'symptom': '자료없음',
      'inhale': '자료없음',
      'skin': '자료없음',
      'eyeball': '자료없음',
      'oral': '자료없음',
      'etc': '자료없음',
  }
  try:
    session = requests.Session()
    session.verify = False
    res = session.get(url, params=params, timeout=8)
    root = ET.fromstring(res.content)
    item = root.find('.//item')
    if item is not None:
      safety_data['symptom'] = item.findtext('symptom') or '자료없음'
      safety_data['inhale'] = item.findtext('inhale') or '자료없음'
      safety_data['skin'] = item.findtext('skin') or '자료없음'
      safety_data['eyeball'] = item.findtext('eyeball') or '자료없음'
      safety_data['oral'] = item.findtext('oral') or '자료없음'
      safety_data['etc'] = item.findtext('etc') or '자료없음'
  except Exception as e:
    print(f'화학물질 안전관리정보 API 에러: {e}')
  return safety_data


def fetch_kosha_msds_info(chem_name, cas_no, unno):
  base_url = 'https://msds.kosha.or.kr/openapi/service/msdschem'
  chem_id = None
  search_trials = [(cas_no, '1'), (unno, '2'), (chem_name, '0')]

  for search_wrd, search_cnd in search_trials:
    if not search_wrd or str(search_wrd).strip() in [
        '-',
        '0000',
        '0',
        '없음',
        '',
    ]:
      continue
    clean_wrd = str(search_wrd).strip()
    list_url = f'{base_url}/getChemList'
    params = {
        'serviceKey': PUBLIC_API_KEY,
        'searchWrd': clean_wrd,
        'searchCnd': search_cnd,
        'numOfRows': '5',
        'pageNo': '1',
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
        if search_cnd == '2':
          if (item.findtext('unno') or '').strip() == clean_wrd.zfill(4):
            matched_id = found_id.strip()
            break
        elif search_cnd == '1':
          if (item.findtext('casNo') or '').strip() == clean_wrd:
            matched_id = found_id.strip()
            break
        elif search_cnd == '0':
          item_ko = (item.findtext('chemKo') or '').strip().replace('·', '')
          item_en = (
              (item.findtext('chemEn') or '').strip().replace('·', '').lower()
          )
          target_wrd = clean_wrd.replace('·', '').lower()
          if target_wrd == item_ko.lower() or target_wrd == item_en:
            matched_id = found_id.strip()
            break
      if matched_id:
        chem_id = matched_id
        break
    except Exception as e:
      print(f'KOSHA 에러: {e}')

  if not chem_id:
    return '안전보건공단 MSDS 연동 데이터 없음 (chemId 미발급)'

  msds_details = []
  for i in range(1, 17):
    detail_url = f'{base_url}/getChemDetail{i:02d}'
    params = {'serviceKey': PUBLIC_API_KEY, 'chemId': chem_id}
    try:
      res = requests.get(detail_url, params=params, timeout=4)
      root = ET.fromstring(res.content)
      for item in root.findall('.//item'):
        name_kor = (item.findtext('msdsItemNameKor') or '').strip()
        detail_val = (item.findtext('itemDetail') or '').strip()
        if detail_val and detail_val != '자료없음':
          msds_details.append(f'[{name_kor}] {detail_val}')
    except Exception:
      continue

  if not msds_details:
    return f'안전보건공단 MSDS 기본 정보 등록 (chemId: {chem_id})'
  return f'[KOSHA MSDS chemId: {chem_id}]\n' + '\n'.join(msds_details[:30])


# 💡 가이드 문서 명세(vsslKorNm, vsslEngNm)를 100% 반영한 다중 선박 API 함수[cite: 6]
@st.cache_data(ttl=300)
def fetch_vessel_spec_list_api(query_str, max_results=50):
  if not PUBLIC_API_KEY or not query_str:
    return []

  url = f'https://apis.data.go.kr/1192000/SicsVsslManp3/Info3?serviceKey={PUBLIC_API_KEY}'
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
  clean_q = str(query_str).strip()

  results = []
  trials = [('clsgn', clean_q), ('vsslNm', clean_q)]

  for param_key, param_val in trials:
    page = 1
    while len(results) < max_results:
      params = {
          param_key: param_val,
          'numOfRows': '50',
          'pageNo': str(page),
      }
      try:
        session = requests.Session()
        session.verify = False
        res = session.get(url, params=params, headers=headers, timeout=8)
        if res.status_code == 200 and res.content:
          root = ET.fromstring(res.content)
          items = root.findall('.//item') or root.findall('body/items/item')
          if not items:
            break

          for item in items:
            # 💡 가이드 문서 명세 표준 태그 파싱[cite: 6]
            kor_name = (item.findtext('vsslKorNm') or '').strip() or '-'
            eng_name = (item.findtext('vsslEngNm') or '').strip() or '-'

            # 영문/한글 병기 표시명 구성
            if eng_name != '-' and kor_name != '-':
              if eng_name.upper() == kor_name.upper():
                display_name = eng_name
              else:
                display_name = f'{eng_name} ({kor_name})'
            elif eng_name != '-':
              display_name = eng_name
            elif kor_name != '-':
              display_name = kor_name
            else:
              display_name = '-'

            spec = {
                'vsslNo': (item.findtext('vsslNo') or '').strip() or '-',
                'imoNo': (item.findtext('imoNo') or '').strip() or '-',
                'mmsiNo': (item.findtext('mmsiNo') or '').strip() or '-',
                'clsgn': (item.findtext('clsgn') or '').strip() or '-',
                'vsslKorNm': kor_name,
                'vsslEngNm': eng_name,
                'displayName': display_name,
                'vsslKnd': (item.findtext('vsslKnd') or '').strip() or '-',
                'vsslNlty': (item.findtext('vsslNlty') or '').strip() or '-',
                'grtg': (item.findtext('grtg') or '').strip() or '-',
                'vsslTotLt': (item.findtext('vsslTotLt') or '').strip() or '-',
                'shdth': (item.findtext('shdth') or '').strip() or '-',
                'vsslDrft': (item.findtext('vsslDrft') or '').strip() or '-',
                'vsslDp': (item.findtext('vsslDp') or '').strip() or '-',
                'brbtSeNm': (item.findtext('brbtSeNm') or '').strip() or '-',
                'nvgShapNm': (item.findtext('nvgShapNm') or '').strip() or '-',
                'vsslCnstrDt': (
                    (item.findtext('vsslCnstrDt') or '-')
                    .strip()
                    .replace('T', ' ')
                ),
            }
            if not any(
                r['clsgn'] == spec['clsgn']
                and r['vsslKorNm'] == spec['vsslKorNm']
                for r in results
            ):
              results.append(spec)

          if len(items) < 50:
            break
          page += 1
        else:
          break
      except Exception as e:
        print(f'선박제원 다중목록 조회 에러: {e}')
        break

    if results:
      break

  return results


def fetch_vessel_spec_api(clsgn, vssl_nm):
  specs = fetch_vessel_spec_list_api(clsgn) or fetch_vessel_spec_list_api(
      vssl_nm
  )
  return specs[0] if specs else None


# ==========================================
# 3. 🧠 Gemini Vision 멀티모달 프롬프트 연동 (우선순위: 3.7 -> 3.6 -> 3.5)
# ==========================================


@st.cache_data(ttl=3600)
def map_search_query_with_gemini(query_text):
  default_res = {
      'chem_ko': query_text,
      'chem_eng': query_text,
      'unno': '0000',
      'cas_no': '-',
      'accident_context': '',
  }
  if not GEMINI_API_KEY or not query_text:
    return default_res

  try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
        사용자 입력 텍스트: "{query_text}"
        이 문장 또는 검색어에서:
        1. 포함되어 있거나 가장 대표적인 위험물/HNS 화학물질의 표준 정보(한글명, 영문명, UN번호 4자리, CAS번호)를 분석하세요.
        2. 만약 특정 사고 상황이 언급되어 있다면 "accident_context"에 요약하세요. (단순 물질명일 경우 빈값)

        JSON 포맷으로만 답변하세요:
        {{
            "chem_ko": "공식 한글명", 
            "chem_eng": "공식 영문명", 
            "unno": "4자리 UN번호", 
            "cas_no": "CAS번호",
            "accident_context": "사고 상황 요약 문장 또는 빈값"
        }}
        """
    # 💡 우선순위 변경: 3.7 -> 3.6 -> 3.5
    for model_id in [
        'gemini-3.7-flash',
        'gemini-3.6-flash',
        'gemini-3.5-flash-lite',
    ]:
      try:
        response = client.models.generate_content(
            model=model_id, contents=prompt
        )
        text = response.text.replace('```json', '').replace('```', '').strip()
        res_json = json.loads(text)
        if 'accident_context' not in res_json:
          res_json['accident_context'] = ''
        return res_json
      except Exception:
        continue
    return default_res
  except Exception:
    return default_res


def generate_gemini_vision_summary(
    chem_name,
    unno,
    cas_no,
    dgst_info,
    safety_info,
    kosha_msds_text,
    rag_text,
    hns_pil_image=None,
    hns_page_no=None,
    rag_images=[],
    accident_context='',
):
  if not GEMINI_API_KEY:
    return '⚠️ Gemini API 키가 설정되지 않았습니다.'

  try:
    client = genai.Client(api_key=GEMINI_API_KEY)

    accident_info = (
        f'\n🚨 [현장 사고 상황 조건]: {accident_context}\n'
        if accident_context
        else ''
    )

    hns_img_prompt = (
        f'\n- [첨부 이미지 1]: 해경 HNS 정보집 {hns_page_no}쪽 원본 스캔'
        ' 이미지입니다. 표, 세로 쓰기, NFPA 픽토그램 수치 등을 직접'
        ' 판독하여 반영하세요.\n'
        if (hns_pil_image and hns_page_no)
        else ''
    )

    rag_img_prompt = ''
    if rag_images:
      rag_pages_str = ', '.join([f"{img['page_no']}쪽" for img in rag_images])
      rag_img_prompt = (
          '\n- [첨부 이미지 추가]: 해경 HNS 해양사고 대응 가이드'
          f' ({rag_pages_str}) 원본 스캔 이미지들입니다.\n'
      )

    prompt_text = f"""
        당신은 해양경찰청 및 항만 HNS 비상대응 상황실의 최고 수석 관제관입니다.
        첨부된 [HNS 정보집 원본 스캔 이미지]와 [해양사고 대응 가이드 원본 스캔 이미지들], 그리고 수집된 다중 데이터(공공 API, MSDS, RAG 텍스트)를 철저히 교차 분석하여, 현장 세력(OSC, 경비함정, 특수구조대 등)에 즉각 하달할 수 있는 가장 전문적이고 완벽한 **비상대응 지시서**를 작성하세요.

        {accident_info}
        {hns_img_prompt}
        {rag_img_prompt}

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

        [선박오염방지규칙 제3조 유해액체물질 분류 기준 및 해상 거동 원칙]
        - X류 물질: 해양 배출 시 심각한 위해 초래, 해양 배출 전면 금지 및 긴급 방제 최우선
        - Y류 물질: 해양 배출 시 위해 발생, 해양 배출 엄격 제한
        - Z류 물질: 해양 배출 시 경미한 위해, 해양 배출 일부 제한
        - 기타물질 / 잠정평가물질: 위해가 없거나 잠정 평가된 물질
        (※ 첨부된 정보집 이미지상의 운송방법/유해액체물질 분류(X,Y,Z류) 및 해상 거동 특성을 식별하여 방제 조치에 반드시 반영하세요.)

        [엄격 작성 규칙]
        1. **데이터 출처별 역할 분담 (최우선 원칙)**:
           - **화학물질의 특성, 성상, 유해성, 증상, MSDS 수치**: 해양수산부 위험물정보, 화학물질안전원, 안전보건공단 API를 통해 수집된 데이터를 최우선으로 반영하여 정확한 물질 정보를 기재하세요.
           - **현장 대응, 초동 이격 거리, 대피 거리, 개인 보호구, 방제 및 소화 요령**: 해경 HNS 정보집 원본 이미지 및 해경 HNS 대응가이드(RAG) 이미지에 기재된 지침을 최우선으로 반영하여 현장 실행 위주로 작성하세요. (근본적인 물질 스펙은 전문 기관 API를 따르고, 실전 대응 전략은 해경 전문 가이드를 따를 것)
        2. **초동대응 핵심요약 작성**: 각 항목의 시작은 `* **항목명**:` 포맷을 사용하고, 현장 세력이 즉시 이해할 수 있는 명확한 개조식 문장으로 작성하세요.
        3. **예외 및 안전 보완 기준**: 
           - 물질명 미확인 시 기본 유출 100m / 화재 800m 이격 조치를 지정하세요.
           - 물 반응성 물질 확인 시 직사주수 절대 금지 및 분무(안개) 주수 지침을 명시하세요.
        4. **출력 형식 엄격 준수**: 공문서 서식(수신, 발신 등)을 절대 생성하지 말고, 반드시 `### 🚨 [초동대응 핵심요약]` 제목부터 곧바로 출력을 시작하세요.

        ### 🚨 [초동대응 핵심요약]

        * **사고물질 및 위험성 판단**: [IMDG 등급, 유해액체물질 분류(X/Y/Z류) 및 핵심위험성(인화성/독성/수반응성 등) 전파 및 위험성 평가 지시]
        * **통제 및 이격거리 지시**: [초기이격, 화재대피, 유출방호 M단위 수치 명시 및 해역/현장 통제 조치 지시]
        * **출동세력 보호구 지정**: [필수 레벨(Level A/B/C/D) 및 필수 장비(공기호흡기, 내화학복, 가스탐지기 등) 착용 지시]
        * **현장 초동 행동 수칙**: [풍상위치 확보, 사고유형 맞춤 행동, 직수금지/소화약제 및 사고선 비상조치 확인 지시]

        ---
        ### 1. ⚠️ 물리·화학적 성상 및 주요 위험성
        ### 2. 🛡️ 개인 보호구 및 초동 방제/소화 요령
        ### 3. ⛔ 금지 행동 (금기 사항)
        ### 4. 🏥 인체 노출 시 영향 및 긴급 응급조치
        """

    contents_input = [prompt_text]
    if hns_pil_image:
      contents_input.append(hns_pil_image)

    for r_img in rag_images:
      contents_input.append(r_img['pil_img'])

    # 💡 우선순위 변경: 3.7 -> 3.6 -> 3.5
    for model_id in [
        'gemini-3.7-flash',
        'gemini-3.6-flash',
        'gemini-3.5-flash-lite',
    ]:
      try:
        response = client.models.generate_content(
            model=model_id, contents=contents_input
        )
        return response.text
      except Exception as ex:
        print(f'{model_id} Vision 처리 시도 실패: {ex}')
        continue

    return '⚠️ Gemini API 호출에 실패했습니다.'
  except Exception as e:
    return f'Gemini API 클라이언트 생성 오류: {e}'


# ==========================================
# ⚓ AISStream WebSocket 실시간 위치 수신 (개선형)
# ==========================================


def fetch_aisstream_vessel_position(
    vssl_nm='', clsgn='', imo_no='', timeout_sec=4
):
  if not AISSTREAM_API_KEY:
    return None

  subscribe_message = {
      'APIKey': AISSTREAM_API_KEY,
      'BoundingBoxes': [[[34.0, 124.0], [38.5, 128.5]]],
  }

  position_data = None
  target_nm = str(vssl_nm).strip().upper()
  target_clsgn = str(clsgn).strip().upper()

  def on_open(ws):
    ws.send(json.dumps(subscribe_message))

  def on_message(ws, message):
    nonlocal position_data
    try:
      data = json.loads(message)
      msg_type = data.get('MessageType')
      metadata = data.get('MetaData', {})

      if msg_type == 'PositionReport':
        recv_ship_name = str(metadata.get('ShipName', '')).strip().upper()
        recv_clsgn = str(metadata.get('CallSign', '')).strip().upper()
        pos = data.get('Message', {}).get('PositionReport', {})

        match_found = False
        if target_clsgn and target_clsgn != '-' and target_clsgn == recv_clsgn:
          match_found = True
        elif (
            target_nm
            and target_nm != '-'
            and (target_nm in recv_ship_name or recv_ship_name in target_nm)
        ):
          match_found = True

        if match_found:
          position_data = {
              'lat': pos.get('Latitude'),
              'lon': pos.get('Longitude'),
              'sog': pos.get('Sog', 0.0),
              'cog': pos.get('Cog', 0.0),
              'time_utc': metadata.get('time_utc', ''),
              'ship_name': metadata.get('ShipName', '-'),
              'mmsi': metadata.get('MMSI', '-'),
          }
          ws.close()
    except Exception as e:
      print(f'AISStream 파싱 예외: {e}')

  def on_error(ws, error):
    print(f'AISStream 에러: {error}')

  try:
    ws = websocket.WebSocketApp(
        'wss://stream.aisstream.io/v0/stream',
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )
    ws.run_forever(ping_timeout=timeout_sec)
  except Exception as e:
    print(f'AISStream 연결 실패: {e}')

  return position_data


# ==========================================
# 🚢 모달 팝업: 지도 이동 시 재로딩 완벽 방지 (HTML 컴포넌트 방식)
# ==========================================


@st.dialog('🚢 선박 제원 및 실시간 AIS 위치 정보', width='large')
def show_vessel_detail_dialog(v):
  st.subheader(f"⚓ {v['vssl_nm']} (`호출부호: {v['clsgn']}`)")
  st.divider()

  col_left, col_right = st.columns([1, 1])

  with col_left:
    st.markdown('#### 📐 선박 제원 스펙 정보')
    spec_info = fetch_vessel_spec_api(v['clsgn'], v['vssl_nm'])

    if spec_info:
      kor_nm = spec_info.get('vsslKorNm', '-')
      eng_nm = spec_info.get('vsslEngNm', '-')

      # 💡 선박명(한/영) 포맷 통일: 둘 다 있거나 한쪽만 있을 때 정확히 표출[cite: 6]
      if kor_nm != '-' and eng_nm != '-':
        name_str = (
            f'{kor_nm} / {eng_nm}'
            if kor_nm.upper() != eng_nm.upper()
            else f'{kor_nm} / {eng_nm}'
        )
      elif kor_nm != '-':
        name_str = f'{kor_nm} / -'
      elif eng_nm != '-':
        name_str = f'- / {eng_nm}'
      else:
        name_str = f"{v.get('vssl_nm', '-')} / -"

      st.write(f'- **선박명(한/영):** {name_str}')
      st.write(
          f"- **선박번호 / IMO:** `{spec_info['vsslNo']}` /"
          f' `{spec_info["imoNo"]}`'
      )
      if spec_info.get('mmsiNo') and spec_info.get('mmsiNo') != '-':
        st.write(f"- **MMSI 번호:** `{spec_info['mmsiNo']}`")
      st.write(
          f"- **선종 / 국적:** {spec_info['vsslKnd']} / {spec_info['vsslNlty']}"
      )
      st.write(f"- **총톤수(GRT):** {spec_info['grtg']} 톤")
      st.write(
          f"- **선박 길이×너비:** {spec_info['vsslTotLt']}m ×"
          f' {spec_info["shdth"]}m'
      )
      st.write(
          f"- **흘수 / 깊이:** {spec_info['vsslDrft']}m / {spec_info['vsslDp']}m"
      )
      st.write(
          f"- **운항형태 / 나용선:** {spec_info['nvgShapNm']} /"
          f' {spec_info["brbtSeNm"]}'
      )
      st.write(f"- **건조일시:** {spec_info['vsslCnstrDt']}")
    else:
      st.warning('💡 해수부 API에 등록된 선박제원 스펙이 없습니다.')

  with col_right:
    st.markdown('#### 🛰️ 실시간 AIS 위치 및 지도')
    imo_number = spec_info.get('imoNo', '-') if spec_info else '-'

    with st.spinner('AISStream 신호 탐색 중...'):
      ais_pos = fetch_aisstream_vessel_position(
          vssl_nm=v['vssl_nm'],
          clsgn=v['clsgn'],
          imo_no=imo_number,
          timeout_sec=3,
      )

    if ais_pos and ais_pos.get('lat') and ais_pos.get('lon'):
      lat, lon = ais_pos['lat'], ais_pos['lon']
      sog, cog = ais_pos['sog'], ais_pos['cog']
      time_utc = ais_pos['time_utc']

      st.success(
          f'📍 **위치 수신 성공** (위도: `{lat:.4f}`, 경도: `{lon:.4f}`)'
      )
      st.write(f'- **속력(SOG):** {sog} kts ｜ **침로(COG):** {cog}°')
      st.write(f'- **수신시각(UTC):** {time_utc}')

      m = folium.Map(location=[lat, lon], zoom_start=13)
      folium.Marker(
          [lat, lon],
          popup=f"{v['vssl_nm']} ({sog}kts)",
          tooltip=f"{v['vssl_nm']}",
          icon=folium.Icon(color='red', icon='ship', prefix='fa'),
      ).add_to(m)

      map_html = m._repr_html_()
      components.html(map_html, height=280)
    else:
      st.info(
          '💡 실시간 AIS 신호가 수신되지 않았습니다. (AISStream 서버 응답'
          ' 대기 중)'
      )

      facility_nm = v.get('laidup_fclty_nm', '-')
      st.write(f'- **PORT-MIS 신고 계선장소:** `{facility_nm}`')

      if imo_number and imo_number not in ['-', '0000', '없음', '']:
        mt_link = f'https://www.marinetraffic.com/en/ais/details/ships/imo:{imo_number}'
        st.markdown(
            f"🔗 **[MarineTraffic에서 `{v['vssl_nm']}` (IMO: {imo_number})"
            f' 실시간 위치 상세 보기]({mt_link})**'
        )
      else:
        mt_area_link = 'https://www.marinetraffic.com/en/ais/home/centerx:126.6/centery:37.0/zoom:11'
        st.markdown(
            '🔗 **[MarineTraffic 평택·대산항 관제 해역 지도에서'
            f' `{v["vssl_nm"]}` 위치 확인하기]({mt_area_link})**'
        )

      default_m = folium.Map(location=[37.00, 126.60], zoom_start=10)
      components.html(default_m._repr_html_(), height=260)


def render_vessel_item_card(v, port_code, idx):
  expander_label = (
      f"🚢 [{v['vssl_nm']}] 호출부호: {v['clsgn']} ｜ 선종:"
      f" {v['vssl_knd_nm']} ｜ 계선장소: {v['laidup_fclty_nm']}"
  )

  with st.expander(expander_label, expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
      st.markdown('**[선박 및 국적 정보]**')
      st.write(f"- **항구청명:** {v['prt_ag_nm']}")
      st.write(f"- **선박명:** **{v['vssl_nm']}**")
      st.write(f"- **호출부호:** `{v['clsgn']}`")
      st.write(f"- **선박국가명:** {v['vssl_nlty_nm']}")
      st.write(f"- **선박종류명:** {v['vssl_knd_nm']}")
      st.write(
          f"- **입항년도/횟수:** {v['etrypt_year']}년 / {v['etrypt_co']}회"
      )
      st.write(f"- **총톤수:** {v['grtg']} 톤")
      st.write(f"- **선원수:** {v['crew_co']} 명")

    with col2:
      st.markdown('**[운항 및 관제 일시]**')
      st.write(f"- **입출항구분명:** {v['etrynd_nm']} ({v['reqst_se_nm']})")
      st.write(f"- **입항목적명:** {v['etrypt_purps_nm']}")
      st.write(f"- **입항일시:** {v['etrypt_dt']}")
      st.write(f"- **출항일시:** {v['tkoff_dt']}")
      st.write(f"- **출항예정일시:** {v['tkoff_prrrn_dt']}")
      st.write(f"- **목적지입항예정일시:** {v['dstn_etrypt_dt']}")
      st.write(f"- **계선시설명:** {v['laidup_fclty_nm']}")
      st.write(f"- **신고업체명:** {v['satmnt_entrps_nm']}")

    with col3:
      st.markdown('**[항로 및 화물 명세]**')
      st.write(f"- **전출항지항구명:** {v['prvs_dpmprt_prt_nm']}")
      st.write(f"- **차출항지항구명:** {v['nxlnpt_prt_nm']}")
      st.write(f"- **목적지항구명:** {v['dstn_prt_nm']}")
      st.write(f"- **화물명세 (코드):** {v['ldadng_frght_cl_cd']}")
      st.write(f"- **적재톤수:** {v['ldadng_ton']} 톤")
      st.write(f"- **환적톤수:** {v['trnpdt_ton']} 톤")
      st.write(f"- **양하화물톤:** {v['landng_frght_ton']} 톤")
      st.write(f"- **적하화물톤:** {v['ld_frght_ton']} 톤")

    st.markdown('---')
    if st.button(
        f"🔍 [{v['vssl_nm']}] 선박제원 및 위치정보 조회",
        key=f'btn_spec_{port_code}_{idx}',
        use_container_width=True,
    ):
      show_vessel_detail_dialog(v)


def render_combined_port_tab_content(port_name, port_code):
  now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
  today_date = now_kst.date()

  c_date1, c_date2, c_btn = st.columns([0.35, 0.35, 0.3])
  with c_date1:
    start_date = st.date_input(
        '조회 시작일', value=today_date, key=f'sdate_{port_code}'
    )
  with c_date2:
    end_date = st.date_input(
        '조회 종료일', value=today_date, key=f'edate_{port_code}'
    )
  with c_btn:
    st.markdown('<br>', unsafe_allow_html=True)
    query_trigger = st.button(
        '🔍 조회', key=f'search_btn_{port_code}', use_container_width=True
    )

  state_key_fetched = f'fetched_{port_code}'

  if query_trigger:
    st.cache_data.clear()
    st.session_state[state_key_fetched] = True

  if not st.session_state.get(state_key_fetched, False):
    st.info(
        f'💡 날짜 설정 후 우측의 **[🔍 조회]** 버튼을 클릭하면 {port_name} 입출항'
        ' 신고 선박 정보가 통합 조회됩니다.'
    )
    return

  sde_str = start_date.strftime('%Y%m%d')
  ede_str = end_date.strftime('%Y%m%d')
  sde_fmt = start_date.strftime('%Y-%m-%d')
  ede_fmt = end_date.strftime('%Y-%m-%d')

  st.markdown(
      f'#### 📊 {port_name} 입출항 신고 선박 현황 (`{sde_fmt}` ~ `{ede_fmt}`'
      ' 기준)'
  )

  with st.spinner(f'{port_name} 입항 및 출항 전체 선박 정보 통합 수집 중...'):
    in_vessels = fetch_vessel_schedule_api(port_code, 'I', sde_str, ede_str)
    out_vessels = fetch_vessel_schedule_api(port_code, 'O', sde_str, ede_str)
    vessels = in_vessels + out_vessels

  if not vessels:
    st.warning(
        f'💡 해당 기간({sde_fmt} ~ {ede_fmt}) {port_name} 입출항 신고 선박'
        ' 정보가 없습니다.'
    )
    return

  st.success(
      f'✅ 총 **{len(vessels)}** 척의 입출항 신고 선박 정보가 수집되었습니다.'
      f' (입항 {len(in_vessels)}척 / 출항 {len(out_vessels)}척)'
  )

  ALL_VIEW_OPTION = f'📋 전체선박 목록 보기 (총 {len(vessels)}척)'
  select_options = [ALL_VIEW_OPTION] + [
      f"🚢 [{v['vssl_nm']}] 구분: {v['etrynd_nm']} ｜ 호출부호: {v['clsgn']} ｜"
      f" 선종: {v['vssl_knd_nm']} ｜ 계선장소: {v['laidup_fclty_nm']}"
      for v in vessels
  ]

  selected_option = st.selectbox(
      '선박 필터링 선택 (개별 선박 선택 시 해당 선박만 표시됩니다):',
      options=select_options,
      key=f'filter_select_{port_code}',
  )

  st.markdown('<br>', unsafe_allow_html=True)

  if selected_option == ALL_VIEW_OPTION:
    for idx, v in enumerate(vessels):
      render_vessel_item_card(v, port_code, f'all_{idx}')
  else:
    selected_idx = select_options.index(selected_option) - 1
    if 0 <= selected_idx < len(vessels):
      selected_vessel = vessels[selected_idx]
      render_vessel_item_card(
          selected_vessel, port_code, f'single_{selected_idx}'
      )


# ==========================================
# 5. 메인 화면 구성 (Hero Section & 로고 정렬)
# ==========================================
if kcg_logo_b64:
  st.markdown(
      f"""
    <div class="hero-container">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 6px;">
            <img src="data:image/png;base64,{kcg_logo_b64}" style="width: 58px; height: auto; object-fit: contain;" alt="해양경찰 로고" />
            <div class="main-header" style="margin: 0;">평택해양경찰서 HNS AI 대응 시스템</div>
        </div>
        <div class="sub-header">공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 정보집, HNS 대응가이드) + Gemini AI</div>
    </div>
    """,
      unsafe_allow_html=True,
  )
else:
  st.markdown(
      """
    <div class="hero-container">
        <div class="main-header">🚢 평택해양경찰서 HNS AI 대응 시스템</div>
        <div class="sub-header">공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 정보집, HNS 대응가이드) + Gemini AI</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

# ==========================================
# 🔥 상단 통합 검색 시스템 (2개 탭 분기)
# ==========================================
st.markdown('### 🔎 통합 검색 시스템')

search_tab_chem, search_tab_vssl = st.tabs([
    '🧪 HNS 물질 및 사고상황 AI 검색',
    '🚢 선박 제원 및 위치 검색',
])

# ------------------------------------------
# [탭 1]: 화학물질 및 사고상황 AI 검색 (Form 기반 안정화 + 검색 버튼)
# ------------------------------------------
with search_tab_chem:
  with st.form(key='chem_search_form', clear_on_submit=False):
    col_c1, col_c2 = st.columns([4, 1])
    with col_c1:
      search_input_val = st.text_input(
          '화학물질명, 화학식, 관용명 또는 사고상황을 자유롭게 입력하세요 (예:'
          ' 황산, H2SO4, LNG / 평택호 좌초로 질산 유출 중):',
          key='global_search_box_input',
      )
    with col_c2:
      st.markdown('<br>', unsafe_allow_html=True)
      submit_chem_search = st.form_submit_button(
          '🔍 검색', use_container_width=True
      )

  if submit_chem_search and search_input_val:
    st.session_state['active_search_query'] = search_input_val.strip()

  # 💡 검색된 질의가 있을 때 AI 매핑 결과와 가이드 생성 버튼을 항상 안정적으로 노출
  curr_q = st.session_state.get('active_search_query', '')
  if curr_q:
    with st.spinner('Gemini AI가 입력 내용을 지능형 분석 중...'):
      mapped_result = map_search_query_with_gemini(curr_q)
      mapped_ko = mapped_result.get('chem_ko', curr_q)
      mapped_eng = mapped_result.get('chem_eng', curr_q)
      mapped_unno = str(mapped_result.get('unno', '0000')).zfill(4)
      mapped_cas = str(mapped_result.get('cas_no', '-'))
      accident_ctx = mapped_result.get('accident_context', '')

    c1, c2 = st.columns([4, 1])
    with c1:
      info_msg = (
          f'💡 **AI 매핑 결과:** 물질명: **{mapped_ko}** ({mapped_eng}) ｜ UN'
          f' NO: `{mapped_unno}` ｜ CAS NO: `{mapped_cas}`'
      )
      if accident_ctx:
        info_msg += f'\n ｜ 🚨 **사고상황 식별:** `{accident_ctx}`'
      st.info(info_msg)
    with c2:
      if st.button(
          '🤖 AI 가이드 생성',
          key='btn_global_search',
          use_container_width=True,
      ):
        st.session_state['active_chem'] = mapped_ko
        st.session_state['active_unno'] = mapped_unno
        st.session_state['active_cas'] = mapped_cas
        st.session_state['active_ship'] = f"자유 통합 검색 ('{curr_q}')"
        st.session_state['active_accident_context'] = accident_ctx
        st.session_state['active_summary'] = ''
        st.session_state['active_key_changed'] = True
        st.rerun()

# ------------------------------------------
# [탭 2]: 선박 제원 및 위치 검색
# ------------------------------------------
with search_tab_vssl:
  with st.form(key='vssl_search_form', clear_on_submit=False):
    col_v1, col_v2 = st.columns([4, 1])
    with col_v1:
      vssl_query_input = st.text_input(
          '선박명(한/영) 또는 호출부호를 입력하세요 (예: DAITOMO 7, 대형카훼리,'
          ' 049034, PACIFIC):',
          key='vssl_direct_search_box',
      )
    with col_v2:
      st.markdown('<br>', unsafe_allow_html=True)
      btn_vssl_search = st.form_submit_button(
          '🔍 검색', use_container_width=True
      )

  if btn_vssl_search and vssl_query_input:
    clean_query = vssl_query_input.strip()
    with st.spinner(
        f"해수부 선박제원 API에서 '{clean_query}' 관련 선박 검색 중..."
    ):
      vssl_list = fetch_vessel_spec_list_api(clean_query, max_results=50)
      st.session_state['vssl_search_results'] = vssl_list
      st.session_state['vssl_search_keyword'] = clean_query

  # 검색 결과 표출 영역
  if 'vssl_search_results' in st.session_state:
    results = st.session_state['vssl_search_results']
    kw = st.session_state.get('vssl_search_keyword', '')

    if not results:
      st.warning(
          f"💡 '{kw}'에 해당하는 선박 정보를 해수부 선박제원 API에서 찾을 수"
          ' 없습니다. 철자를 확인해주세요.'
      )
    else:
      st.success(
          f"✅ '{kw}' 검색 결과 총 **{len(results)}**척의 선박이 검색되었습니다."
          ' (최대 50척 표출)'
      )

      vssl_labels = [
          f"🚢 [{s['displayName']}] 호출부호: {s['clsgn']} ｜ MMSI:"
          f" {s.get('mmsiNo', '-')} ｜ 국적: {s['vsslNlty']} ｜ 선종:"
          f" {s['vsslKnd']}"
          for s in results
      ]
      selected_vessel_label = st.selectbox(
          '조회할 선박을 선택하세요:',
          options=vssl_labels,
          key='vssl_search_select_box',
      )

      if selected_vessel_label:
        sel_idx = vssl_labels.index(selected_vessel_label)
        selected_spec = results[sel_idx]

        st.markdown('<br>', unsafe_allow_html=True)
        col_btn1, _ = st.columns([2, 3])
        with col_btn1:
          if st.button(
              f"🔍 [{selected_spec['displayName']}] 선박제원 및 실시간 위치"
              ' 상세 보기',
              key='btn_open_searched_vessel_modal',
              use_container_width=True,
          ):
            dummy_vessel_obj = {
                'vssl_nm': selected_spec['displayName'],
                'clsgn': selected_spec.get('clsgn', kw).upper(),
                'laidup_fclty_nm': '선박 직접 검색 결과',
            }
            show_vessel_detail_dialog(dummy_vessel_obj)

# ------------------------------------------
# ⚡ AI 대응 가이드 출력 모달/컨테이너 (Vision 연동)
# ------------------------------------------
if 'active_chem' in st.session_state:
  st.divider()
  chem = st.session_state['active_chem']
  unno = st.session_state['active_unno']
  cas = st.session_state.get('active_cas', '-')
  ship_info = st.session_state['active_ship']
  accident_ctx = st.session_state.get('active_accident_context', '')

  status_header = (
      f'⚡ [지능형 비상대응 가이드] 대상: {ship_info} ｜ 물질: {chem} (UN NO:'
      f' {unno} / CAS NO: {cas})'
  )
  if accident_ctx:
    status_header += f' ｜ 상황: {accident_ctx}'
  st.error(status_header)

  st.caption(
      '⚠️ **[할루시네이션 주의]** 본 대응 가이드는 공공 API 3종 및 해경 HNS'
      ' 정보집, HNS 대응가이드를 통합한 Gemini RAG(검색증강생성) 모델로 AI 환각'
      ' 현상을 최소화했습니다. **단, 현장 상황은 가이드와 다를 수 있으므로'
      ' 반드시 재확인하시기 바랍니다.**'
  )

  if (
      'active_summary' not in st.session_state
      or st.session_state.get('active_key_changed', False)
      or not st.session_state['active_summary']
  ):
    with st.spinner(
        '공공 API + HNS 정보집 + HNS 대응가이드 Gemini AI 종합 분석 중...'
    ):
      dgst_info = fetch_dgst_info(unno)
      safety_info = fetch_chem_safety_info(cas)
      kosha_msds_text = fetch_kosha_msds_info(chem, cas, unno)

      pil_image, page_no = get_hns_page_image(
          unno if unno != '0000' else chem, cas_no=cas
      )

      hazard_kind = dgst_info.get('kndNm', '')
      em_s = dgst_info.get('emergManagtCd', '')
      situation_keyword = (
          accident_ctx if accident_ctx else '해상 화재 및 유출 복합사고'
      )

      rag_search_query = (
          f'위험유해물질 HNS {chem} 성상 분류 {hazard_kind} 비상대응지침'
          f' EmS {em_s} 상황별 방제조치 및 안전거리 이격지침'
          f' {situation_keyword}'
      )

      print(f'🔍 [RAG 고도화 쿼리]: {rag_search_query}')

      # 💡 5페이지(k=5)로 빠른 추출
      rag_text, rag_images = fetch_rag_context_and_images(
          rag_search_query, k=5
      )

      st.session_state['active_source_data'] = {
          'dgst': dgst_info,
          'safety': safety_info,
          'hns_image': pil_image,
          'hns_page_no': page_no,
          'rag_text': rag_text,
          'rag_images': rag_images,
          'kosha': kosha_msds_text,
      }

      st.session_state['active_summary'] = generate_gemini_vision_summary(
          chem,
          unno,
          cas,
          dgst_info,
          safety_info,
          kosha_msds_text,
          rag_text,
          hns_pil_image=pil_image,
          hns_page_no=page_no,
          rag_images=rag_images,
          accident_context=accident_ctx,
      )
      st.session_state['active_key_changed'] = False

  st.markdown(st.session_state['active_summary'])

  # ------------------------------------------
  # 📚 활용 원본 자료 확인 탭 (스캔 이미지 뷰어 제공)
  # ------------------------------------------
  if 'active_source_data' in st.session_state:
    src = st.session_state['active_source_data']
    with st.expander(
        '📚 생성 정보 출처 및 활용 원본 데이터 검증/보기', expanded=False
    ):
      t1, t2, t3, t4, t5 = st.tabs([
          '🚢 해수부 위험물정보',
          '🛡️ 화학물질안전원',
          '🖼️ 해경 HNS 정보집',
          '🧠 해경 HNS 대응가이드',
          '🏥 안전보건공단 MSDS',
      ])

      with t1:
        st.markdown('**[해양수산부 위험물정보 API]**')
        d = src.get('dgst', {})
        st.write(
            f"- **IMDG 한글/영문명:** {d.get('imdgNm', '-')} ({d.get('imdgEngNm', '-')})"
        )
        st.write(
            f"- **IMDG 등급 / 종류:** {d.get('imdgGradCd', '-')} /"
            f" {d.get('kndNm', '-')}"
        )
        st.write(f"- **비상조치코드(EmS):** {d.get('emergManagtCd', '-')}")
        st.write(f"- **선박 적재방법:** {d.get('ldadngMth', '-')}")
        st.write(f"- **주의사항:** {d.get('catinMatter', '-')}")

      with t2:
        st.markdown('**[화학물질안전원 화학물질안전관리정보 API]**')
        s = src.get('safety', {})
        st.write(f"- **표적장기 및 주요증상:** {s.get('symptom', '-')}")
        st.write(f"- **흡입 영향:** {s.get('inhale', '-')}")
        st.write(f"- **피부 노출:** {s.get('skin', '-')}")
        st.write(f"- **안구 노출:** {s.get('eyeball', '-')}")
        st.write(f"- **기타 유의사항:** {s.get('etc', '-')}")

      with t3:
        st.markdown('**[해양경찰청 HNS 정보집]**')
        hns_img = src.get('hns_image')
        hns_pno = src.get('hns_page_no')
        if hns_img and hns_pno:
          st.success(
              f'📖 해상운송 위험유해물질 정보집(2024년 개정판) **{hns_pno}쪽**'
              ' 원본 페이지입니다.'
          )
          st.image(
              hns_img,
              caption=f'HNS 정보집 {hns_pno}쪽 실시간 렌더링 스캔 이미지',
              use_container_width=True,
          )
        else:
          st.info(
              '💡 해당 물질의 HNS 정보집 원본 페이지 스캔을 찾을 수 없습니다.'
          )

      with t4:
        st.markdown('**[해양경찰청 HNS 해양사고 대응 가이드]**')
        rag_imgs = src.get('rag_images', [])
        if rag_imgs:
          for r_item in rag_imgs:
            st.caption(
                f"📖 위험유해물질(HNS) 해양사고 대응 가이드 **{r_item['page_no']}쪽**"
                ' 원본 페이지'
            )
            st.image(r_item['pil_img'], use_container_width=True)
            st.divider()
        else:
          st.info('💡 연관 대응가이드 원본 스캔 페이지가 없습니다.')

      with t5:
        st.markdown('**[안전보건공단 MSDS API]**')
        st.text_area(
            'MSDS 세부 수집 정보',
            value=src.get('kosha', ''),
            height=200,
            disabled=True,
        )

  st.markdown('<br>', unsafe_allow_html=True)
  if st.button(
      '❌ 가이드 창 닫기', key='close_global_guide', use_container_width=True
  ):
    for key in [
        'active_chem',
        'active_unno',
        'active_cas',
        'active_ship',
        'active_accident_context',
        'active_summary',
        'active_source_data',
        'active_key_changed',
    ]:
      if key in st.session_state:
        del st.session_state[key]
    st.rerun()

st.divider()

# ------------------------------------------
# ⚓ 실시간 항만 선박 모니터링 (평택항 / 대산항 2개 탭)
# ------------------------------------------
st.markdown('### ⚓ 항만별 실시간 선박입출항현황 (PORT-MIS)')

tab_pt, tab_ds = st.tabs(['🚢 평택항 입출항 선박', '🚢 대산항 입출항 선박'])

with tab_pt:
  render_combined_port_tab_content('평택항', '031')

with tab_ds:
  render_combined_port_tab_content('대산항', '300')
