import base64
from datetime import datetime, timedelta, timezone
import json
import os
import re
import threading
import time
import urllib3
import xml.etree.ElementTree as ET

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
      _ = datetime.now()
    except Exception:
      pass


if 'keep_alive_started' not in st.session_state:
  st.session_state['keep_alive_started'] = True
  t = threading.Thread(target=keep_app_alive, daemon=True)
  t.start()

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
    page_title='평택해양경찰서 HNS AI 대응 시스템 (Vision)',
    page_icon=KCG_LOGO_PATH if os.path.exists(KCG_LOGO_PATH) else '🚢',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# Reflex.dev 감성 Light UI + 모바일 완벽 가시성 보장 CSS
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


# ==========================================
# 1. 🖼️ PDF 인덱스 맵 생성 (스캔 범위를 인덱스 38~221로 안전 확장)
# ==========================================

@st.cache_data
def build_hns_pdf_index(pdf_path):
    """
    HNS 정보집 PDF 물리적 인덱스 직접 매핑
    - 과산화수소 누락 방지를 위해 인덱스 38(물리 39쪽)부터 안전 스캔
    """
    if not os.path.exists(pdf_path):
        return []

    index_list = []
    with pdfplumber.open(pdf_path) as pdf:
        # 안전하게 인덱스 38 ~ 220 (물리적 39페이지 ~ 221페이지) 스캔
        start_idx = 38
        end_idx = min(221, len(pdf.pages))
        
        for idx in range(start_idx, end_idx):
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            
            if not text.strip():
                continue

            # UN번호 추출 정규식 보완 (공백/줄바꿈 유연 처리)
            unno_match = re.search(r'UN\s*번호\s*[:\s]*(\d{4})', text, re.IGNORECASE) or re.search(r'UN\s*(\d{4})', text, re.IGNORECASE)
            unno = unno_match.group(1).strip() if unno_match else ""
            
            # 페이지 줄 단위 파싱
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            title = lines[0] if lines else ""
            
            synonym_match = re.search(r'유사명\s*[:\s]*([^\n]+)', text)
            synonyms = synonym_match.group(1).strip() if synonym_match else ""

            index_list.append({
                "page_index": idx,            # 0-based 물리 인덱스 (예: 39)
                "display_page_no": idx + 1,   # 1-based 물리 페이지 번호 (예: 40페이지)
                "unno": unno,
                "title": title,
                "synonyms": synonyms,
                "raw_text": text.upper()      # 💡 전체 텍스트 상위 매칭용 보관
            })
    return index_list

hns_pdf_index = build_hns_pdf_index(HNS_PDF_PATH)


def get_hns_page_image(unno_or_query):
    """UN번호, 물질명, 유사명 및 전체 텍스트 교차 검색으로 해당 PDF 페이지 렌더링"""
    if not hns_pdf_index or not unno_or_query or not os.path.exists(HNS_PDF_PATH):
        return None, None

    q = str(unno_or_query).strip().upper()
    target_item = None

    for item in hns_pdf_index:
        unno = str(item['unno']).strip()
        title = item['title'].upper()
        synonyms = item['synonyms'].upper()
        raw_text = item.get('raw_text', '')

        # 1차: UN번호 또는 제목/유사명 직접 일치
        if (q and q == unno) or (q in title) or (q in synonyms):
            target_item = item
            break
        # 2차: 페이지 전체 텍스트 내 포함 여부 (과산화수소 등 예외 케이스 방지)
        elif len(q) >= 2 and q in raw_text:
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
        print(f"HNS 정보집 PDF 이미지 렌더링 에러: {e}")
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


def fetch_rag_context_and_images(query, k=5):
  """RAG 검색 결과 텍스트와 함께 해당 페이지의 고화질 이미지 리스트를 반환"""
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

    # 🖼️ 해당 쪽수들을 대응가이드 PDF에서 고화질 이미지로 렌더링
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
  search_trials = [(unno, '2'), (cas_no, '1'), (chem_name, '0')]

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


def fetch_vessel_spec_api(clsgn, vssl_nm):
  if not PUBLIC_API_KEY:
    return None
  url = f'https://apis.data.go.kr/1192000/SicsVsslManp3/Info3?serviceKey={PUBLIC_API_KEY}'
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
  trials = [('clsgn', str(clsgn).strip()), ('vsslNm', str(vssl_nm).strip())]

  for param_key, param_val in trials:
    if not param_val or param_val == '-':
      continue
    params = {param_key: param_val, 'numOfRows': '1', 'pageNo': '1'}
    try:
      session = requests.Session()
      session.verify = False
      res = session.get(url, params=params, headers=headers, timeout=8)
      if res.status_code == 200 and res.content:
        root = ET.fromstring(res.content)
        item = root.find('.//item')
        if item is not None:
          return {
              'vsslNo': (item.findtext('vsslNo') or '-').strip(),
              'imoNo': (item.findtext('imoNo') or '-').strip(),
              'vsslKorNm': (item.findtext('vsslKorNm') or '-').strip(),
              'vsslEngNm': (item.findtext('vsslEngNm') or '-').strip(),
              'vsslKnd': (item.findtext('vsslKnd') or '-').strip(),
              'vsslNlty': (item.findtext('vsslNlty') or '-').strip(),
              'grtg': (item.findtext('grtg') or '-').strip(),
              'vsslTotLt': (item.findtext('vsslTotLt') or '-').strip(),
              'shdth': (item.findtext('shdth') or '-').strip(),
              'vsslDrft': (item.findtext('vsslDrft') or '-').strip(),
              'vsslDp': (item.findtext('vsslDp') or '-').strip(),
              'brbtSeNm': (item.findtext('brbtSeNm') or '-').strip(),
              'nvgShapNm': (item.findtext('nvgShapNm') or '-').strip(),
              'vsslCnstrDt': (
                  (item.findtext('vsslCnstrDt') or '-')
                  .strip()
                  .replace('T', ' ')
              ),
          }
    except Exception as e:
      print(f'선박제원 API 조회 에러: {e}')
  return None


def fetch_cargo_inout_api(port_code, etrypt_year, etrypt_co, clsgn):
  if not PUBLIC_API_KEY:
    return []
  url_in = f'https://apis.data.go.kr/1192000/CargFrghtIn2/Info?serviceKey={PUBLIC_API_KEY}'
  url_out = f'https://apis.data.go.kr/1192000/CargFrghtOut4/Info4?serviceKey={PUBLIC_API_KEY}'
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
  params = {
      'prtAgCd': str(port_code).strip(),
      'etryptYear': str(etrypt_year).strip(),
      'etryptCo': str(etrypt_co).strip().zfill(3),
      'clsgn': str(clsgn).strip(),
      'numOfRows': '50',
      'pageNo': '1',
  }
  combined_cargo_list = []

  # 1. 내항화물
  try:
    session = requests.Session()
    session.verify = False
    res_in = session.get(url_in, params=params, headers=headers, timeout=8)
    if res_in.status_code == 200 and res_in.content:
      root = ET.fromstring(res_in.content)
      for item in root.findall('.//item') or root.findall('body/items/item'):
        kor_nm = (
            item.findtext('frghtPrdlstKorNm')
            or item.findtext('frghtPrdlstNm')
            or '-'
        ).strip()
        dgst_code = (item.findtext('dgstLdadngYn') or '-').strip()
        extracted_unno = ''
        if kor_nm != '-' or dgst_code == '1':
          map_res = map_search_query_with_gemini(kor_nm)
          extracted_unno = str(map_res.get('unno', '0000')).zfill(4)
          if extracted_unno == '0000':
            extracted_unno = ''

        combined_cargo_list.append({
            'cargo_type': '내항화물',
            'etryndDt': (
                (item.findtext('etryndDt') or '-').strip().replace('T', ' ')
            ),
            'tkinTkoutNm': (item.findtext('tkinTkoutNm') or '-').strip(),
            'laidupFcltyNm': (item.findtext('laidupFcltyNm') or '-').strip(),
            'frghtPrdlstCd': (item.findtext('frghtPrdlstCd') or '-').strip(),
            'frghtPrdlstKorNm': kor_nm,
            'frghtPrdlstEngNm': (
                item.findtext('frghtPrdlstEngNm') or '-'
            ).strip(),
            'dgstLdadngYn': (
                '⚠️ 위험물 적재' if dgst_code == '1' else '일반/비위험물'
            ),
            'frghtClCd': (item.findtext('frghtClCd') or '-').strip(),
            'frghtClNm': (item.findtext('frghtClNm') or '-').strip(),
            'lduldPrtNm': (item.findtext('lduldPrtNm') or '-').strip(),
            'wtTon': (item.findtext('wtTon') or '-').strip(),
            'satmntDt': (
                (item.findtext('satmntDt') or '-').strip().replace('T', ' ')
            ),
            'packngKndNm': (item.findtext('packngKndNm') or '-').strip(),
            'lnlMthNm': (item.findtext('lnlMthNm') or '-').strip(),
            'aprtfEtryptDt': (
                (item.findtext('aprtfEtryptDt') or '-')
                .strip()
                .replace('T', ' ')
            ),
            'lnlEntrpsNm': (item.findtext('lnlEntrpsNm') or '-').strip(),
            'unno': extracted_unno,
        })
  except Exception as e:
    print(f'내항화물 API 예외: {e}')

  # 2. 외항화물
  try:
    session = requests.Session()
    session.verify = False
    res_out = session.get(url_out, params=params, headers=headers, timeout=8)
    if res_out.status_code == 200 and res_out.content:
      root = ET.fromstring(res_out.content)
      for item in root.findall('.//item') or root.findall('body/items/item'):
        kor_nm = (
            item.findtext('frghtPrdlstKorNm')
            or item.findtext('frghtPrdlstNm')
            or '-'
        ).strip()
        raw_unno = (item.findtext('unno') or '').strip()
        raw_unno = '' if raw_unno in ['-', '0000', 'None', ''] else raw_unno.zfill(4)

        extracted_unno = raw_unno
        if not extracted_unno and kor_nm != '-':
          map_res = map_search_query_with_gemini(kor_nm)
          extracted_unno = str(map_res.get('unno', '0000')).zfill(4)
          if extracted_unno == '0000':
            extracted_unno = ''

        combined_cargo_list.append({
            'cargo_type': '외항화물',
            'laidupFcltyNm': (item.findtext('laidupFcltyNm') or '-').strip(),
            'frghtPrdlstCd': (item.findtext('frghtPrdlstCd') or '-').strip(),
            'frghtPrdlstKorNm': kor_nm,
            'unno': extracted_unno,
            'frghtClCd': (item.findtext('frghtClCd') or '-').strip(),
            'frghtClNm': (item.findtext('frghtClNm') or '-').strip(),
            'etryndDt': (
                (item.findtext('etryndDt') or '-').strip().replace('T', ' ')
            ),
            'ldPrtNm': (item.findtext('ldPrtNm') or '-').strip(),
            'satmntDt': (
                (item.findtext('satmntDt') or '-').strip().replace('T', ' ')
            ),
            'packngKndNm': (item.findtext('packngKndNm') or '-').strip(),
            'lnlEntrpsNm': (item.findtext('lnlEntrpsNm') or '-').strip(),
            'lnlMthNm': (item.findtext('lnlMthNm') or '-').strip(),
            'wtTon': (item.findtext('wtTon') or '-').strip(),
            'aprtfEtryptDt': (
                (item.findtext('aprtfEtryptDt') or '-')
                .strip()
                .replace('T', ' ')
            ),
            'spprnNm': (item.findtext('spprnNm') or '-').strip(),
            'contnCo': (item.findtext('contnCo') or '-').strip(),
            'dgstYn': '⚠️ 위험물(UN.No)' if extracted_unno else '일반화물',
        })
  except Exception as e:
    print(f'외항화물 API 예외: {e}')

  return combined_cargo_list


# ==========================================
# 3. 🧠 Gemini Vision 멀티모달 프롬프트 연동
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
    for model_id in [
        'gemini-3.6-flash',
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash',
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
  """[멀티모달 Gemini Vision 통합 생성] - HNS 정보집 스캔 이미지 1장 + RAG 대응가이드 스캔 이미지 N장 동시 전달"""
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
        f'\n- [첨부 이미지 1]: 해경 HNS 정보집 {hns_page_no}쪽 스캔'
        ' 이미지입니다. 표와 세로 쓰기, 픽토그램 수치(NFPA 등)를 이미지에서 직접'
        ' 판독하여 반영하세요.\n'
        if (hns_pil_image and hns_page_no)
        else ''
    )

    rag_img_prompt = ''
    if rag_images:
      rag_pages_str = ', '.join([f"{img['page_no']}쪽" for img in rag_images])
      rag_img_prompt = (
          '\n- [첨부 이미지 추가]: 해경 HNS 대응가이드'
          f' ({rag_pages_str}) 스캔 이미지들입니다.\n'
      )

    prompt_text = f"""
        당신은 해양경찰청 및 항만 HNS 비상대응 상황실 관제관입니다.
        첨부된 [HNS 정보집 스캔 이미지] 및 [HNS 해양사고 대응가이드 스캔 이미지들], 그리고 수집된 공공 API 데이터들을 종합 분석하여, 관제관이 현장 세력(OSC, 함정, 구조대 등)에 바로 지시/전파할 수 있는 비상대응 가이드를 작성하세요.

        {accident_info}
        {hns_img_prompt}
        {rag_img_prompt}

        [해양수산부 위험물정보 API 수집 데이터]
        - 물질명: {chem_name} (UN NO: {unno})
        - IMDG 명칭: {dgst_info.get('imdgNm')} ({dgst_info.get('imdgEngNm')})
        - IMDG 등급코드 / 종류명: {dgst_info.get('imdgGradCd', '-')} / {dgst_info.get('kndNm', '-')}
        - 비상조치코드(EmS): {dgst_info.get('emergManagtCd', '-')}
        - 주의사항: {dgst_info.get('catinMatter', '-')}

        [화학물질안전원 안전관리정보 API 수집 데이터 (CAS NO: {cas_no})]
        - 일반 증상 및 표적장기: {safety_info.get('symptom', '-')}
        - 흡입/피부/안구/경구 영향: {safety_info.get('inhale', '-')}, {safety_info.get('skin', '-')}, {safety_info.get('eyeball', '-')}, {safety_info.get('oral', '-')}

        [안전보건공단 MSDS 1~16번 종합 수집 데이터]
        {kosha_msds_text}

        [상황실 지침 반영 엄격 작성 규칙]
        1. [초동대응 핵심요약]: 각 항목의 시작은 `* **항목명**:` 포맷을 사용하고, 현장 실행 위주의 명확한 개조식 문장으로 작성하세요.
        2. [수치 및 안전 기준]: 
           - 이격거리 및 보호구 등 핵심 수치는 첨부된 HNS 대응가이드 및 정보집 원본 이미지상의 수치를 최우선 반영하세요.
           - 물질명 미확인 시 기본 유출 100m / 화재 800m 이격 조치를 지정하세요.
           - 물 반응성 물질 확인 시 직사주수 절대 금지를 명시하세요.
        3. [사고 상황 맞춤 지침]: [현장 사고 상황 조건]이 존재할 경우 해당 사고 유형별 비상조치 지침을 최우선 포함하세요.

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

    # 🖼️ 멀티모달 입력 구성 (텍스트 + HNS정보집 이미지 1장 + RAG 대응가이드 이미지 N장)
    contents_input = [prompt_text]
    if hns_pil_image:
      contents_input.append(hns_pil_image)

    for r_img in rag_images:
      contents_input.append(r_img['pil_img'])

    for model_id in [
        'gemini-3.6-flash',
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash',
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
# 4. 모달 팝업 및 UI 렌더링 함수
# ==========================================


@st.dialog('🚢 선박 제원 및 화물 반출입 상세정보', width='large')
def show_vessel_detail_dialog(v, port_code):
  st.subheader(f"⚓ {v['vssl_nm']} (`{v['clsgn']}`)")
  with st.spinner(
      '해양수산부 API로부터 선박제원 및 화물반출입 정보를 수집 중...'
  ):
    spec_info = fetch_vessel_spec_api(v['clsgn'], v['vssl_nm'])
    cargo_list = fetch_cargo_inout_api(
        port_code, v['etrypt_year'], v['etrypt_co'], v['clsgn']
    )

  st.markdown('#### 📐 선박 제원 정보')
  if spec_info:
    c1, c2, c3 = st.columns(3)
    with c1:
      st.write(f"- **선박번호:** {spec_info['vsslNo']}")
      st.write(f"- **IMO번호:** {spec_info['imoNo']}")
      st.write(f"- **선박한글명:** {spec_info['vsslKorNm']}")
      st.write(f"- **선박영문명:** {spec_info['vsslEngNm']}")
      st.write(f"- **선박종류:** {spec_info['vsslKnd']}")
    with c2:
      st.write(f"- **선박국적:** {spec_info['vsslNlty']}")
      st.write(f"- **총톤수:** {spec_info['grtg']} 톤")
      st.write(f"- **선박총길이:** {spec_info['vsslTotLt']} m")
      st.write(f"- **선박너비:** {spec_info['shdth']} m")
      st.write(f"- **선박흘수:** {spec_info['vsslDrft']} m")
    with c3:
      st.write(f"- **선박깊이:** {spec_info['vsslDp']} m")
      st.write(f"- **나용선구분:** {spec_info['brbtSeNm']}")
      st.write(f"- **운항형태:** {spec_info['nvgShapNm']}")
      st.write(f"- **선박건조일시:** {spec_info['vsslCnstrDt']}")
  else:
    st.warning('💡 해당 선박의 제원 정보가 없거나 조회에 실패했습니다.')

  st.divider()

  st.markdown(f'#### 📦 화물 반출입 정보 (통합 수집: 총 {len(cargo_list)}건)')
  if cargo_list:
    for c_idx, c in enumerate(cargo_list):
      with st.container(border=True):
        st.caption(f"📌 구분: **{c['cargo_type']}**")
        if c['cargo_type'] == '외항화물':
          c1, c2, c3 = st.columns(3)
          with c1:
            st.write(f"- **계선장소명:** {c['laidupFcltyNm']}")
            st.write(f"- **화물품목코드:** `{c['frghtPrdlstCd']}`")
            st.write(f"- **화물품목한글명:** **{c['frghtPrdlstKorNm']}**")
            st.write(f"- **UN.No:** `{c['unno'] if c['unno'] else '-'}`")
            st.write(
                f"- **화물분류 (코드/명):** {c['frghtClNm']}"
                f" (`{c['frghtClCd']}`)"
            )
            st.write(f"- **위험물구분:** {c['dgstYn']}")
          with c2:
            st.write(f"- **입출항일시:** {c['etryndDt']}")
            st.write(f"- **적하항구명:** {c['ldPrtNm']}")
            st.write(f"- **신고일시:** {c['satmntDt']}")
            st.write(f"- **포장종류명:** {c['packngKndNm']}")
            st.write(f"- **하역업체명:** {c['lnlEntrpsNm']}")
          with c3:
            st.write(f"- **하역방법명:** {c['lnlMthNm']}")
            st.write(f"- **중량톤:** {c['wtTon']} 톤")
            st.write(f"- **기항지입항일시:** {c['aprtfEtryptDt']}")
            st.write(f"- **선사명:** {c['spprnNm']}")
            st.write(f"- **컨테이너수:** {c['contnCo']} 개")
        else:
          c1, c2, c3 = st.columns(3)
          with c1:
            st.write(f"- **입출항일시:** {c['etryndDt']}")
            st.write(f"- **반입반출구분명:** {c['tkinTkoutNm']}")
            st.write(f"- **계선장소명:** {c['laidupFcltyNm']}")
            st.write(f"- **화물품목코드:** `{c['frghtPrdlstCd']}`")
            st.write(f"- **화물품목명:** **{c['frghtPrdlstKorNm']}**")
            st.write(f"- **화물품목영문명:** {c['frghtPrdlstEngNm']}")
          with c2:
            st.write(f"- **위험물적재유무:** {c['dgstLdadngYn']}")
            st.write(
                f"- **화물분류 (코드/명):** {c['frghtClNm']}"
                f" (`{c['frghtClCd']}`)"
            )
            st.write(f"- **양적하항구명:** {c['lduldPrtNm']}")
            st.write(f"- **중량톤:** {c['wtTon']} 톤")
            st.write(f"- **신고일시:** {c['satmntDt']}")
          with c3:
            st.write(f"- **포장종류명:** {c['packngKndNm']}")
            st.write(f"- **하역방법명:** {c['lnlMthNm']}")
            st.write(f"- **기항지입항일시:** {c['aprtfEtryptDt']}")
            st.write(f"- **하역업체명:** {c['lnlEntrpsNm']}")
            st.write(f"- **추정 UN.No:** `{c['unno'] if c['unno'] else '-'}`")

        target_unno = c.get('unno', '')
        chem_query = c['frghtPrdlstKorNm']
        if (
            target_unno
            or '위험물' in c.get('dgstLdadngYn', '')
            or c.get('dgstYn') == '⚠️ 위험물(UN.No)'
        ):
          if st.button(
              f'🤖 [{chem_query}] HNS AI 비상대응가이드 생성',
              key=f'btn_cargo_ai_{c_idx}',
              use_container_width=True,
          ):
            st.session_state['active_chem'] = chem_query
            st.session_state['active_unno'] = (
                target_unno if target_unno else '0000'
            )
            st.session_state['active_cas'] = '-'
            st.session_state['active_ship'] = (
                f"[{v['vssl_nm']}] 적재화물 ({chem_query})"
            )
            st.session_state['active_accident_context'] = (
                f"{c['laidupFcltyNm']} 반출입 화물 사고"
            )
            st.session_state['active_summary'] = ''
            st.session_state['active_key_changed'] = True
            st.rerun()
  else:
    st.info('💡 해당 선박의 내항/외항 화물 반출입 신고 내역이 없습니다.')


def render_vessel_item_card(v, port_name, port_code, de_gb, idx):
  expander_label = f"🚢 [{v['vssl_nm']}] 호출부호: {v['clsgn']} ｜ 선종: {v['vssl_knd_nm']} ｜ 계선장소: {v['laidup_fclty_nm']}"

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
        f"🔍 [{v['vssl_nm']}] 선박제원 및 화물반출입정보 조회",
        key=f'btn_spec_{port_code}_{de_gb}_{idx}',
        use_container_width=True,
    ):
      show_vessel_detail_dialog(v, port_code)


def render_vessel_tab_content(port_name, port_code, de_gb):
  gb_title = '입항' if de_gb == 'I' else '출항'
  now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
  today_date = now_kst.date()

  c_date1, c_date2, c_btn = st.columns([0.35, 0.35, 0.3])
  with c_date1:
    start_date = st.date_input(
        '조회 시작일', value=today_date, key=f'sdate_{port_code}_{de_gb}'
    )
  with c_date2:
    end_date = st.date_input(
        '조회 종료일', value=today_date, key=f'edate_{port_code}_{de_gb}'
    )
  with c_btn:
    st.markdown('<br>', unsafe_allow_html=True)
    query_trigger = st.button(
        '🔍 조회',
        key=f'search_btn_{port_code}_{de_gb}',
        use_container_width=True,
    )

  state_key_fetched = f'fetched_{port_code}_{de_gb}'

  if query_trigger:
    st.cache_data.clear()
    st.session_state[state_key_fetched] = True

  if not st.session_state.get(state_key_fetched, False):
    st.info(
        f'💡 날짜 설정 후 우측의 **[🔍 조회]** 버튼을 클릭하면 {port_name}'
        f' {gb_title} 신고 선박 정보가 조회됩니다.'
    )
    return

  sde_str = start_date.strftime('%Y%m%d')
  ede_str = end_date.strftime('%Y%m%d')
  sde_fmt = start_date.strftime('%Y-%m-%d')
  ede_fmt = end_date.strftime('%Y-%m-%d')

  st.markdown(
      f'#### 📊 {port_name} {gb_title} 신고 선박 현황 (`{sde_fmt}` ~ `{ede_fmt}`'
      ' 기준)'
  )

  with st.spinner(f'{port_name} {gb_title} 전체 선박 정보 수집 중...'):
    vessels = fetch_vessel_schedule_api(port_code, de_gb, sde_str, ede_str)

  if not vessels:
    st.warning(
        f'💡 해당 기간({sde_fmt} ~ {ede_fmt}) {port_name} {gb_title} 신고'
        ' 선박 정보가 없습니다.'
    )
    return

  st.success(
      f'✅ 총 **{len(vessels)}** 척의 {gb_title} 신고 선박이 수집되었습니다.'
  )

  ALL_VIEW_OPTION = f'📋 전체선박 목록 보기 (총 {len(vessels)}척)'
  select_options = [ALL_VIEW_OPTION] + [
      f"🚢 [{v['vssl_nm']}] 호출부호: {v['clsgn']} ｜ 선종: {v['vssl_knd_nm']} ｜"
      f" 계선장소: {v['laidup_fclty_nm']}"
      for v in vessels
  ]

  selected_option = st.selectbox(
      '선박 필터링 선택 (개별 선박 선택 시 해당 선박만 표시됩니다):',
      options=select_options,
      key=f'filter_select_{port_code}_{de_gb}',
  )

  st.markdown('<br>', unsafe_allow_html=True)

  if selected_option == ALL_VIEW_OPTION:
    for idx, v in enumerate(vessels):
      render_vessel_item_card(v, port_name, port_code, de_gb, f'all_{idx}')
  else:
    selected_idx = select_options.index(selected_option) - 1
    if 0 <= selected_idx < len(vessels):
      selected_vessel = vessels[selected_idx]
      render_vessel_item_card(
          selected_vessel,
          port_name,
          port_code,
          de_gb,
          f'single_{selected_idx}',
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
            <div class="main-header" style="margin: 0;">평택해양경찰서 HNS AI 대응 시스템 (Vision)</div>
        </div>
        <div class="sub-header">공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 원본 스캔 이미지, HNS 대응가이드) + Gemini Vision AI</div>
    </div>
    """,
      unsafe_allow_html=True,
  )
else:
  st.markdown(
      """
    <div class="hero-container">
        <div class="main-header">🚢 평택해양경찰서 HNS AI 대응 시스템 (Vision)</div>
        <div class="sub-header">공공 API(해양수산부, 화학물질안전원, 안전보건공단) + 해경 DB(HNS 원본 스캔 이미지, HNS 대응가이드) + Gemini Vision AI</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

# ------------------------------------------
# 🔥 HNS AI 통합 검색창 (물질명 및 사고상황 자유 입력)
# ------------------------------------------
st.markdown('### 🔎 AI 통합검색 (화학물질 또는 사고상황 입력)')
search_input = st.text_input(
    '화학물질명, 화학식, 관용명 또는 사고상황을 자유롭게 입력하세요 (예: 황산, H2SO4,'
    ' LNG / 평택호 좌초로 질산 유출 중)',
    key='global_search_box',
)

if search_input:
  with st.spinner('Gemini AI가 입력 내용을 지능형 분석 중...'):
    mapped_result = map_search_query_with_gemini(search_input)
    mapped_ko = mapped_result.get('chem_ko', search_input)
    mapped_eng = mapped_result.get('chem_eng', search_input)
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
        st.session_state['active_ship'] = (
            f"자유 통합 검색 ('{search_input}')"
        )
        st.session_state['active_accident_context'] = accident_ctx
        st.session_state['active_summary'] = ''
        st.session_state['active_key_changed'] = True
        st.rerun()

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
      ' 정보집 스캔 이미지 DB, HNS 대응가이드를 통합한 Gemini Vision'
      ' 모델로 AI 환각 현상을 최소화했습니다.'
  )

  if (
      'active_summary' not in st.session_state
      or st.session_state.get('active_key_changed', False)
      or not st.session_state['active_summary']
  ):
    with st.spinner(
        'HNS 정보집 PDF + 대응가이드 PDF 이미지 렌더링 및 Gemini Vision 종합'
        ' 분석 중...'
    ):
      dgst_info = fetch_dgst_info(unno)
      safety_info = fetch_chem_safety_info(cas)
      kosha_msds_text = fetch_kosha_msds_info(chem, cas, unno)

      # 1. HNS 정보집 원본 스캔 이미지 1장 렌더링
      pil_image, page_no = get_hns_page_image(unno if unno != '0000' else chem)

      # 2. RAG 대응가이드 검색 텍스트 + 연관 5개 쪽수 원본 이미지들 렌더링
      rag_search_query = f'{chem} {unno} {accident_ctx} 사고 대응 방제 조치'
      rag_text, rag_images = fetch_rag_context_and_images(
          rag_search_query, k=5
      )

      st.session_state['active_source_data'] = {
          'dgst': dgst_info,
          'safety': safety_info,
          'hns_image': pil_image,
          'hns_page_no': page_no,
          'rag_text': rag_text,
          'rag_images': rag_images,  # RAG 원본 스캔 이미지 세트 보관
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
  # 📚 활용 원본 자료 확인 탭 (진짜 스캔 이미지 뷰어 제공)
  # ------------------------------------------
  if 'active_source_data' in st.session_state:
    src = st.session_state['active_source_data']
    with st.expander(
        '📚 생성 정보 출처 및 활용 원본 데이터 검증/보기', expanded=False
    ):
      t1, t2, t3, t4, t5 = st.tabs([
          '🚢 해수부 위험물정보',
          '🛡️ 화학물질안전원',
          '🖼️ 해경 HNS 정보집 원본',
          '🧠 해경 HNS 대응가이드',
          '🏥 안전보건공단 MSDS',
      ])

      with t1:
        st.markdown("**[해양수산부 위험물정보 API]**")
        d = src.get('dgst', {})
        # 💡 [수정] f-string 따옴표 닫기 연산 오류 수정
        st.write(
            f"- **IMDG 한글/영문명:** {d.get('imdgNm', '-')} ({d.get('imdgEngNm', '-')})"
        )
        st.write(
            f"- **IMDG 등급 / 종류:** {d.get('imdgGradCd', '-')} / {d.get('kndNm', '-')}"
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
        st.markdown('**[해양경찰청 HNS 정보집 실시간 PDF 원본 스캔]**')
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
        st.markdown('**[해양경찰청 HNS 해양사고 대응 가이드 원본 스캔]**')
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
# ⚓ 실시간 항만 선박 모니터링 (4개 탭 구분)
# ------------------------------------------
st.markdown('### ⚓ 항만별 실시간 선박입출항현황 (PORT-MIS)')

tab_pt_in, tab_pt_out, tab_ds_in, tab_ds_out = st.tabs([
    '📥 평택항 입항 선박',
    '📤 평택항 출항 선박',
    '📥 대산항 입항 선박',
    '📤 대산항 출항 선박',
])

with tab_pt_in:
  render_vessel_tab_content('평택항', '031', 'I')

with tab_pt_out:
  render_vessel_tab_content('평택항', '031', 'O')

with tab_ds_in:
  render_vessel_tab_content('대산항', '300', 'I')

with tab_ds_out:
  render_vessel_tab_content('대산항', '300', 'O')
