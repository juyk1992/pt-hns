import base64
from datetime import datetime, timedelta, timezone
import json
import os
import re
import time
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

# ==========================================
# 🎨 디자인 시스템 (토큰 기반 리뉴얼)
# ==========================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

    :root {
        /* --- 컬러 토큰 --- */
        --bg-main: #F5F7FA;
        --bg-card: #FFFFFF;
        --bg-sub: #EEF2F7;
        --text-main: #0F172A;
        --text-sub: #64748B;
        --text-faint: #94A3B8;
        --border-color: #E2E8F0;
        --accent-blue: #2563EB;
        --accent-blue-hover: #1D4ED8;
        --accent-blue-soft: #EFF6FF;
        --accent-red: #DC2626;
        --accent-red-soft: #FEF2F2;
        --accent-orange: #EA580C;
        --accent-orange-soft: #FFF7ED;
        --accent-green: #059669;
        --accent-green-soft: #ECFDF5;

        /* --- 간격 토큰 --- */
        --space-1: 4px;
        --space-2: 8px;
        --space-3: 12px;
        --space-4: 16px;
        --space-6: 24px;
        --space-8: 32px;

        /* --- 반경 / 그림자 --- */
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 22px;
        --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
        --shadow-md: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-main: #0B1220;
            --bg-card: #151E2E;
            --bg-sub: #1E293B;
            --text-main: #F1F5F9;
            --text-sub: #94A3B8;
            --text-faint: #64748B;
            --border-color: #263349;
            --accent-blue: #60A5FA;
            --accent-blue-hover: #3B82F6;
            --accent-blue-soft: #1E293B;
            --accent-red-soft: #2A1618;
            --accent-orange-soft: #2A1F14;
            --accent-green-soft: #12241D;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 24px rgba(0,0,0,0.4);
        }
    }
    [data-theme="dark"] {
        --bg-main: #0B1220; --bg-card: #151E2E; --bg-sub: #1E293B;
        --text-main: #F1F5F9; --text-sub: #94A3B8; --text-faint: #64748B;
        --border-color: #263349; --accent-blue: #60A5FA; --accent-blue-hover: #3B82F6;
        --accent-blue-soft: #1E293B; --accent-red-soft: #2A1618;
        --accent-orange-soft: #2A1F14; --accent-green-soft: #12241D;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }
    p, span, div, label, h1, h2, h3, h4, h5, h6 { color: var(--text-main) !important; }
    .block-container { padding-top: 2rem !important; max-width: 1200px; }

    /* ===== 히어로 ===== */
    .hero-container {
        padding: 1.75rem 2rem;
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--accent-blue-soft) 130%) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-md) !important;
        margin-bottom: var(--space-6);
    }
    .hero-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
    .hero-left { display: flex; align-items: center; gap: 16px; }
    .hero-logo-badge {
        width: 54px; height: 54px; border-radius: 14px; background: var(--bg-card);
        display: flex; align-items: center; justify-content: center;
        box-shadow: var(--shadow-sm); border: 1px solid var(--border-color); overflow: hidden;
    }
    .main-header { font-size: 1.65rem; font-weight: 800; letter-spacing: -0.4px; margin: 0; line-height: 1.25; }
    .sub-header { color: var(--text-sub) !important; font-size: 0.85rem; font-weight: 500; margin-top: 2px; }
    .live-chip {
        display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px;
        background: var(--accent-green-soft); color: var(--accent-green) !important;
        border-radius: 999px; font-size: 0.78rem; font-weight: 700; border: 1px solid rgba(5,150,105,0.2);
    }
    .live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-green); animation: pulse 1.6s infinite; }
    @keyframes pulse { 0%{opacity:1;} 50%{opacity:.35;} 100%{opacity:1;} }

    /* ===== 섹션 타이틀 ===== */
    .section-title { display:flex; align-items:center; gap:10px; margin: var(--space-6) 0 var(--space-3) 0; }
    .section-title h3 { font-size: 1.15rem !important; font-weight: 800 !important; margin: 0 !important; }
    .section-title .icon-box {
        width: 34px; height: 34px; border-radius: 10px; background: var(--accent-blue-soft);
        display:flex; align-items:center; justify-content:center; font-size: 1rem;
    }

    /* ===== 입력창 / 버튼 ===== */
    .stTextInput input {
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-main) !important;
        font-weight: 600 !important;
        padding: 0.6rem 0.9rem !important;
    }
    .stTextInput input:focus { border-color: var(--accent-blue) !important; box-shadow: 0 0 0 3px var(--accent-blue-soft) !important; }
    .stDateInput input {
        background-color: var(--bg-card) !important; border-radius: var(--radius-sm) !important;
        border: 1.5px solid var(--border-color) !important; font-weight: 600 !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: var(--radius-sm) !important;
        background-color: var(--accent-blue) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1.1rem !important;
        transition: all 0.15s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stButton > button *, .stFormSubmitButton > button * { color: #FFFFFF !important; }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: var(--accent-blue-hover) !important;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md) !important;
    }

    /* ===== 탭: 필(pill) 스타일 ===== */
    div[data-baseweb="tab-list"] {
        gap: 6px !important; background: var(--bg-sub) !important;
        padding: 6px !important; border-radius: 14px !important;
        border: 1px solid var(--border-color) !important; width: fit-content;
    }
    button[data-baseweb="tab"] {
        border-radius: 10px !important; padding: 8px 20px !important;
        font-weight: 700 !important; background: transparent !important;
    }
    button[data-baseweb="tab"] p { color: var(--text-sub) !important; font-weight: 700 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background: var(--accent-blue) !important; box-shadow: var(--shadow-sm); }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF !important; }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

    /* ===== 카드형 컨테이너 (st.container(border=True)) ===== */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: var(--radius-md) !important;
        border-color: var(--border-color) !important;
        background: var(--bg-card) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ===== expander ===== */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        font-weight: 700 !important;
        padding: 0.85rem 1.1rem !important;
    }
    .streamlit-expanderContent {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-bottom-left-radius: var(--radius-sm) !important;
        border-bottom-right-radius: var(--radius-sm) !important;
        padding: 1.1rem !important;
    }

    /* ===== 상태 배너 ===== */
    .status-banner {
        display:flex; align-items:center; gap: 10px; flex-wrap: wrap;
        background: var(--accent-red-soft); border: 1px solid rgba(220,38,38,0.25);
        border-radius: var(--radius-md); padding: 14px 18px; margin-bottom: var(--space-3);
        font-weight: 700; font-size: 0.92rem;
    }
    .status-banner b { color: var(--accent-red) !important; }
    .caution-banner {
        background: var(--accent-orange-soft); border: 1px solid rgba(234,88,12,0.25);
        border-radius: var(--radius-sm); padding: 10px 16px; font-size: 0.82rem;
        color: var(--text-sub) !important; margin-bottom: var(--space-4);
    }

    /* ===== AI 매핑 결과 / 메트릭 스트립 ===== */
    .mapping-strip { display:flex; gap: 10px; flex-wrap: wrap; margin: var(--space-3) 0; }
    .mapping-pill {
        background: var(--bg-card); border: 1px solid var(--border-color);
        border-radius: 12px; padding: 8px 14px; box-shadow: var(--shadow-sm);
    }
    .mapping-pill .k { font-size: 0.68rem; font-weight: 700; color: var(--text-faint) !important; text-transform: uppercase; letter-spacing: .04em; }
    .mapping-pill .v { font-size: 0.95rem; font-weight: 800; color: var(--text-main) !important; margin-top: 2px; }
    .mapping-pill.accent .v { color: var(--accent-blue) !important; }

    .metric-strip { display:flex; gap:12px; flex-wrap:wrap; margin: var(--space-2) 0 var(--space-6) 0; }
    .metric-card {
        flex: 1 1 220px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        border-left: 4px solid var(--accent-blue);
        box-shadow: var(--shadow-sm);
    }


    .metric-card.danger { border-left-color: var(--accent-red); }
    .metric-card.warning { border-left-color: var(--accent-orange); }
    .metric-card.success { border-left-color: var(--accent-green); }
    .metric-label { font-size:.72rem; font-weight:800; color:var(--text-sub) !important; letter-spacing:.02em; margin-bottom:5px; }
    .metric-value { font-size:.92rem; font-weight:600; color:var(--text-main) !important; line-height:1.55; }
    .summary-action {
        display:flex; align-items:flex-start; gap:7px; margin:5px 0; line-height:1.45;
    }
    .summary-dot {
        flex:0 0 auto; font-weight:900; color:var(--accent-blue) !important; margin-top:1px;
    }

    .badge {
        display:inline-block; padding: 1px 9px; border-radius: 999px;
        font-size:.76rem; font-weight:800; margin: 1px 2px;
    }
    .badge-red { background:#FEE2E2; color:#B91C1C; }
    .badge-orange { background:#FFEDD5; color:#C2410C; }
    .badge-blue { background:#DBEAFE; color:#1D4ED8; }
    .badge-gray { background:#F1F5F9; color:#475569; }

    /* ===== 선박 카드 상태 칩 ===== */
    .vessel-chip { display:inline-block; padding: 3px 11px; border-radius: 999px; font-size:.75rem; font-weight:800; }
    .vessel-chip.in { background: var(--accent-green-soft); color: var(--accent-green) !important; }
    .vessel-chip.out { background: var(--accent-orange-soft); color: var(--accent-orange) !important; }

    hr { border-color: var(--border-color) !important; }
</style>
""",
    unsafe_allow_html=True,
)

PUBLIC_API_KEY = st.secrets.get('PUBLIC_API_KEY', '')
GEMINI_API_KEY = st.secrets.get('GEMINI_API_KEY', '')
AISSTREAM_API_KEY = st.secrets.get('AISSTREAM_API_KEY', '')

# ==========================================
# 🧩 디자인 헬퍼 함수 (신규)
# ==========================================


def section_title(icon, text):
  st.markdown(
      f"""
<div class="section-title">
    <div class="icon-box">{icon}</div>
    <h3>{text}</h3>
</div>
""",
      unsafe_allow_html=True,
  )


def render_mapping_strip(pairs):
  """AI 매핑 결과를 pill 형태로 렌더링. pairs: [(label, value, accent_bool), ...]"""
  html = '<div class="mapping-strip">'
  for label, value, accent in pairs:
    cls = 'mapping-pill accent' if accent else 'mapping-pill'
    html += (
        f'<div class="{cls}"><div class="k">{label}</div>'
        f'<div class="v">{value}</div></div>'
    )
  html += '</div>'
  st.markdown(html, unsafe_allow_html=True)


def _escape_html(text):
  return (text or '').replace('<', '&lt;').replace('>', '&gt;')


def highlight_badges(text):
  text = _escape_html(text)

  # 1) **볼드** 문법 변환
  text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

  # 2) 수치 / 단위 / 범위 하이라이트 (소수점 및 범위 기호 완벽 지원)
  # 예: 1.1~7.0%, 50m, 27℃, 0.86, 100~800m 등 모두 매칭
  number_unit_pattern = (
      r'(\d+(?:\.\d+)?(?:\s*[\~～\-]\s*\d+(?:\.\d+)?)?\s*(?:%LEL|vol%|m|미터|kts|℃|°C|%))'
  )
  text = re.sub(
      number_unit_pattern, r'<span class="badge badge-blue">\1</span>', text
  )

  # 3) 보호구 및 유해액체물질 등급
  text = re.sub(
      r'(Level\s?[A-D])', r'<span class="badge badge-red">\1</span>', text
  )
  text = re.sub(
      r'\b([XYZ]류)\b', r'<span class="badge badge-orange">\1</span>', text
  )
  text = re.sub(
      r'(절대\s*금지|즉각\s*퇴각|진입\s*금지|작업\s*중지)',
      r'<span class="badge badge-red">\1</span>',
      text,
  )

  return text


def parse_ai_summary(raw_text):
  """Gemini 응답에서 초동대응 핵심요약과 상세 섹션을 안정적으로 분리."""
  if not raw_text:
    return [], []

  expected_summary = [
      ('RISK', '사고상황 및 최우선 위험'),
      ('DISTANCE', '접근·통제 및 진입조건'),
      ('PPE', '인명구조·보호구 및 제독'),
      ('ACTION', '출동세력별 즉시 임무'),
  ]
  bullets = []

  # 1) 신규 고정 SUMMARY 블록을 우선 파싱
  summary_match = re.search(
      r'\[INITIAL_SUMMARY\](.*?)\[/INITIAL_SUMMARY\]',
      raw_text,
      re.DOTALL | re.IGNORECASE,
  )

  if summary_match:
    summary_text = summary_match.group(1)
    for key, label in expected_summary:
      m = re.search(
          rf'^\s*{key}\s*[:：]\s*(.+?)(?=^\s*(?:RISK|DISTANCE|PPE|ACTION)\s*[:：]|\Z)',
          summary_text,
          re.MULTILINE | re.DOTALL | re.IGNORECASE,
      )
      if m:
        value = ' '.join(m.group(1).strip().split())
        bullets.append((label, value))

  # 2) 기존 Markdown 형식도 계속 지원
  if not bullets:
    legacy_pattern = re.compile(
        r'^\s*[-*•]?\s*\*{0,2}'
        r'(사고상황 및 최우선 위험|접근·통제 및 진입조건|'
        r'인명구조·보호구 및 제독|출동세력별 즉시 임무|'
        r'사고물질 및 위험성 판단|통제 및 이격거리 지시|'
        r'출동세력 보호구 지정|현장 초동 행동 수칙)'
        r'\*{0,2}\s*[:：]\s*(.+)$',
        re.MULTILINE,
    )
    for m in legacy_pattern.finditer(raw_text):
      bullets.append((m.group(1).strip(), m.group(2).strip()))

  # 3) SUMMARY 블록을 제거한 뒤 상세 섹션 파싱
  clean_text = re.sub(
      r'\[INITIAL_SUMMARY\].*?\[/INITIAL_SUMMARY\]',
      '',
      raw_text,
      flags=re.DOTALL | re.IGNORECASE,
  )

  section_pattern = re.compile(r'^###\s+(.+?)\s*$', re.MULTILINE)
  matches = list(section_pattern.finditer(clean_text))

  sections = []
  for i, match in enumerate(matches):
    title = match.group(1).strip()
    if '초동대응 핵심요약' in title:
      continue

    start = match.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
    body = clean_text[start:end].strip()
    if body:
      sections.append((title, body))

  return bullets, sections


def render_ai_summary(raw_text):
  bullets, sections = parse_ai_summary(raw_text)

  if not bullets and not sections:
    st.markdown(raw_text)
    return

  st.markdown(
      '<div class="section-title" style="margin-top:0;">'
      '<div class="icon-box">🚨</div><h3>해경 초동작전 핵심요약</h3></div>',
      unsafe_allow_html=True,
  )

  if bullets:
    tone_cycle = ['danger', 'warning', 'primary', 'success']
    icon_cycle = ['🚨', '🧭', '🛟', '🚤']
    html = '<div class="metric-strip">'
    for i, (label, value) in enumerate(bullets):
      tone = tone_cycle[i % len(tone_cycle)]
      icon = icon_cycle[i % len(icon_cycle)]
      items = [item.strip() for item in value.split('|') if item.strip()]
      items_html = ''.join(
          f'<div class="summary-action">'
          f'<span class="summary-dot">•</span>'
          f'<span>{highlight_badges(item)}</span>'
          f'</div>'
          for item in items
      )
      html += (
          f'<div class="metric-card {tone}">'
          f'<div class="metric-label">{icon} {label}</div>'
          f'<div class="metric-value">{items_html}</div>'
          '</div>'
      )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
  else:
    st.warning(
        '초동대응 핵심요약을 구조화하지 못했습니다. '
        '아래 상세 대응지침을 우선 확인하십시오.'
    )

  for idx, (title, body) in enumerate(sections):
    with st.expander(title, expanded=(idx == 0)):
      st.markdown(body)


# ==========================================
# 1. 🖼️ PDF 인덱스 맵 생성
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


class VesselAPIError(RuntimeError):
  """선박 공공 API의 통신·인증·응답 오류."""


def _api_result_error(root):
  """공공데이터 XML 응답의 오류를 사용자 표시용 문구로 변환."""
  result_code = (root.findtext('.//resultCode') or '').strip()
  result_msg = (root.findtext('.//resultMsg') or '').strip()
  return_reason = (root.findtext('.//returnReasonCode') or '').strip()
  return_auth_msg = (root.findtext('.//returnAuthMsg') or '').strip()
  error_msg = (root.findtext('.//errMsg') or '').strip()

  if result_code and result_code not in {'0', '00'}:
    return f'{result_code}: {result_msg or "응답 오류"}'
  if return_reason and return_reason not in {'0', '00'}:
    return f'{return_reason}: {return_auth_msg or error_msg or "응답 오류"}'
  return ''


@st.cache_data(ttl=300)
def fetch_vessel_schedule_api(port_code, de_gb, sde_str, ede_str):
  if not PUBLIC_API_KEY:
    raise VesselAPIError('공공데이터 API 키가 설정되지 않았습니다.')
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
      if res.status_code != 200:
        raise VesselAPIError(f'HTTP {res.status_code} 응답 오류')
      if not res.content:
        raise VesselAPIError('빈 응답이 수신되었습니다.')

      root = ET.fromstring(res.content)
      result_error = _api_result_error(root)
      if result_error:
        raise VesselAPIError(result_error)

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

          detail_nodes = item.findall('.//detail') or item.findall(
              'details/detail'
          )
          expected_etrynd = '입항' if str(de_gb).upper() == 'I' else '출항'
          detail_node = next(
              (
                  node
                  for node in detail_nodes
                  if expected_etrynd
                  in (node.findtext('etryndNm') or '').strip()
              ),
              detail_nodes[0] if detail_nodes else None,
          )
          reqst_se_nm = etrynd_nm = etrypt_dt = tkoff_dt = ibobprt_nm = (
              laidup_fclty_nm
          ) = ldadng_frght_cl_cd = ldadng_ton = trnpdt_ton = landng_frght_ton = (
              ld_frght_ton
          ) = grtg = satmnt_entrps_nm = crew_co = mr_num = tkoff_prrrn_dt = (
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
            mr_num = (detail_node.findtext('mrNum') or '-').strip()
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
              'mr_num': mr_num,
              'tkoff_prrrn_dt': tkoff_prrrn_dt,
              'dstn_etrypt_dt': dstn_etrypt_dt,
              'reqst_se_nm': reqst_se_nm,
          })

      if len(vessels) >= total_cnt or len(items) < rows_per_page:
        break
      page += 1
    except VesselAPIError:
      raise
    except Exception as e:
      print(f'선박 API 예외 ({port_code}/{de_gb}): {e}')
      raise VesselAPIError(f'선박 입출항 API 처리 오류: {e}') from e

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


@st.cache_data(ttl=300)
def fetch_vessel_spec_list_api(query_str, max_results=50):
  if not query_str:
    return []
  if not PUBLIC_API_KEY:
    raise VesselAPIError('공공데이터 API 키가 설정되지 않았습니다.')

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
        if res.status_code != 200:
          raise VesselAPIError(f'HTTP {res.status_code} 응답 오류')
        if not res.content:
          raise VesselAPIError('빈 응답이 수신되었습니다.')

        root = ET.fromstring(res.content)
        result_error = _api_result_error(root)
        if result_error:
          raise VesselAPIError(result_error)

        items = root.findall('.//item') or root.findall('body/items/item')
        if not items:
          break

        for item in items:
          kor_name = (item.findtext('vsslKorNm') or '').strip() or '-'
          eng_name = (item.findtext('vsslEngNm') or '').strip() or '-'

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
              # 일부 운영 응답에 존재할 경우에만 활용한다. 공식 가이드의
              # 기본 응답 필드는 아니므로 없으면 '-'로 유지한다.
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
          dedupe_key = (
              spec['vsslNo'],
              spec['imoNo'],
              spec['clsgn'].upper(),
              spec['vsslKorNm'].upper(),
              spec['vsslEngNm'].upper(),
          )
          if not any(r['_dedupe_key'] == dedupe_key for r in results):
            spec['_dedupe_key'] = dedupe_key
            results.append(spec)
            if len(results) >= max_results:
              break

        if len(items) < 50 or len(results) >= max_results:
          break
        page += 1
      except VesselAPIError:
        raise
      except Exception as e:
        print(f'선박제원 다중목록 조회 에러: {e}')
        raise VesselAPIError(f'선박제원 API 처리 오류: {e}') from e

    if len(results) >= max_results:
      break

  for spec in results:
    spec.pop('_dedupe_key', None)

  normalized_query = _normalize_vessel_text(clean_q)
  results.sort(
      key=lambda spec: _vessel_spec_match_score(spec, normalized_query)
  )
  return results


def _normalize_vessel_text(value):
  return re.sub(r'[^0-9A-Z가-힣]', '', str(value or '').strip().upper())


def _vessel_spec_match_score(spec, normalized_query):
  if not normalized_query:
    return 99

  identifiers = [
      _normalize_vessel_text(spec.get('clsgn')),
      _normalize_vessel_text(spec.get('vsslNo')),
      _normalize_vessel_text(spec.get('imoNo')),
      _normalize_vessel_text(spec.get('mmsiNo')),
  ]
  names = [
      _normalize_vessel_text(spec.get('vsslKorNm')),
      _normalize_vessel_text(spec.get('vsslEngNm')),
  ]

  if normalized_query in [value for value in identifiers if value]:
    return 0
  if normalized_query in [value for value in names if value]:
    return 1
  if any(
      normalized_query in value or value in normalized_query
      for value in names
      if value
  ):
    return 10
  return 99


def fetch_vessel_spec_api(clsgn, vssl_nm):
  target_clsgn = _normalize_vessel_text(clsgn)
  target_name = _normalize_vessel_text(vssl_nm)
  queries = []
  for value in (clsgn, vssl_nm):
    if value and str(value).strip() not in {'-', '없음'}:
      if value not in queries:
        queries.append(value)

  specs = []
  seen = set()
  for query in queries:
    for spec in fetch_vessel_spec_list_api(query):
      key = (
          spec.get('vsslNo'),
          spec.get('imoNo'),
          spec.get('clsgn'),
          spec.get('vsslKorNm'),
          spec.get('vsslEngNm'),
      )
      if key not in seen:
        seen.add(key)
        specs.append(spec)

  exact_callsign = [
      spec
      for spec in specs
      if target_clsgn
      and target_clsgn == _normalize_vessel_text(spec.get('clsgn'))
  ]
  if exact_callsign:
    return exact_callsign[0]

  exact_name = [
      spec
      for spec in specs
      if target_name
      and target_name
      in {
          _normalize_vessel_text(spec.get('vsslKorNm')),
          _normalize_vessel_text(spec.get('vsslEngNm')),
      }
  ]
  return exact_name[0] if exact_name else None


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
        첨부된 [HNS 정보집 원본 스캔 이미지]와 [해양사고 대응 가이드 원본 스캔 이미지들], 그리고 수집된 다중 데이터(공공 API, MSDS, RAG 텍스트)를 철저히 교차 분석하여, 현장 세력(OSC, 경비함정, 방제함정, 특수구조대 등)에 즉각 하달할 수 있는 가장 전문적이고 완벽한 **비상대응 지시서**를 작성하세요.

        {accident_info}
        {hns_img_prompt}
        {rag_img_prompt}

        [해경 HNS 해양사고 대응가이드 RAG 검색 텍스트]
        {rag_text}

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

        [해경 HNS 현장대응 작성 원칙]
        1. **정보 우선순위 및 충돌 처리**:
           - 사용자가 입력한 사고상황은 현장 확인정보로 취급하되, 입력되지 않은 인명피해·누출량·누출원·화재 여부·선박 손상·기상·풍향·조류·조석은 추정하지 말고 `현장 확인 필요`라고 명시할 것.
           - 화학물질의 성상·유해성·MSDS 수치는 해양수산부 위험물정보, 화학물질안전원, 안전보건공단 데이터를 우선할 것.
           - 이격거리·진입조건·보호구·인명구조·방제·소화·제독 등 현장 전술은 해경 HNS 정보집 및 대응가이드 RAG 지침을 최우선으로 적용할 것.
           - 출처 간 수치나 지침이 충돌하면 현장대원의 안전을 확보하는 보수적 기준을 적용하고 상세 본문에 충돌 사실을 표시할 것.

        2. **인명구조 우선 및 무리한 진입 금지**:
           - 인명피해·실종자·고립자·오염환자의 수와 위치를 가장 먼저 확인하도록 지시할 것.
           - 구조대의 진입은 물질식별, 풍향·조류, 산소·LEL·유해가스 측정, 적정 보호구, 2인 1조, 예비구조조 및 제독선 확보를 전제로 작성할 것.
           - 일반 함정·파출소 인력이 보호구와 탐지 없이 오염구역에 진입하도록 지시하지 말 것.
           - 구조한 오염환자는 안전구역으로 바로 이송하지 말고 제독 후 의료진에게 인계하도록 명시할 것.

        3. **해상거동 기반 방제전략**:
           - 물질의 부유·증발·용해·침강·수반응성 여부를 먼저 판정한 뒤 적합한 방제수단만 제시할 것.
           - 오일펜스·회수기·흡착재·소화포·분무주수·중화제 사용을 모든 물질에 일률적으로 지시하지 말 것.
           - 오일펜스는 수면 부유성과 자재 적합성이 확인되고, 화재·폭발·증기 노출 위험이 통제된 경우에만 전장하도록 작성할 것.
           - 용해성·침강성·고휘발성·물반응성 물질은 부적합한 수면 회수수단을 명확히 `사용 금지` 또는 `효과 제한`으로 표시할 것.
           - 누출원 차단은 선박 안전과 작업자 진입조건이 확보된 경우에만 실시하고, 회수물·오염자재의 밀폐보관 및 폐기물 인계까지 포함할 것.

        4. **출동세력별 임무 분리**:
           - OSC/상황실, 경비함정, 구조대·특수구조대, 파출소, 방제정·방제세력의 임무를 서로 섞지 말고 별도로 작성할 것.
           - 경비함정은 해상 접근·통항 통제, 안전측 감시, 구조지원 및 주변선박 경고를 중심으로 작성할 것.
           - 구조대는 요구조자 탐색·구조, 진입조/예비조 운영, 연속 계측 및 제독 인계를 중심으로 작성할 것.
           - 파출소는 부두·육상 출입통제, 승선원·실종자 확인, 적하목록·MSDS·탱크/배관 정보 확보, 주민·관계기관 연락을 중심으로 작성할 것.
           - 방제정·방제세력은 물질 거동에 맞는 확산방지·회수, 누출원 통제 지원, 회수물 보관 및 오염장비 제독을 중심으로 작성할 것.
           - 해당 세력이 수행하기 부적절하거나 위험한 작업은 `대기·통제·지원` 임무로 제한할 것.

        5. **현장통제·탐지·LEL 표기·재평가**:
           - 풍상측과 조류 상류측을 함께 고려하여 접근방향을 제시하고, Hot/Warm/Cold Zone 또는 이에 준하는 위험구역을 구분할 것.
           - 이격거리와 대피거리는 유출·대량유출·화재·탱크화재 등 적용 조건을 구분하고, 근거자료가 없는 수치를 생성하지 말 것.
           - 산소, LEL, PID/VOC 또는 물질별 탐지값과 풍향·조류 변화에 따른 작업중지 기준 및 통제구역 재설정 지시를 포함할 것.
           - 물질 고유의 폭발범위는 반드시 `LEL [하한값] vol% ~ UEL [상한값] vol%`처럼 공기 중 체적농도와 LEL·UEL을 함께 표시할 것. 확인된 물질별 값만 사용하고, UEL 수치를 허용기준이나 철수기준으로 사용하지 말 것.
           - 가연성가스 검지기 측정값은 반드시 `10%LEL`, `20%LEL`처럼 `%LEL` 단위로 작성할 것. `LEL 10%`, `10%`, `20%`처럼 체적농도와 혼동되는 표현은 절대 사용하지 말 것.
           - 기관 SOP·현장지휘관 지시·측정기 경보값 또는 근거자료가 더 엄격하면 그 기준을 우선할 것. 별도 근거가 없을 때는 `10%LEL 이상: 일반작업 즉시 중지·비필수 인원 철수·점화원 차단`, `20%LEL 이상 또는 농도 급상승: Hot Zone 전원 즉시 철수·통제구역 확대`로 단계화할 것.
           - LEL 검지값과 독성 노출기준은 서로 대체하지 말 것. TWA·STEL·IDLH는 ppm 단위로 별도 평가하고, 독성기준이 더 먼저 도달하면 독성기준을 우선 적용할 것. LEL 검지값이 낮거나 미검출이어도 호흡 안전으로 판단하지 말 것.
           - ppm과 %LEL의 환산은 해당 물질의 신뢰 가능한 LEL 값과 검지기 교정조건이 확인된 경우에만 제시하고, 근거 없이 계산하지 말 것.
           - 보호구 하향은 현장 측정과 OSC 승인 전에는 지시하지 말 것.

        6. **상황전파·지원요청**:
           - 사고유형에 따라 VTS·소방·화학재난합동방재센터·화학물질안전원·해양환경공단·항만공사·지자체·의료기관 등 필요한 기관만 선별하여 지원요청 사항을 작성할 것.
           - 기관명이나 장비를 단순 나열하지 말고 무엇을 요청할지 명확히 작성할 것.

        7. **초동작전 핵심요약 — 절대 생략 금지**:
           - 모든 응답은 아래 `[INITIAL_SUMMARY]` 블록으로 시작하며 RISK, DISTANCE, PPE, ACTION 4개 키를 정확히 한 번씩 작성할 것.
           - 각 키의 행동지침은 ` | `로 구분하고, 현장 지휘자가 5초 안에 읽을 수 있도록 짧은 명령형 문장으로 작성할 것.
           - ACTION에는 반드시 `함정:`, `구조대:`, `파출소:`, `방제정:` 4개 세력별 첫 임무를 각각 작성할 것.
           - 확인되지 않은 정보나 수치는 임의로 생성하지 말고 `현장 확인 필요` 또는 `자료 확인 필요`라고 표시할 것.

        8. **상세 출력 형식**:
           - INITIAL_SUMMARY 블록 뒤에 아래 상세 섹션 1~6을 순서대로 모두 작성할 것.
           - 각 상세 섹션은 실행지침 중심의 글머리표로 작성하고, 같은 내용을 여러 섹션에 반복하지 말 것.
           - 공문서 서식(수신·발신·결재선 등), 면책성 문구, 불필요한 서론과 결론은 생성하지 말 것.

        [INITIAL_SUMMARY]
        RISK: [물질·사고유형·인명상황] | [화재·독성·반응성·해상거동 중 최우선 위험] | [즉시 확인할 미확인 핵심정보]
        DISTANCE: [풍상·조류 상류측 접근방향] | [조건별 초기 통제·대피거리] | [가스검지기 값은 10%LEL 형식으로 표시한 진입 금지·작업 중지 기준]
        PPE: [요구조자 유무와 구조 우선순위] | [진입조 보호구·탐지·예비조 조건] | [제독 후 의료 인계]
        ACTION: 함정: [통항통제·안전측 감시·구조지원 첫 임무] | 구조대: [탐지·진입·인명구조 첫 임무] | 파출소: [육상통제·인원확인·자료확보 첫 임무] | 방제정: [물질거동에 맞는 확산방지·회수 첫 임무]
        [/INITIAL_SUMMARY]

        ---
        ### 1. 🚨 사고상황 및 즉시 확인사항
        ### 2. 🛟 현장통제·접근조건·인명구조 및 제독
        ### 3. 🚤 출동세력별 임무
        ### 4. 🧪 물질 거동별 방제·누출원 통제·회수전략
        ### 5. 🔥 화재·폭발·금지행동 및 노출자 응급조치
        ### 6. 📡 상황전파·지원요청·모니터링 및 재평가
        """

    contents_input = [prompt_text]
    if hns_pil_image:
      contents_input.append(hns_pil_image)

    for r_img in rag_images:
      contents_input.append(r_img['pil_img'])

    for model_id in [
        'gemini-3.8-flash',
        'gemini-3.7-flash',
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
# ⚓ AISStream WebSocket 실시간 위치 수신
# ==========================================


def fetch_aisstream_vessel_position(
    vssl_nm='', clsgn='', mmsi_no='', timeout_sec=4
):
  if not AISSTREAM_API_KEY:
    return None

  subscribe_message = {
      'APIKey': AISSTREAM_API_KEY,
      'BoundingBoxes': [[[34.0, 124.0], [38.5, 128.5]]],
  }

  target_nm = _normalize_vessel_text(vssl_nm)
  target_clsgn = _normalize_vessel_text(clsgn)
  target_mmsi = re.sub(r'\D', '', str(mmsi_no or ''))
  if target_mmsi:
    subscribe_message['FiltersShipMMSI'] = [target_mmsi]

  if not any((target_nm, target_clsgn, target_mmsi)):
    return None

  ws = None
  try:
    ws = websocket.create_connection(
        'wss://stream.aisstream.io/v0/stream', timeout=timeout_sec
    )
    ws.send(json.dumps(subscribe_message))
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
      remaining = max(0.1, deadline - time.monotonic())
      ws.settimeout(remaining)
      message = ws.recv()
      if not message:
        continue

      data = json.loads(message)
      if data.get('MessageType') != 'PositionReport':
        continue

      metadata = data.get('MetaData', {})
      recv_ship_name = _normalize_vessel_text(metadata.get('ShipName'))
      recv_clsgn = _normalize_vessel_text(metadata.get('CallSign'))
      recv_mmsi = re.sub(r'\D', '', str(metadata.get('MMSI', '')))

      matched_by = ''
      if target_mmsi and recv_mmsi and target_mmsi == recv_mmsi:
        matched_by = 'MMSI 완전일치'
      elif target_clsgn and recv_clsgn and target_clsgn == recv_clsgn:
        matched_by = '호출부호 완전일치'
      elif target_nm and recv_ship_name and target_nm == recv_ship_name:
        matched_by = '선박명 완전일치'

      if not matched_by:
        continue

      pos = data.get('Message', {}).get('PositionReport', {})
      return {
          'lat': pos.get('Latitude'),
          'lon': pos.get('Longitude'),
          'sog': pos.get('Sog', 0.0),
          'cog': pos.get('Cog', 0.0),
          'time_utc': metadata.get('time_utc', ''),
          'ship_name': metadata.get('ShipName', '-'),
          'mmsi': metadata.get('MMSI', '-'),
          'matched_by': matched_by,
      }
  except websocket.WebSocketTimeoutException:
    return None
  except Exception as e:
    print(f'AISStream 연결 실패: {e}')
  finally:
    if ws is not None:
      try:
        ws.close()
      except Exception:
        pass

  return None


# ==========================================
# 🚢 모달 팝업: 선박 제원 및 실시간 위치
# ==========================================


def _valid_imo_for_link(value):
  digits = re.sub(r'\D', '', str(value or ''))
  return digits if len(digits) == 7 else ''


@st.dialog('🚢 선박 제원 및 실시간 AIS 위치 정보', width='large')
def show_vessel_detail_dialog(v):
  st.markdown(
      f"### ⚓ {v['vssl_nm']} <code>{v['clsgn']}</code>", unsafe_allow_html=True
  )
  st.divider()

  col_left, col_right = st.columns([1, 1])

  with col_left:
    with st.container(border=True):
      st.markdown('#### 📐 선박 제원 스펙 정보')
      spec_info = v.get('spec_info')
      spec_error = ''
      if not spec_info:
        try:
          spec_info = fetch_vessel_spec_api(v['clsgn'], v['vssl_nm'])
        except VesselAPIError as e:
          spec_error = str(e)

      if spec_error:
        st.error(f'선박제원 API 조회 실패: {spec_error}')

      if spec_info:
        kor_nm = spec_info.get('vsslKorNm', '-')
        eng_nm = spec_info.get('vsslEngNm', '-')

        if kor_nm != '-' and eng_nm != '-':
          name_str = f'{kor_nm} / {eng_nm}'
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
            f"- **선종 / 국적:** {spec_info['vsslKnd']} /"
            f" {spec_info['vsslNlty']}"
        )
        st.write(f"- **총톤수(GRT):** {spec_info['grtg']} 톤")
        st.write(
            f"- **선박 길이×너비:** {spec_info['vsslTotLt']}m ×"
            f' {spec_info["shdth"]}m'
        )
        st.write(
            f"- **흘수 / 깊이:** {spec_info['vsslDrft']}m /"
            f" {spec_info['vsslDp']}m"
        )
        st.write(
            f"- **운항형태 / 나용선:** {spec_info['nvgShapNm']} /"
            f' {spec_info["brbtSeNm"]}'
        )
        st.write(f"- **건조일시:** {spec_info['vsslCnstrDt']}")
      elif not spec_error:
        st.warning('💡 해수부 API에 등록된 선박제원 스펙이 없습니다.')

  with col_right:
    with st.container(border=True):
      st.markdown('#### 🛰️ 실시간 AIS 위치 및 지도')
      imo_number = spec_info.get('imoNo', '-') if spec_info else '-'
      mmsi_number = spec_info.get('mmsiNo', '-') if spec_info else '-'
      ais_vessel_name = v['vssl_nm']
      if spec_info:
        ais_vessel_name = (
            spec_info.get('vsslEngNm')
            if spec_info.get('vsslEngNm') not in {'', '-', None}
            else spec_info.get('vsslKorNm', v['vssl_nm'])
        )

      with st.spinner('AISStream 신호 탐색 중...'):
        ais_pos = fetch_aisstream_vessel_position(
            vssl_nm=ais_vessel_name,
            clsgn=v['clsgn'],
            mmsi_no=mmsi_number,
            timeout_sec=3,
        )

      if (
          ais_pos
          and ais_pos.get('lat') is not None
          and ais_pos.get('lon') is not None
      ):
        lat, lon = ais_pos['lat'], ais_pos['lon']
        sog, cog = ais_pos['sog'], ais_pos['cog']
        time_utc = ais_pos['time_utc']

        st.success(
            f'📍 **위치 수신 성공** (위도: `{lat:.4f}`, 경도: `{lon:.4f}`)'
        )
        st.write(f'- **속력(SOG):** {sog} kts ｜ **침로(COG):** {cog}°')
        st.write(f'- **수신시각(UTC):** {time_utc}')
        st.write(
            f"- **식별근거:** {ais_pos.get('matched_by', '-')} ｜ "
            f"**AIS MMSI:** `{ais_pos.get('mmsi', '-')}`"
        )

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
            '💡 제한시간 내 대상 선박과 식별값이 완전히 일치하는 AIS 신호를 '
            '수신하지 못했습니다.'
        )

        facility_nm = v.get('laidup_fclty_nm', '-')
        st.write(f'- **PORT-MIS 신고 계선장소:** `{facility_nm}`')

        valid_imo = _valid_imo_for_link(imo_number)
        if valid_imo:
          mt_link = f'https://www.marinetraffic.com/en/ais/details/ships/imo:{valid_imo}'
          st.markdown(
              f"🔗 **[MarineTraffic에서 `{v['vssl_nm']}` (IMO: {valid_imo})"
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
  chip_cls = 'in' if '입' in str(v['etrynd_nm']) else 'out'
  expander_label = (
      f"🚢 {v['vssl_nm']}  ·  {v['clsgn']}  ·  {v['vssl_knd_nm']}  ·"
      f" {v['laidup_fclty_nm']}"
  )

  with st.expander(expander_label, expanded=False):
    st.markdown(
        f'<span class="vessel-chip {chip_cls}">{v["etrynd_nm"]}'
        f' · {v["reqst_se_nm"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
      st.markdown('**⚓ 선박 및 국적 정보**')
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
      st.markdown('**🕐 운항 및 관제 일시**')
      st.write(f"- **입출항구분명:** {v['etrynd_nm']} ({v['reqst_se_nm']})")
      st.write(f"- **입항목적명:** {v['etrypt_purps_nm']}")
      st.write(f"- **입항일시:** {v['etrypt_dt']}")
      st.write(f"- **출항일시:** {v['tkoff_dt']}")
      st.write(f"- **출항예정일시:** {v['tkoff_prrrn_dt']}")
      st.write(f"- **목적지입항예정일시:** {v['dstn_etrypt_dt']}")
      st.write(f"- **계선시설명:** {v['laidup_fclty_nm']}")
      st.write(f"- **신고업체명:** {v['satmnt_entrps_nm']}")

    with col3:
      st.markdown('**🧭 항로 및 신고화물 정보**')
      st.write(f"- **전출항지항구명:** {v['prvs_dpmprt_prt_nm']}")
      st.write(f"- **차출항지항구명:** {v['nxlnpt_prt_nm']}")
      st.write(f"- **목적지항구명:** {v['dstn_prt_nm']}")
      st.write(
          f"- **PORT-MIS 화물명세 코드:** {v['ldadng_frght_cl_cd']}"
      )
      st.write(f"- **적하목록관리번호:** {v.get('mr_num', '-')}")
      st.write(f"- **신고 적재톤수:** {v['ldadng_ton']} 톤")
      st.write(f"- **환적톤수:** {v['trnpdt_ton']} 톤")
      st.write(f"- **양하화물톤:** {v['landng_frght_ton']} 톤")
      st.write(f"- **적하화물톤:** {v['ld_frght_ton']} 톤")
      st.caption(
          '※ 화물명세 코드는 실제 HNS 물질명·UN번호를 뜻하지 않습니다. '
          '위험물 적하목록·MSDS·적하목록관리번호를 별도로 확인하십시오.'
      )

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

  if start_date > end_date:
    st.error('조회 시작일은 종료일보다 늦을 수 없습니다.')
    return

  if query_trigger:
    fetch_vessel_schedule_api.clear()
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

  try:
    with st.spinner(f'{port_name} 입항 및 출항 신고정보 통합 수집 중...'):
      in_vessels = fetch_vessel_schedule_api(port_code, 'I', sde_str, ede_str)
      out_vessels = fetch_vessel_schedule_api(port_code, 'O', sde_str, ede_str)
      vessels = in_vessels + out_vessels
  except VesselAPIError as e:
    st.error(
        f'PORT-MIS 입출항 신고정보를 조회하지 못했습니다: {e} '
        '잠시 후 다시 조회하십시오.'
    )
    return

  if not vessels:
    st.warning(
        f'💡 해당 기간({sde_fmt} ~ {ede_fmt}) {port_name} 입출항 신고 선박'
        ' 정보가 없습니다.'
    )
    return

  unique_vessel_keys = {
      (
          f"CALL:{str(v.get('clsgn', '')).strip().upper()}"
          if str(v.get('clsgn', '')).strip() not in {'', '-'}
          else f"NAME:{_normalize_vessel_text(v.get('vssl_nm', ''))}"
      )
      for v in vessels
  }
  fetched_at_kst = (
      datetime.now(timezone.utc) + timedelta(hours=9)
  ).strftime('%Y-%m-%d %H:%M:%S')

  st.success(
      f'✅ 입출항 신고 총 **{len(vessels)}건**이 수집되었습니다. '
      f'(입항 {len(in_vessels)}건 / 출항 {len(out_vessels)}건 / '
      f'고유 선박 {len(unique_vessel_keys)}척)'
  )
  st.caption(
      f'PORT-MIS 입출항 신고자료 ｜ 최근 조회 {fetched_at_kst} KST ｜ '
      '조회 결과는 5분간 캐시됩니다.'
  )

  ALL_VIEW_OPTION = f'📋 전체 입출항 신고 목록 보기 (총 {len(vessels)}건)'
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
# 5. 메인 화면 구성 (Hero Section)
# ==========================================
_logo_html = (
    f'<img src="data:image/png;base64,{kcg_logo_b64}" style="width:100%;height:100%;object-fit:contain;" alt="해양경찰 로고" />'
    if kcg_logo_b64
    else '🚢'
)

st.markdown(
    f"""
<div class="hero-container">
    <div class="hero-top">
        <div class="hero-left">
            <div class="hero-logo-badge">{_logo_html}</div>
            <div>
                <div class="main-header">평택해양경찰서 HNS AI 대응 시스템</div>
                <div class="sub-header">공공 API(해양수산부·화학물질안전원·안전보건공단) + 해경 DB(HNS 정보집·대응가이드) + Gemini AI</div>
            </div>
        </div>
        <div class="live-chip"><span class="live-dot"></span>SYSTEM ONLINE</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 🔥 상단 통합 검색 시스템
# ==========================================
section_title('🔎', '통합 검색 시스템')

search_tab_chem, search_tab_vssl = st.tabs([
    '🧪 HNS 물질 및 사고상황 AI 검색',
    '🚢 선박 제원 및 위치 검색',
])

# ------------------------------------------
# [탭 1]: 화학물질 및 사고상황 AI 검색
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
      pill_pairs = [
          ('물질명', mapped_ko, True),
          ('영문명', mapped_eng, False),
          ('UN NO', mapped_unno, False),
          ('CAS NO', mapped_cas, False),
      ]
      render_mapping_strip(pill_pairs)
      if accident_ctx:
        st.markdown(
            f'<div class="caution-banner">🚨 <b>사고상황 식별:</b>'
            f' {accident_ctx}</div>',
            unsafe_allow_html=True,
        )
    with c2:
      st.markdown('<br>', unsafe_allow_html=True)
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
      try:
        vssl_list = fetch_vessel_spec_list_api(clean_query, max_results=50)
        st.session_state['vssl_search_results'] = vssl_list
        st.session_state['vssl_search_error'] = ''
      except VesselAPIError as e:
        st.session_state['vssl_search_results'] = []
        st.session_state['vssl_search_error'] = str(e)
      st.session_state['vssl_search_keyword'] = clean_query

  if 'vssl_search_results' in st.session_state:
    results = st.session_state['vssl_search_results']
    kw = st.session_state.get('vssl_search_keyword', '')
    search_error = st.session_state.get('vssl_search_error', '')

    if search_error:
      st.error(
          f'선박제원 API를 조회하지 못했습니다: {search_error} '
          '잠시 후 다시 검색하십시오.'
      )
    elif not results:
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
          f"🚢 [{s['displayName']}] 호출부호: {s['clsgn']} ｜ IMO:"
          f" {s.get('imoNo', '-')} ｜ 국적: {s['vsslNlty']} ｜ 선종:"
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
                'vssl_nm': (
                    selected_spec['vsslEngNm']
                    if selected_spec['vsslEngNm'] != '-'
                    else selected_spec['vsslKorNm']
                ),
                'clsgn': selected_spec.get('clsgn', kw).upper(),
                'laidup_fclty_nm': '선박 직접 검색 결과',
                'spec_info': selected_spec,
            }
            show_vessel_detail_dialog(dummy_vessel_obj)

# ------------------------------------------
# ⚡ AI 대응 가이드 출력 영역
# ------------------------------------------
if 'active_chem' in st.session_state:
  st.divider()
  chem = st.session_state['active_chem']
  unno = st.session_state['active_unno']
  cas = st.session_state.get('active_cas', '-')
  ship_info = st.session_state['active_ship']
  accident_ctx = st.session_state.get('active_accident_context', '')

  status_line = f'대상: {ship_info}  ·  물질: {chem} (UN {unno} / CAS {cas})'
  if accident_ctx:
    status_line += f'  ·  상황: {accident_ctx}'
  st.markdown(
      f'<div class="status-banner">⚡ <b>해경 HNS 초동작전 가이드</b>'
      f' &nbsp;|&nbsp; {status_line}</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="caution-banner">⚠️ <b>할루시네이션 주의</b> — 본 대응'
      ' 가이드는 공공 API 3종 및 해경 HNS 정보집·대응가이드를 통합한 Gemini'
      ' RAG 모델로 AI 환각 현상을 최소화했습니다. 단, 현장 상황은 가이드와'
      ' 다를 수 있으므로 반드시 재확인하시기 바랍니다.</div>',
      unsafe_allow_html=True,
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
          f' EmS {em_s} 인명구조 현장통제 물질탐지 진입조건 보호구 제독'
          f' 누출원봉쇄 세력별임무 해상거동별 방제 회수 안전거리 이격지침'
          f' 상황전파 지원요청 사후관리 {situation_keyword}'
      )

      print(f'🔍 [RAG 고도화 쿼리]: {rag_search_query}')

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

  render_ai_summary(st.session_state['active_summary'])

  # ------------------------------------------
  # 📚 활용 원본 자료 확인 탭
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
# ⚓ 항만 선박 입출항 신고현황
# ------------------------------------------
section_title('⚓', '항만별 PORT-MIS 입출항 신고현황 (5분 갱신)')

tab_pt, tab_ds = st.tabs(['🚢 평택항 입출항 선박', '🚢 대산항 입출항 선박'])

with tab_pt:
  render_combined_port_tab_content('평택항', '031')

with tab_ds:
  render_combined_port_tab_content('대산항', '300')
