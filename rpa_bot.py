from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import re

# ==========================================
# ⚙️ 계정 설정 영역
# ==========================================
PORTMIS_ID = os.getenv("PORTMIS_ID", "")
PORTMIS_PW = os.getenv("PORTMIS_PW", "")

def run_real_rpa_crawler():
    print("🤖 [정밀 페이지네이션 전수 탐색 RPA] 포트미스 자동화 봇 가동...")

    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')          # 최신 Headless 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # 크롬 브라우저 바이너리 지정
    chrome_binaries = ["/usr/bin/chromium-browser", "/usr/bin/google-chrome", "/usr/bin/chromium"]
    for bin_path in chrome_binaries:
        if os.path.exists(bin_path):
            options.binary_location = bin_path
            break

    # OCI/Linux ARM64 환경용 ChromeDriver Service 지정
    chromedriver_paths = ["/usr/bin/chromium-chromedriver", "/usr/bin/chromedriver"]
    chromedriver_bin = None
    for driver_path in chromedriver_paths:
        if os.path.exists(driver_path):
            chromedriver_bin = driver_path
            break

    if chromedriver_bin:
        service = Service(executable_path=chromedriver_bin)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    wait = WebDriverWait(driver, 20)

    # 📌 수집 대상 항구 정의 (평택항: 031, 대산항: 300)
    ports_to_collect = [
        {"name": "평택항", "code": "031", "filename": "hns_pyeongtaek_report.csv"},
        {"name": "대산항", "code": "300", "filename": "hns_daesan_report.csv"}
    ]

    try:
        # ------------------------------------------
        # 1단계 ~ 5단계: 로그인 및 페이지 진입
        # ------------------------------------------
        print("👉 [1단계] 인트로 페이지 접속 중...")
        driver.get("https://new.portmis.go.kr/portmis/websquare/websquare.jsp?w2xPath=/portmis/w2/main/intro.xml")
        time.sleep(4)

        print("👉 [2단계] 로그인 팝업 호출...")
        login_trigger = wait.until(EC.element_to_be_clickable((By.ID, "mf_btnLogin")))
        login_trigger.click()
        time.sleep(4)

        print("👉 [3단계] 아이디 및 비밀번호 입력...")
        driver.execute_script(f"""
            var pwInputs = document.querySelectorAll('input[type="password"]');
            if (pwInputs.length > 0) {{
                var targetPw = pwInputs[pwInputs.length - 1];
                targetPw.value = '{PORTMIS_PW}';
                targetPw.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}

            var textInputs = document.querySelectorAll('input[type="text"]');
            for (var i = textInputs.length - 1; i >= 0; i--) {{
                if (textInputs[i].style.display != 'none' && textInputs[i].offsetParent != null) {{
                    textInputs[i].value = '{PORTMIS_ID}';
                    textInputs[i].dispatchEvent(new Event('input', {{ bubbles: true }}));
                    break;
                }}
            }}
        """)
        time.sleep(2)

        print("👉 [4단계] 팝업 내부 로그인 버튼 클릭...")
        popup_login_btn = wait.until(EC.element_to_be_clickable((By.ID, "mf_frameLogin1_btnLogin")))
        driver.execute_script("arguments[0].click();", popup_login_btn)
        print("⏳ 로그인 인증 대기 중 (8초)...")
        time.sleep(8)

        print("👉 [5단계] 위험물반입신고현황 페이지 진입...")
        target_url = "https://new.portmis.go.kr/portmis/websquare/websquare.jsp?w2xPath=/portmis/w2/main/index.xml&page=/portmis/w2/fr/dgst/dinu/dinu/UI-PM-FR-018-17.xml&menuId=9999&menuCd=M9024&menuNm=%EC%9C%84%ED%97%98%EB%AC%BC%EB%B0%98%EC%9E%85%EC%8B%A0%EA%B3%A0%ED%98%84%ED%99%A9"
        driver.get(target_url)
        time.sleep(5)

        # 💡 ESC 키를 통한 안전 팝업 닫기
        print("👉 [안전 장치] 팝업 닫기를 위한 ESC 키 입력 중...")
        time.sleep(3)
        try:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            print("✅ ESC 키 신호 전송 완료.")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ ESC 전송 중 예외 발생: {e}")

        # ------------------------------------------
        # 🔄 항구별 순회 수집 시작 (평택항 -> 대산항)
        # ------------------------------------------
        for port in ports_to_collect:
            port_name = port["name"]
            port_code = port["code"]
            output_filename = port["filename"]

            print(f"\n==========================================")
            print(f"🚢 [{port_name} (청코드: {port_code})] 데이터 수집 프로세스 개시")
            print(f"==========================================")

            port_integrated_data = []

            # 6단계: 청코드 입력
            print(f"👉 [6단계] 청코드 '{port_code}'({port_name}) 입력 및 갱신...")
            driver.execute_script(f"""
                var codeInput = document.getElementById('mf_tacMain_contents_M9024_body_prtAgCd_cmmCd') || 
                                document.querySelector('[id*="prtAgCd_cmmCd"]');
                if (codeInput) {{
                    codeInput.focus();
                    codeInput.value = '{port_code}';
                    codeInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    codeInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    codeInput.blur();
                }}
            """)
            time.sleep(3)

            # 7단계: 신고일자 자동 입력 (시작일: 3일 전 / 종료일: 오늘)
            kst_now = datetime.utcnow() + timedelta(hours=9)
            today_date = kst_now.date()
            today_str = kst_now.strftime("%Y%m%d")
            from_str = (kst_now - timedelta(days=3)).strftime("%Y%m%d")

            print(f"👉 [7단계] 신고일자 설정: 시작일({from_str}) ~ 종료일({today_str})...")
            driver.execute_script(f"""
                var fromInput = document.getElementById('mf_tacMain_contents_M9024_0_body_calfromReqstDt_input') || 
                                document.getElementById('mf_tacMain_contents_M9024_body_calfromReqstDt_input') ||
                                document.querySelector('[id*="calfromReqstDt_input"]');
                var toInput = document.getElementById('mf_tacMain_contents_M9024_0_body_caltoReqstDt_input') || 
                              document.getElementById('mf_tacMain_contents_M9024_body_caltoReqstDt_input') ||
                              document.querySelector('[id*="caltoReqstDt_input"]');
                
                if (fromInput) {{
                    fromInput.focus();
                    fromInput.value = '{from_str}';
                    fromInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    fromInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    fromInput.blur();
                }}
                if (toInput) {{
                    toInput.focus();
                    toInput.value = '{today_str}';
                    toInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    toInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    toInput.blur();
                }}
            """)
            time.sleep(3)

            # 8단계: 메인 검색 버튼 클릭
            print("👉 [8단계] 메인 [검색] 버튼 클릭...")
            driver.execute_script("""
                var searchBtn = document.getElementById('mf_tacMain_contents_M9024_body_udcSearch_btnSearch');
                if (searchBtn) {
                    var anchor = searchBtn.querySelector('a');
                    if (anchor) { anchor.click(); } else { searchBtn.click(); }
                }
            """)
            print("⏳ 데이터 조회 대기 중 (5초)...")
            time.sleep(5)

            # 9단계: 총 건수 및 총 페이지 수 파싱 (기본 10개씩 보기 모드)
            total_count = 0
            for attempt in range(4):
                total_count = driver.execute_script("""
                    var countSpan = document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageView_txtTotalDataCount');
                    if (countSpan) {
                        var text = countSpan.innerText;
                        var match = text.match(/\\d+/);
                        if (match) { return parseInt(match[0]); }
                    }
                    return 0;
                """)
                if total_count > 0:
                    break
                time.sleep(2)

            print(f"📊 [{port_name}] 총 데이터 건수: {total_count}건")

            if total_count == 0:
                print(f"⚠️ [{port_name}] 조회된 데이터가 없어 다음 항구로 이동합니다.")
                continue

            total_pages = (total_count + 9) // 10  # 올림 계산 (10개씩 기준)
            print(f"📄 [{port_name}] 총 {total_pages}개 페이지 전수 수집을 시작합니다.")

            # ------------------------------------------
            # 10단계: 웹스퀘어 정밀 ID 기반 전수 페이지네이션 탐색
            # ------------------------------------------
            for page_idx in range(1, total_pages + 1):
                print(f"\n==================== [{port_name} | {page_idx} / {total_pages} 페이지 탐색] ====================")

                # 페이지 이동 (1페이지가 아닐 때 웹스퀘어 정밀 ID로 정확히 클릭)
                if page_idx > 1:
                    moved = driver.execute_script(f"""
                        var targetPage = {page_idx};
                        // 웹스퀘어 고유 ID 구조 탐색
                        var pageEl = document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageList_pglGridView_page_' + targetPage);
                        
                        if (pageEl) {{
                            pageEl.click();
                            return 'direct';
                        }} else {{
                            // 페이지 번호가 보이지 않는 경우 (10페이지 단위 넘어감) -> '다음 페이지(>)' 버튼 클릭
                            var nextBtn = document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageList_pglGridView_next_btn');
                            if (nextBtn) {{
                                var aTag = nextBtn.querySelector('a') || nextBtn;
                                aTag.click();
                                return 'next_group';
                            }}
                        }}
                        return false;
                    """)
                    time.sleep(3.0)

                    # 10단위 그룹 이동(>) 후 목표 번호 다시 클릭
                    if moved == 'next_group':
                        driver.execute_script(f"""
                            var targetPage = {page_idx};
                            var pageEl = document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageList_pglGridView_page_' + targetPage);
                            if (pageEl) pageEl.click();
                        """)
                        time.sleep(3.0)

                # 현재 페이지 내의 10개 행 순회 (row 0~9)
                for row_idx in range(10):
                    cell_found = driver.execute_script(f"""
                        var rIdx = {row_idx};
                        var cell = document.querySelector('[id*="_tab1_grid_cell_' + rIdx + '_3"]');
                        if (cell && cell.offsetParent !== null) {{
                            cell.click();
                            return true;
                        }}
                        return false;
                    """)

                    if not cell_found:
                        break  # 현재 페이지의 마지막 행에 도달하면 다음 페이지로 이동

                    time.sleep(2.5)  # 상세화면 로딩 대기

                    # 기본 신고서 정보 수집
                    report_info = driver.execute_script("""
                        var getVal = function(idPattern) {
                            var el = document.querySelector('[id*="' + idPattern + '"]');
                            return el ? (el.value || el.innerText || '') : '';
                        };
                        return {
                            '선명': getVal('input270'),
                            '호출부호': getVal('input269'),
                            '사용목적': getVal('input274'),
                            '운송형태': getVal('input276'),
                            '화물명': getVal('input399'),
                            '하역업체': getVal('input302'),
                            '하역시작': getVal('input329'),
                            '하역종료': getVal('input330'),
                            '사용장소': getVal('input310'),
                            '전출항지': getVal('input336')
                        };
                    """)

                    ship_name = report_info['선명'] if report_info['선명'] else f"P{page_idx}_R{row_idx+1}"
                    call_sign = report_info['호출부호']
                    use_purpose = report_info['사용목적']
                    transport_type = report_info['운송형태']
                    cargo_name = report_info['화물명']
                    haeyuk_corp = report_info['하역업체']
                    haeyuk_start = report_info['하역시작']
                    haeyuk_end = report_info['하역종료']
                    haeyuk_period = haeyuk_start + ' ~ ' + haeyuk_end
                    use_place = report_info['사용장소']
                    prev_port = report_info['전출항지']

                    # 하역종료일 및 시간 파싱 (예: '2026-08-02 18:00' 형태)
                    end_clean_str = re.sub(r'[^0-9]', '', str(haeyuk_end))[:12]  # 숫자만 추출하여 최대 12자리(YYYYMMDDHHMM) 확보
                    
                    if len(end_clean_str) >= 12:
                        try:
                            # '202608021800' 형식을 datetime 객체로 변환
                            end_dt_obj = datetime.strptime(end_clean_str[:12], "%Y%m%d%H%M")
                            
                            # 현재 시간 기준 정확히 24시간 이전 시점 계산
                            cutoff_dt = datetime.now() - timedelta(hours=24)
                            
                            if end_dt_obj < cutoff_dt:
                                print(f"⏭️ [{port_name} | {page_idx}p-{row_idx+1}행] {ship_name} ➔ 하역종료시각({haeyuk_end})이 현재 기준 24시간 이전이므로 스킵")
                                # 목록 탭 복귀
                                driver.execute_script("""
                                    var listTab = document.querySelector('[id*="_tabDgst_tab_tabs0_tabHTML"]');
                                    if (listTab) listTab.click();
                                """)
                                time.sleep(2.0)
                                continue  # 스킵 후 다음 선박 탐색 계속 진행!
                        except Exception as dt_err:
                            print(f"⚠️ 날짜/시간 파싱 오류 (계속 진행): {dt_err}")

                    print(f"📌 [{port_name} | {page_idx}p-{row_idx+1}행] {ship_name} (호출부호: {call_sign}) 상세 수집 진입")

                    # 적하일람표 탭 이동
                    driver.execute_script("""
                        var tab = document.querySelector('[id*="_tabDgst_tab_tabs2_tabHTML"]');
                        if (tab) tab.click();
                    """)
                    time.sleep(2.5)

                    # 적하일람표 데이터 파싱
                    cargo_rows = driver.execute_script("""
                        var extractedData = [];
                        for (var r = 0; r < 50; r++) {
                            var cellNo = document.querySelector('[id*="_tab3_grid_cell_' + r + '_0"]');
                            if (!cellNo || cellNo.offsetParent === null) break;
                            
                            var getCellText = function(colIdx) {
                                var el = document.querySelector('[id*="' + '_tab3_grid_cell_' + r + '_' + colIdx + '"]');
                                return el ? el.innerText.trim() : '';
                            };
                            
                            extractedData.push([
                                getCellText(3),  // UNNO
                                getCellText(4),  // IMDG
                                getCellText(5),  // 품명
                                getCellText(6),  // 중량
                                getCellText(7)   // 단위
                            ]);
                        }
                        return extractedData;
                    """)

                    if cargo_rows:
                        for row in cargo_rows:
                            combined_row = [
                                ship_name, call_sign, use_purpose, transport_type, cargo_name,
                                haeyuk_corp, haeyuk_period, use_place, prev_port
                            ] + row
                            port_integrated_data.append(combined_row)
                        print(f"✅ [{ship_name}] 적하일람표 {len(cargo_rows)}건 수집 완료!")
                    else:
                        combined_row = [
                            ship_name, call_sign, use_purpose, transport_type, cargo_name,
                            haeyuk_corp, haeyuk_period, use_place, prev_port
                        ] + ['', '', '', '', '']
                        port_integrated_data.append(combined_row)
                        print(f"⚠️ [{ship_name}] 기본 신고서 정보 수집 완료!")

                    # 목록 탭 복귀
                    driver.execute_script("""
                        var listTab = document.querySelector('[id*="_tabDgst_tab_tabs0_tabHTML"]');
                        if (listTab) listTab.click();
                    """)
                    time.sleep(2.5)

            # ------------------------------------------
            # 11단계: 항구별 CSV 저장
            # ------------------------------------------
            if port_integrated_data:
                columns = [
                    "선박명(선택)", "호출부호", "사용목적", "운송형태", "화물명",
                    "하역업체", "하역기간", "사용장소", "전출항지",
                    "UNNO", "IMDG", "품명", "중량", "단위"
                ]
                df = pd.DataFrame(port_integrated_data, columns=columns)
                df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                print(f"\n🎉 [{port_name} 완료] 총 {len(port_integrated_data)}건 수집 완료 -> '{output_filename}' 저장")
            else:
                print(f"\n⚠️ [{port_name}] 조건에 맞는 수집 대상 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ [전체 RPA 에러 발생]: {e}")
    finally:
        time.sleep(3)
        driver.quit()
        print("🤖 RPA 봇 안전 종료.")

if __name__ == "__main__":
    run_real_rpa_crawler()
