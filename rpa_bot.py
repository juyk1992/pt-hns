from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from datetime import datetime, timedelta
import pandas as pd
import time
import os

# ==========================================
# ⚙️ 계정 설정 영역
# ==========================================
PORTMIS_ID = os.getenv("PORTMIS_ID", "")
PORTMIS_PW = os.getenv("PORTMIS_PW", "")

def run_real_rpa_crawler():
    print("🤖 [종합 데이터 통합 마스터 RPA] 포트미스 자동화 봇 가동...")
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')          # 화면 없이 실행
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    chrome_binaries = ["/usr/bin/chromium-browser", "/usr/bin/google-chrome", "/usr/bin/chromium"]
    for bin_path in chrome_binaries:
        if os.path.exists(bin_path):
            options.binary_location = bin_path
            break

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)
    
    # 📌 수집 대상 항구 정의 (평택항: 031, 대산항: 300)
    ports_to_collect = [
        {"name": "평택항", "code": "031", "filename": "hns_pyeongtaek_report.csv"},
        {"name": "대산항", "code": "300", "filename": "hns_daesan_report.csv"}
    ]
    
    try:
        # ------------------------------------------
        # 1단계 ~ 5단계: 로그인 및 페이지 진입 (1회만 수행)
        # ------------------------------------------
        print("👉 [1단계] 인트로 페이지 접속 중...")
        driver.get("https://new.portmis.go.kr/portmis/websquare/websquare.jsp?w2xPath=/portmis/w2/main/intro.xml")
        time.sleep(4)

        print("👉 [2단계] 로그인 팝업 호출...")
        login_trigger = wait.until(
            EC.element_to_be_clickable((By.ID, "mf_btnLogin"))
        )
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
        popup_login_btn = wait.until(
            EC.element_to_be_clickable((By.ID, "mf_frameLogin1_btnLogin"))
        )
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
            
            port_integrated_data = [] # 항구별 데이터를 담을 독립 리스트

            # ------------------------------------------
            # 6단계: 청코드 입력
            # ------------------------------------------
            print(f"👉 [6단계] 청코드 '{port_code}'({port_name}) 입력 및 갱신 이벤트 실행...")
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
            time.sleep(2)

            # ------------------------------------------
            # 7단계: 오늘 날짜 자동 입력
            # ------------------------------------------
            kst_now = datetime.utcnow() + timedelta(hours=9)
            today_str = kst_now.strftime("%Y%m%d")
            
            print(f"👉 [7단계] 신고일자 시작일/종료일 오늘 날짜({today_str}) 설정...")
            driver.execute_script(f"""
                var fromInput = document.getElementById('mf_tacMain_contents_M9024_0_body_calfromReqstDt_input') || 
                                document.getElementById('mf_tacMain_contents_M9024_body_calfromReqstDt_input') ||
                                document.querySelector('[id*="calfromReqstDt_input"]');
                var toInput = document.getElementById('mf_tacMain_contents_M9024_0_body_caltoReqstDt_input') || 
                              document.getElementById('mf_tacMain_contents_M9024_body_caltoReqstDt_input') ||
                              document.querySelector('[id*="caltoReqstDt_input"]');
                
                if (fromInput) {{
                    fromInput.focus();
                    fromInput.value = '{today_str}';
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
            time.sleep(2)

            # ------------------------------------------
            # 8단계: 메인 검색 버튼 클릭
            # ------------------------------------------
            print("👉 [8단계] 우측 상단 메인 [검색] 버튼 정밀 클릭...")
            driver.execute_script("""
                var searchBtn = document.getElementById('mf_tacMain_contents_M9024_body_udcSearch_btnSearch');
                if (searchBtn) {
                    var anchor = searchBtn.querySelector('a');
                    if (anchor) { anchor.click(); } else { searchBtn.click(); }
                }
            """)
            print("⏳ 데이터 1차 조회 대기 중 (5초)...")
            time.sleep(5)

            # ------------------------------------------
            # 9단계: '100개씩 보기' 선택 및 적용
            # ------------------------------------------
            print("👉 [9단계] 목록 표시 수 '100개씩 보기' 변경...")
            try:
                select_element = driver.execute_script("""
                    return document.getElementById('mf_tacMain_contents_M9024_0_body_udcGridPageView_sbxRecordCount_input_0') ||
                           document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageView_sbxRecordCount_input_0') ||
                           document.querySelector('[id*="sbxRecordCount_input_0"]');
                """)
                if select_element:
                    select_box = Select(select_element)
                    select_box.select_by_visible_text("100개씩 보기")
                    driver.execute_script("""
                        arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                    """, select_element)
                    print("✅ '100개씩 보기' 적용 완료! 대기 중 (5초)...")
                    time.sleep(5)
            except Exception as sel_err:
                print(f"⚠️ '100개씩 보기' 변경 중 예외 발생: {sel_err}")

            # ------------------------------------------
            # 10단계: 건수 파싱 및 전수 수집
            # ------------------------------------------
            print(f"👉 [10단계] {port_name} 총 건수 파싱 및 수집 시작...")
            
            total_count = driver.execute_script("""
                var countSpan = document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageView_txtTotalDataCount');
                if (countSpan) {
                    var text = countSpan.innerText;
                    var match = text.match(/\\d+/);
                    if (match) { return parseInt(match[0]); }
                }
                var maxRnum = 0;
                for (var i = 0; i < 100; i++) {
                    var cellNo = document.querySelector('[id*="_tab1_grid_cell_' + i + '_0"]');
                    if (cellNo) {
                        var val = parseInt(cellNo.innerText.trim()) || 0;
                        if (val > maxRnum) maxRnum = val;
                    }
                }
                return maxRnum > 0 ? maxRnum : 0;
            """)
            
            print(f"📊 [{port_name}] 실시간 파싱된 총 데이터 건수: {total_count}건.")

            if total_count == 0:
                print(f"⚠️ [{port_name}] 조회된 데이터가 없어 다음 항구로 이동합니다.")
                continue

            for target_rnum in range(total_count, 0, -1):
                print(f"\n--- [{port_name} | 순번: {target_rnum}] 데이터 수집 시도 ---")

                try:
                    click_success = driver.execute_script(f"""
                        var targetRnum = {target_rnum};
                        var scrollDiv = document.querySelector('[id*="tab1_grid_scrollY_div"]') || document.querySelector('.w2grid_scrollY');
                        
                        if (scrollDiv) {{ scrollDiv.scrollTop = 0; }}
                        
                        for (var s = 0; s < 40; s++) {{
                            if (scrollDiv) {{ 
                                scrollDiv.scrollTop = s * 25; 
                            }}
                            
                            for (var i = 0; i < 100; i++) {{
                                var cellNo = document.querySelector('[id*="_tab1_grid_cell_' + i + '_0"]');
                                if (cellNo && parseInt(cellNo.innerText.trim()) === targetRnum) {{
                                    var targetCell = document.querySelector('[id*="_tab1_grid_cell_' + i + '_3"]');
                                    if (targetCell) {{
                                        targetCell.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                                        targetCell.click();
                                        return true;
                                    }}
                                }}
                            }}
                        }}
                        return false;
                    """)

                    if not click_success:
                        time.sleep(1)
                        click_success = driver.execute_script(f"""
                            var targetRnum = {target_rnum};
                            var scrollDiv = document.querySelector('[id*="tab1_grid_scrollY_div"]') || document.querySelector('.w2grid_scrollY');
                            
                            if (scrollDiv) {{ scrollDiv.scrollTop = scrollDiv.scrollHeight; }}
                            
                            for (var s = 30; s >= 0; s--) {{
                                if (scrollDiv) {{ scrollDiv.scrollTop = s * 30; }}
                                
                                for (var i = 0; i < 100; i++) {{
                                    var cellNo = document.querySelector('[id*="_tab1_grid_cell_' + i + '_0"]');
                                    if (cellNo && parseInt(cellNo.innerText.trim()) === targetRnum) {{
                                        var targetCell = document.querySelector('[id*="_tab1_grid_cell_' + i + '_3"]');
                                        if (targetCell) {{
                                            targetCell.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                                            targetCell.click();
                                            return true;
                                        }}
                                    }}
                                }}
                            }}
                            return false;
                        """)

                    if not click_success:
                        print(f"⚠️ [{port_name} | 순번 {target_rnum}] 행을 찾지 못해 건너뜁니다.")
                        continue

                    time.sleep(3.0)

                    # 신고서 기본 정보 수집
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

                    ship_name = report_info['선명'] if report_info['선명'] else f"순번_{target_rnum}"
                    call_sign = report_info['호출부호']
                    use_purpose = report_info['사용목적']
                    transport_type = report_info['운송형태']
                    cargo_name = report_info['화물명']
                    haeyuk_corp = report_info['하역업체']
                    haeyuk_period = report_info['하역시작'] + ' ~ ' + report_info['하역종료']
                    use_place = report_info['사용장소']
                    prev_port = report_info['전출항지']

                    print(f"📌 [{port_name} | 순번 {target_rnum}] 선명: {ship_name} (호출부호: {call_sign}) 상세 진입")

                    # 적하일람표 탭 이동
                    driver.execute_script("""
                        var tab = document.querySelector('[id*="_tabDgst_tab_tabs2_tabHTML"]');
                        if (tab) tab.click();
                    """)
                    time.sleep(3.0)

                    # 적하일람표 테이블 데이터 파싱
                    cargo_rows = driver.execute_script("""
                        var extractedData = [];
                        for (var r = 0; r < 50; r++) {
                            var cellNo = document.querySelector('[id*="_tab3_grid_cell_' + r + '_0"]');
                            if (!cellNo || cellNo.offsetParent === null) break;
                            
                            var getCellText = function(colIdx) {
                                var el = document.querySelector('[id*="_tab3_grid_cell_' + r + '_' + colIdx + '"]');
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
                        print(f"✅ [{port_name} - {ship_name}] 적하일람표 {len(cargo_rows)}건 수집 완료!")
                    else:
                        combined_row = [
                            ship_name, call_sign, use_purpose, transport_type, cargo_name,
                            haeyuk_corp, haeyuk_period, use_place, prev_port
                        ] + ['', '', '', '', '']
                        port_integrated_data.append(combined_row)
                        print(f"⚠️ [{port_name} - {ship_name}] 기본 신고서 정보만 수집 완료!")

                    # 목록 탭 복귀
                    driver.execute_script("""
                        var listTab = document.querySelector('[id*="_tabDgst_tab_tabs0_tabHTML"]');
                        if (listTab) listTab.click();
                    """)
                    time.sleep(3.0)

                except Exception as e:
                    print(f"⚠️ [{port_name} | 순번 {target_rnum}] 처리 중 예외 발생: {e}")
                    driver.execute_script("""
                        var listTab = document.querySelector('[id*="_tabDgst_tab_tabs0_tabHTML"]');
                        if (listTab) listTab.click();
                    """)
                    time.sleep(3.0)
                    continue

            # ------------------------------------------
            # 11단계: 항구별 저장 (Loop 완료 시 저장)
            # ------------------------------------------
            if port_integrated_data:
                columns = [
                    "선박명(선택)", "호출부호", "사용목적", "운송형태", "화물명",
                    "하역업체", "하역기간", "사용장소", "전출항지",
                    "UNNO", "IMDG", "품명", "중량", "단위"
                ]
                
                df = pd.DataFrame(port_integrated_data, columns=columns)
                df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                print(f"\n🎉 [{port_name} 완료] 총 {len(port_integrated_data)}건의 데이터가 '{output_filename}' 파일로 저장되었습니다!")
            else:
                print(f"\n⚠️ [{port_name}] 수집된 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ [전체 RPA 에러 발생]: {e}")
    finally:
        time.sleep(3)
        driver.quit()
        print("🤖 RPA 봇 안전 종료.")

if __name__ == "__main__":
    run_real_rpa_crawler()
