from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
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
    
    # 💡 GitHub Actions (우분투 리눅스) 실행을 위한 Chrome Headless 옵션 설정
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
    
    all_integrated_data = [] # 모든 선박의 신고서 + 적하일람표 데이터를 모을 리스트
    
    try:
        # ------------------------------------------
        # 1단계 ~ 5단계: 로그인 및 페이지 진입
        # ------------------------------------------
        print("👉 [1단계] 인트로 페이지 접속 중...")
        driver.get("https://new.portmis.go.kr/portmis/websquare/websquare.jsp?w2xPath=/portmis/w2/main/intro.xml")
        time.sleep(4)

        print("👉 [2단계] 로그인 팝업 호출...")
        login_trigger = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[text()='로그인'] | //button[text()='로그인'] | //*[contains(@class, 'login')]//a"))
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

        # ------------------------------------------
        # 6단계: 청코드 '031' 입력 및 '평택' 갱신 트리거 실행
        # ------------------------------------------
        print("👉 [6단계] 청코드 '031' 입력 및 갱신 이벤트 실행...")
        driver.execute_script("""
            var codeInput = document.getElementById('mf_tacMain_contents_M9024_body_prtAgCd_cmmCd');
            if (codeInput) {
                codeInput.focus();
                codeInput.value = '031';
                codeInput.dispatchEvent(new Event('input', { bubbles: true }));
                codeInput.dispatchEvent(new Event('change', { bubbles: true }));
                codeInput.blur();
            }
        """)
        time.sleep(2)

        # ------------------------------------------
        # 6-1단계: KST 기준 오늘 날짜 자동 입력
        # ------------------------------------------
        kst_now = datetime.utcnow() + timedelta(hours=9)
        today_str = kst_now.strftime("%Y%m%d") # 예: '20260730'
        
        print(f"👉 [6-1단계] 신고일자 시작일/종료일 오늘 날짜({today_str}) 설정 중...")
        driver.execute_script(f"""
            var fromInput = document.getElementById('mf_tacMain_contents_M9024_0_body_calfromReqstDt_input') || document.getElementById('mf_tacMain_contents_M9024_body_calfromReqstDt_input');
            var toInput = document.getElementById('mf_tacMain_contents_M9024_0_body_caltoReqstDt_input') || document.getElementById('mf_tacMain_contents_M9024_body_caltoReqstDt_input');
            
            if (fromInput) {{
                fromInput.value = '{today_str}';
                fromInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                fromInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
            if (toInput) {{
                toInput.value = '{today_str}';
                toInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                toInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
        """)
        time.sleep(2)

        # ------------------------------------------
        # 7단계: 메인 검색 버튼 클릭
        # ------------------------------------------
        print("👉 [7단계] 메인 조회 검색 버튼 클릭...")
        search_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[@title='검색' and text()='검색'] | //a[text()='조회']"))
        )
        driver.execute_script("arguments[0].click();", search_btn)
        print("⏳ 데이터 1차 조회 대기 중 (5초)...")
        time.sleep(5)

        # ------------------------------------------
        # 7-1단계: '100개씩 보기' 선택 및 재조회
        # ------------------------------------------
        print("👉 [7-1단계] 목록 표시 수 '100개씩 보기'로 변경 중...")
        try:
            select_element = driver.execute_script("""
                return document.getElementById('mf_tacMain_contents_M9024_0_body_udcGridPageView_sbxRecordCount_input_0') ||
                       document.getElementById('mf_tacMain_contents_M9024_body_udcGridPageView_sbxRecordCount_input_0');
            """)
            if select_element:
                select_box = Select(select_element)
                select_box.select_by_visible_text("100개씩 보기")
                
                # 변경 이벤트 트리거로 웹스퀘어 목록 갱신
                driver.execute_script("""
                    arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                """, select_element)
                print("✅ '100개씩 보기' 적용 완료! 100건 데이터 재로딩 대기 중 (5초)...")
                time.sleep(5)
            else:
                print("⚠️ '100개씩 보기' 드롭다운을 찾지 못하여 기본 갯수로 진행합니다.")
        except Exception as sel_err:
            print(f"⚠️ '100개씩 보기' 변경 중 소폭 예외 발생 (기존 개수로 계속): {sel_err}")

        # ------------------------------------------
        # 8단계: 선박 리스트 순회 및 데이터 통합 수집
        # ------------------------------------------
        print("👉 [8단계] 선박 리스트 분석 및 양쪽 탭 데이터 수집 시작...")
        
        row_indices = driver.execute_script("""
            var cells = document.querySelectorAll('[id*="mf_tacMain_contents_M9024_"][id*="_tab1_grid_cell_"][id$="_3"]');
            var validIndices = [];
            for (var i = 0; i < cells.length; i++) {
                if (cells[i].offsetParent !== null) {
                    validIndices.push(i);
                }
            }
            return validIndices;
        """)
        
        print(f"📊 화면에 실제로 노출된 총 {len(row_indices)}척의 선박 데이터를 순회합니다.")

        for idx, i in enumerate(row_indices):
            print(f"\n--- [{idx+1}/{len(row_indices)}번째 선박] 데이터 추출 중 ---")
            
            try:
                # 1. 선박 셀 클릭
                cell_id_script = f"""
                    var el = document.querySelector('[id*="mf_tacMain_contents_M9024_"][id*="_tab1_grid_cell_{i}_3"]');
                    return el ? el.id : 'mf_tacMain_contents_M9024_body_tab1_grid_cell_{i}_3';
                """
                target_cell_id = driver.execute_script(cell_id_script)
                ship_cell = wait.until(EC.element_to_be_clickable((By.ID, target_cell_id)))
                ship_name = ship_cell.text.strip()
                
                driver.execute_script("arguments[0].click();", ship_cell)
                time.sleep(2.5)

                # 2. [위험물반입신고서 탭 정보 수집]
                report_info = driver.execute_script("""
                    var getVal = function(idPattern) {
                        var el = document.querySelector('[id*="mf_tacMain_contents_M9024_"][id*="' + idPattern + '"]');
                        return el ? el.value : '';
                    };
                    return {
                        '청명': getVal('input266'),
                        '반입구분': getVal('input268'),
                        '호출부호': getVal('input269'),
                        '선명': getVal('input270'),
                        '화물명': getVal('input399'),
                        '컨테이너갯수': getVal('input297'),
                        '총량': getVal('input298'),
                        '하역업체': getVal('input302'),
                        '하역기간시작': getVal('input329'),
                        '하역기간종료': getVal('input330'),
                        '사용장소': getVal('input310'),
                        '신고일시': getVal('input331')
                    };
                """)

                # 3. [위험물적하일람표] 탭으로 이동
                cargo_tab = driver.execute_script("""
                    return document.querySelector('[id*="mf_tacMain_contents_M9024_"][id*="_tabDgst_tab_tabs2_tabHTML"]');
                """)
                if cargo_tab:
                    driver.execute_script("arguments[0].click();", cargo_tab)
                time.sleep(2.5)

                # 4. 적하일람표 테이블 데이터 정밀 파싱
                cargo_rows = driver.execute_script("""
                    var extractedData = [];
                    for (var r = 0; r < 50; r++) {
                        var cellNo = document.querySelector('[id*="mf_tacMain_contents_M9024_"][id*="_tab3_grid_cell_' + r + '_0"]');
                        if (!cellNo || cellNo.offsetParent === null) break;
                        
                        var getCellText = function(colIdx) {
                            var el = document.querySelector('[id*="mf_tacMain_contents_M9024_"][id*="_tab3_grid_cell_' + r + '_' + colIdx + '"]');
                            return el ? el.innerText.trim() : '';
                        };
                        
                        extractedData.push([
                            getCellText(0),  // 순번
                            getCellText(1),  // 검사증번호
                            getCellText(2),  // 순번2
                            getCellText(3),  // UNNO
                            getCellText(4),  // IMDG
                            getCellText(5),  // 품명
                            getCellText(6),  // 중량
                            getCellText(7),  // 단위
                            getCellText(8),  // 업체코드
                            getCellText(10), // 업체명
                            getCellText(16), // 하역장소
                            getCellText(17), // 반입일
                            getCellText(18), // 작업구분
                            getCellText(19), // 수하인
                            getCellText(20)  // 송하인
                        ]);
                    }
                    return extractedData;
                """)

                # 데이터 결합 저장
                if cargo_rows:
                    for row in cargo_rows:
                        combined_row = [
                            ship_name, report_info['청명'], report_info['반입구분'], 
                            report_info['호출부호'], report_info['화물명'], report_info['하역업체'], 
                            report_info['하역기간시작'] + ' ~ ' + report_info['하역기간종료'], report_info['사용장소']
                        ] + row
                        all_integrated_data.append(combined_row)
                    print(f"✅ [{ship_name}] 신고서 및 적하일람표 항목 {len(cargo_rows)}건 통합 수집 완료!")
                else:
                    combined_row = [
                        ship_name, report_info['청명'], report_info['반입구분'], 
                        report_info['호출부호'], report_info['화물명'], report_info['하역업체'], 
                        report_info['하역기간시작'] + ' ~ ' + report_info['하역기간종료'], report_info['사용장소']
                    ] + ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
                    all_integrated_data.append(combined_row)
                    print(f"⚠️ [{ship_name}] 적하일람표 데이터는 없으나 기본 신고서 정보를 수집했습니다.")

                # 5. 목록으로 복귀
                list_tab = driver.execute_script("""
                    return document.querySelector('[id*="mf_tacMain_contents_M9024_"][id*="_tabDgst_tab_tabs0_tabHTML"]');
                """)
                if list_tab:
                    driver.execute_script("arguments[0].click();", list_tab)
                time.sleep(3)

            except Exception as e:
                print(f"⚠️ [{idx+1}번째 선박] 처리 중 예외 발생: {e}")
                try:
                    driver.get(target_url)
                    time.sleep(3)
                except:
                    pass
                continue

        # ------------------------------------------
        # 9단계: 최종 통합 파일(CSV) 저장
        # ------------------------------------------
        if all_integrated_data:
            columns = [
                "선박명(선택)", "청명", "반입구분", "호출부호", "대표화물명", "하역업체", "하역기간", "사용장소",
                "순번", "검사증번호", "순번2", "UNNO", "IMDG CLA", "품명", "중량", "단위", 
                "업체코드", "업체명", "하역장소(상세)", "반입일", "작업구분", "수하인", "송하인"
            ]
            
            df = pd.DataFrame(all_integrated_data, columns=columns)
            output_filename = "hns_fully_integrated_report.csv"
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n🎉 [최종 완료] 총 {len(all_integrated_data)}건의 신고서 및 적하일람표 데이터가 '{output_filename}' 파일로 완벽하게 통합 저장되었습니다!")
        else:
            print("\n⚠️ 수집된 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ [전체 RPA 에러 발생]: {e}")
    finally:
        time.sleep(3)
        driver.quit()
        print("🤖 RPA 봇 안전 종료.")

if __name__ == "__main__":
    run_real_rpa_crawler()
