from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

# ==========================================
# ⚙️ 계정 설정 영역
# ==========================================
PORTMIS_ID = os.getenv("PORTMIS_ID")
PORTMIS_PW = os.getenv("PORTMIS_PW")

def run_real_rpa_crawler():
    print("🤖 [종합 데이터 통합 마스터 RPA] 포트미스 자동화 봇 가동...")
    
    options = webdriver.ChromeOptions()
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
        # 6단계: 청코드 '031' 입력 및 '평택' 갱신 트리거 실행 (복구 완료)
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
        print("⏳ [대기] 청이름('평택') 자동 갱신 대기 중 (3초)...")
        time.sleep(3)

        # ------------------------------------------
        # 7단계: 메인 검색 버튼 클릭
        # ------------------------------------------
        print("👉 [7단계] 메인 조회 검색 버튼 클릭...")
        search_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[@title='검색' and text()='검색']"))
        )
        driver.execute_script("arguments[0].click();", search_btn)
        print("⏳ 데이터 조회 렌더링 대기 중 (5초)...")
        time.sleep(5)

        # ------------------------------------------
        # 8단계: 선박 리스트 순회 및 신고서 + 적하일람표 데이터 통합 수집
        # ------------------------------------------
        print("👉 [8단계] 선박 리스트 분석 및 양쪽 탭 데이터 수집 시작...")
        
        row_indices = driver.execute_script("""
            var cells = document.querySelectorAll('[id^="mf_tacMain_contents_M9024_body_tab1_grid_cell_"][id$="_3"]');
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
                cell_id = f"mf_tacMain_contents_M9024_body_tab1_grid_cell_{i}_3"
                ship_cell = wait.until(EC.element_to_be_clickable((By.ID, cell_id)))
                ship_name = ship_cell.text.strip()
                
                driver.execute_script("arguments[0].click();", ship_cell)
                time.sleep(2.5)

                # 2. [위험물반입신고서 탭 정보 수집]
                report_info = driver.execute_script("""
                    return {
                        '청명': document.getElementById('mf_tacMain_contents_M9024_body_input266') ? document.getElementById('mf_tacMain_contents_M9024_body_input266').value : '',
                        '반입구분': document.getElementById('mf_tacMain_contents_M9024_body_input268') ? document.getElementById('mf_tacMain_contents_M9024_body_input268').value : '',
                        '호출부호': document.getElementById('mf_tacMain_contents_M9024_body_input269') ? document.getElementById('mf_tacMain_contents_M9024_body_input269').value : '',
                        '선명': document.getElementById('mf_tacMain_contents_M9024_body_input270') ? document.getElementById('mf_tacMain_contents_M9024_body_input270').value : '',
                        '화물명': document.getElementById('mf_tacMain_contents_M9024_body_input399') ? document.getElementById('mf_tacMain_contents_M9024_body_input399').value : '',
                        '컨테이너갯수': document.getElementById('mf_tacMain_contents_M9024_body_input297') ? document.getElementById('mf_tacMain_contents_M9024_body_input297').value : '',
                        '총량': document.getElementById('mf_tacMain_contents_M9024_body_input298') ? document.getElementById('mf_tacMain_contents_M9024_body_input298').value : '',
                        '하역업체': document.getElementById('mf_tacMain_contents_M9024_body_input302') ? document.getElementById('mf_tacMain_contents_M9024_body_input302').value : '',
                        '하역기간시작': document.getElementById('mf_tacMain_contents_M9024_body_input329') ? document.getElementById('mf_tacMain_contents_M9024_body_input329').value : '',
                        '하역기간종료': document.getElementById('mf_tacMain_contents_M9024_body_input330') ? document.getElementById('mf_tacMain_contents_M9024_body_input330').value : '',
                        '사용장소': document.getElementById('mf_tacMain_contents_M9024_body_input310') ? document.getElementById('mf_tacMain_contents_M9024_body_input310').value : '',
                        '신고일시': document.getElementById('mf_tacMain_contents_M9024_body_input331') ? document.getElementById('mf_tacMain_contents_M9024_body_input331').value : ''
                    };
                """)

                # 3. [위험물적하일람표] 탭으로 이동
                tab_id = "mf_tacMain_contents_M9024_body_tabDgst_tab_tabs2_tabHTML"
                cargo_tab = wait.until(EC.element_to_be_clickable((By.ID, tab_id)))
                driver.execute_script("arguments[0].click();", cargo_tab)
                time.sleep(2.5)

                # 4. 적하일람표 테이블 데이터 정밀 파싱
                cargo_rows = driver.execute_script("""
                    var extractedData = [];
                    for (var r = 0; r < 20; r++) {
                        var cellNo = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_0');
                        if (!cellNo || cellNo.offsetParent === null) break;
                        
                        var rnum = cellNo.innerText.trim();
                        var ispctn = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_1').innerText.trim();
                        var docSeq = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_2').innerText.trim();
                        var unno = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_3').innerText.trim();
                        var imdg = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_4').innerText.trim();
                        var productName = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_5').innerText.trim();
                        var weight = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_6').innerText.trim();
                        var unit = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_7').innerText.trim();
                        var entrpsCd = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_8').innerText.trim();
                        var entrpsNm = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_10').innerText.trim();
                        
                        var placeNm = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_16');
                        var tkinDt = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_17');
                        var workKind = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_18');
                        var consge = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_19');
                        var secong = document.getElementById('mf_tacMain_contents_M9024_body_tab3_grid_cell_' + r + '_20');
                        
                        extractedData.push([
                            rnum, ispctn, docSeq, unno, imdg, productName, weight, unit, 
                            entrpsCd, entrpsNm, 
                            placeNm ? placeNm.innerText.trim() : '', 
                            tkinDt ? tkinDt.innerText.trim() : '', 
                            workKind ? workKind.innerText.trim() : '', 
                            consge ? consge.innerText.trim() : '', 
                            secong ? secong.innerText.trim() : ''
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
                list_tab_id = "mf_tacMain_contents_M9024_body_tabDgst_tab_tabs0_tabHTML"
                list_tab = wait.until(EC.element_to_be_clickable((By.ID, list_tab_id)))
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