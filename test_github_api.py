import os
import sys
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 환경 변수(GitHub Secrets) 또는 로컬 secrets.toml에서 키 로드
PUBLIC_API_KEY = os.environ.get("PUBLIC_API_KEY", "")

if not PUBLIC_API_KEY:
    print("❌ [ERROR] PUBLIC_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

def run_api_test():
    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
    sde = (now_kst - timedelta(days=1)).strftime("%Y%m%d")
    ede = (now_kst + timedelta(days=1)).strftime("%Y%m%d")

    targets = [
        ("평택항 입항", "031", "I"),
        ("평택항 출항", "031", "O"),
        ("대산항 입항", "300", "I"),
        ("대산항 출항", "300", "O")
    ]

    print("=" * 70)
    print("🔬 [공공데이터포털 API 수신 테스트] (GitHub Actions / Local 공용)")
    print(f"📅 조회 기준일자 (KST): {sde} ~ {ede}")
    print("=" * 70)

    session = requests.Session()
    session.verify = False

    success_flag = True

    for name, port, gb in targets:
        url = f"https://apis.data.go.kr/1192000/VsslEtrynd5/Info5?serviceKey={PUBLIC_API_KEY}"
        params = {
            'prtAgCd': port,
            'sde': sde,
            'ede': ede,
            'deGb': gb,
            'numOfRows': '50',
            'pageNo': '1'
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        print(f"\n▶️ {name} ({port}/{gb}) 호출 중...")
        try:
            res = session.get(url, params=params, headers=headers, timeout=10)
            print(f"   📡 HTTP Status: {res.status_code}")

            if res.status_code == 200 and res.content:
                root = ET.fromstring(res.content)
                result_code = root.findtext('.//resultCode', 'NONE')
                result_msg = root.findtext('.//resultMsg', 'NONE')
                total_cnt = root.findtext('.//totalCount', '0')

                print(f"   📊 응답코드: {result_code} ({result_msg}) | 총 수신 건수: {total_cnt}건")

                if result_code == "00" and int(total_cnt) > 0:
                    print(f"   ✅ [성공] 정상적으로 {total_cnt}건 수신되었습니다.")
                else:
                    print(f"   ⚠️ [경고] 응답 성공했으나 데이터가 0건이거나 에러코드 반환됨.")
            else:
                print(f"   ❌ [실패] HTTP 상태 코드: {res.status_code}")
                success_flag = False

        except requests.exceptions.Timeout:
            print("   🚨 [실패] ConnectTimeoutError 발생 (해외 IP 차단 확정)")
            success_flag = False
        except Exception as e:
            print(f"   ❌ [실패] 예외 발생: {e}")
            success_flag = False

    print("\n" + "=" * 70)
    if success_flag:
        print("🎉 모든 API 호출이 정상 수신되었습니다! (국내 IP 환경)")
    else:
        print("💥 해외 IP 차단 또는 API 수신 실패가 확인되었습니다.")
    print("=" * 70)

if __name__ == "__main__":
    run_api_test()
