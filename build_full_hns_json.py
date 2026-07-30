import pdfplumber
import json
import re

def extract_full_hns_text(pdf_path, output_json_path):
    print("🚀 [HNS 정보집 182종 전체 원본 텍스트 추출 가동...]")
    hns_database = []

    with pdfplumber.open(pdf_path) as pdf:
        # 본문 물질 페이지: 34페이지(인덱스 33) ~ 215페이지(인덱스 214)
        for page_num in range(39, 221):
            page = pdf.pages[page_num]
            text = page.extract_text()
            if not text:
                continue

            # 페이지 첫 줄에서 국문/영문 물질명 파악
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            header_title = lines[0] if lines else f"물질_{page_num+1}"

            # UN 번호 정밀 추출 (있을 경우 4자리, 없으면 빈값)
            unno_match = re.search(r'UN번호\s*(\d{4})', text)
            unno = unno_match.group(1).strip() if unno_match else ""

            # CAS 번호 정밀 추출
            cas_match = re.search(r'CAS번호\s*([\d\-]+)', text)
            cas_no = cas_match.group(1).strip() if cas_match else ""

            # 유사명(관용명) 추출
            synonym_match = re.search(r'유사명\s*([^\n]+)', text)
            synonyms = synonym_match.group(1).strip() if synonym_match else ""

            item = {
                "page": page_num + 1,
                "title_header": header_title,
                "unno": unno,
                "cas_no": cas_no,
                "synonyms": synonyms,
                "raw_full_text": text  # 💡 물질 페이지의 전체 텍스트 원본 보존!
            }
            
            hns_database.append(item)
            print(f"✅ [{page_num+1}페이지] {header_title} (UN: {unno}) 원본 추출 완료")

    # JSON 저장
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(hns_database, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 완료! 총 {len(hns_database)}개 물질 페이지 원본이 '{output_json_path}'에 저장되었습니다.")

if __name__ == "__main__":
    extract_full_hns_text("해상운송 위험유해물질 정보집(HNS 정보집)2024.pdf", "hns_full_text_database.json")