# app.py

import io
import re
import json
import streamlit as st
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account

#
# 1) Streamlit Secrets에서 서비스 계정 JSON 가져오기
#
# Secrets에 "gcp_service_account" 키로 붙여넣은 JSON 전체를 파이썬 dict로 로드
service_account_info = json.loads(st.secrets["gcp_service_account"])
creds = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

#
# 2) 추출할 필드별 정규표현식 패턴 정의
#
FIELD_PATTERNS = {
    # 기본정보
    "이름":            r"이름[:\s]*([가-힣A-Za-z· ]+)",
    "전번":            r"전번[:\s]*([\d\s\-]+)",
    "생년":            r"생년[:\s]*(\d{6,8})",
    "결합":            r"결합[:\s]*([가-힣A-Za-z0-9]+)",
    "주소":            r"주소[:\s]*(.+?)(?=\n)",

    # U+ 인터넷
    "U+ 인터넷":       r"U\+\s*인터넷[:\s]*([0-9]+)",
    "인터넷_요금제":    r"요금제[:\s]*([^\n]+)",
    "인터넷_약정시작":   r"약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "인터넷_약정종료":   r"약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "인터넷_단말":     r"단말[:\s]*([^\n]+)",

    # U+ TV (주)
    "U+ TV (주)":      r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "TV주_요금제":      r"U\+\s*TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":    r"U\+\s*TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV주_약정종료":    r"U\+\s*TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":       r"U\+\s*TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",

    # U+ TV (부)
    "U+ TV (부)":      r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":      r"U\+\s*TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":    r"U\+\s*TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV부_약정종료":    r"U\+\s*TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":       r"U\+\s*TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",

    # U+ 스마트홈
    "U+ 스마트홈":     r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":   r"U\+\s*스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작": r"U\+\s*스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "스마트홈_약정종료": r"U\+\s*스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":    r"U\+\s*스마트홈[\s\S]*?단말[:\s]*([^\n]+)",

    # 기타
    "공용단말":         r"공용단말[:\s]*([^\n]+)",
    "고객희망일":       r"고객희망일[:\s]*([0-9\-]+)"
}

#
# 3) OCR 호출 및 텍스트 반환 함수
#
def ocr_google_vision(image_bytes: bytes) -> str:
    image = vision.Image(content=image_bytes)
    resp  = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

#
# 4) 파싱 함수: raw_text에서 패턴별로 값만 추출
#
def parse_all_fields(raw_text: str) -> dict:
    data = {}
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, raw_text)
        data[field] = m.group(1).strip() if m else None
    return data

#
# 5) Streamlit UI
#
st.set_page_config(page_title="OCR → 항목별 추출 → 엑셀", layout="wide")
st.title("🧾 이미지 일괄 OCR → 지정 필드만 추출 → 엑셀 저장")

uploaded = st.file_uploader("📂 이미지 업로드 (여러 장 선택)", 
                             type=["jpg","jpeg","png"], 
                             accept_multiple_files=True)

if uploaded:
    rows = []
    progress = st.progress(0)
    for i, file in enumerate(uploaded):
        try:
            raw = ocr_google_vision(file.read())
            parsed = parse_all_fields(raw)
            parsed["파일명"] = file.name
        except Exception as e:
            parsed = {"파일명": file.name, "오류": str(e)}
        rows.append(parsed)
        progress.progress((i+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # 엑셀 출력
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "📥 엑셀 다운로드",
        data=buffer.getvalue(),
        file_name="ocr_extracted_fields.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
