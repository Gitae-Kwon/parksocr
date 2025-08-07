# app.py
import os
import io
import re
import streamlit as st
import pandas as pd
from google.cloud import vision

# 1) Google Vision 클라이언트 초기화
#    환경 변수 GOOGLE_APPLICATION_CREDENTIALS로 JSON 키 경로가 지정되어 있어야 함
client = vision.ImageAnnotatorClient()

# 2) 추출할 필드별 정규표현식 패턴 정의
FIELD_PATTERNS = {
    "이름":        r"이름[:\s]*([가-힣A-Za-z]+)",
    "U+ 인터넷":  r"U\+\s*인터넷[:\s]*([0-9]+)",
    "U+ TV (주)":  r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "U+ TV (부)":  r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "U+ 스마트홈": r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "결합":        r"결합[:\s]*([가-힣A-Za-z0-9]+)",
    "주소":        r"주소[:\s]*(.+?)(?=\s{2,}|\n|$)"
}

def ocr_google_vision(image_bytes: bytes) -> str:
    """Google Vision으로 이미지 전체 텍스트 추출"""
    image = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

def parse_fields(text: str) -> dict:
    """미리 정의한 FIELD_PATTERNS에 맞춰 텍스트에서 값만 뽑아 dict로 반환"""
    data = {}
    for fld, pat in FIELD_PATTERNS.items():
        m = re.search(pat, text)
        data[fld] = m.group(1).strip() if m else None
    return data

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Google OCR → 지정 필드 추출 → 엑셀", layout="wide")
st.title("📷 이미지 일괄 OCR → 지정 필드만 추출 → 엑셀 저장")

uploaded = st.file_uploader(
    "📂 이미지 업로드 (여러 장 선택 가능)", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

if uploaded:
    rows = []
    progress = st.progress(0)

    for idx, file in enumerate(uploaded):
        try:
            img_bytes = file.read()
            raw_text = ocr_google_vision(img_bytes)
            parsed = parse_fields(raw_text)
            parsed["파일명"] = file.name
            rows.append(parsed)
        except Exception as e:
            rows.append({"파일명": file.name, "오류": str(e)})
        progress.progress((idx+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            "📥 엑셀 파일 다운로드",
            data=buf.getvalue(),
            file_name="ocr_extracted_fields.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
