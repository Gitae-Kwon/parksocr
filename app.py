# app.py

import io
import re
import json
import streamlit as st
import pandas as pd
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# 1) 인증: Streamlit Secrets에 등록한 서비스 계정 JSON을 dict로 꺼내기
service_account_info = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

# 2) OCR 호출 함수 (전체 이미지)
def ocr_google_vision(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    content = buf.getvalue()
    resp = client.document_text_detection(image=vision.Image(content=content))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

# 3) 필드별 정규표현식 패턴 정의 (공용단말 강화)
FIELD_PATTERNS = {
    "이름":            r"이름[:\s]*([가-힣A-Za-z· ]+)",
    "전번":            r"전번[:\s]*([\d\s\-]+)",
    "생년":            r"생년[:\s]*(\d{6,8})",
    "결합":            r"결합[:\s]*([가-힣A-Za-z0-9]+)",
    "주소":            r"주소[:\s]*(.+?)(?=\n)",

    "U+ 인터넷":       r"U\+\s*인터넷[:\s]*([0-9]+)",
    "인터넷_요금제":    r"요금제[:\s]*([^\n]+)",
    "인터넷_약정시작":   r"약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "인터넷_약정종료":   r"약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "인터넷_단말":     r"단말[:\s]*([^\n]+)",

    "U+ TV (주)":      r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "TV주_요금제":      r"TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":    r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV주_약정종료":    r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":      r"TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ TV (부)":      r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":      r"TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":    r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV부_약정종료":    r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":      r"TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ 스마트홈":     r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":   r"스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작": r"스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "스마트홈_약정종료": r"스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":   r"스마트홈[\s\S]*?단말[:\s]*([^\n]+)",

    # 공용단말: 줄 머리에서만 매칭
    "공용단말":        r"(?m)^[ \t]*공용단말[:\s]*([^\n]+)",

    "고객희망일":      r"고객희망일[:\s]*([0-9\-]+)"
}

def parse_all_fields(raw_text: str) -> dict:
    """전체 OCR 텍스트에서 모든 패턴을 돌며 값 추출"""
    data = {}
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, raw_text)
        data[field] = m.group(1).strip() if m else None
    return data

# 4) Streamlit UI
st.set_page_config(page_title="OCR → 필드 추출 → 엑셀", layout="wide")
st.title("🧾 이미지 OCR → 지정 필드만 추출 → 엑셀 저장")

uploaded = st.file_uploader("이미지 업로드 (여러 장)", 
                             type=["jpg","jpeg","png"], 
                             accept_multiple_files=True)

if uploaded:
    rows = []
    progress = st.progress(0)
    for i, file in enumerate(uploaded):
        img = Image.open(file).convert("RGB")
        try:
            text = ocr_google_vision(img)
            parsed = parse_all_fields(text)
            parsed["파일명"] = file.name
        except Exception as e:
            parsed = {"파일명": file.name, "오류": str(e)}
        rows.append(parsed)
        progress.progress((i+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "📥 엑셀 다운로드",
        buf.getvalue(),
        file_name="ocr_extracted.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
