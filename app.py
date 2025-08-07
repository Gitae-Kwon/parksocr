# app.py

import io
import re
import streamlit as st
import pandas as pd
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# 1️⃣ 인증: Streamlit Secrets에 등록한 서비스 계정 JSON을 dict로 꺼내기
service_account_info = st.secrets["gcp_service_account"]
creds  = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

# 2️⃣ OCR 함수: PIL Image → Google Vision → 전체 텍스트 반환
def ocr_google_vision(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

# 3️⃣ 헤더(이름·전번·생년·결합·주소) 영역 OCR
HEADER_ROI = (0.60, 0.00, 1.00, 0.30)  # (left, top, right, bottom) 비율
def ocr_header(img: Image.Image) -> str:
    W, H = img.size
    l, t, r, b = int(HEADER_ROI[0]*W), int(HEADER_ROI[1]*H), int(HEADER_ROI[2]*W), int(HEADER_ROI[3]*H)
    crop = img.crop((l, t, r, b))
    return ocr_google_vision(crop)

# 4️⃣ 하단 50% 영역에서 공용단말 정보 추출
def extract_common_device(img: Image.Image) -> str:
    W, H = img.size
    crop = img.crop((0, H//2, W, H))
    text = ocr_google_vision(crop)
    m = re.search(r"WIFI\s*([^\n]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None

# 5️⃣ 정규표현식 패턴 정의
HEADER_PATTERNS = {
    "이름": r"이름[:\s]*([가-힣A-Za-z· ]+)",
    "전번": r"전번[:\s]*([\d\s\-]+)",
    "생년": r"생년[:\s]*(\d{6,8})",
    "결합": r"결합[:\s]*([가-힣A-Za-z0-9]+)",
    "주소": r"주소[:\s]*(.+?)(?=\n)",
}

OTHER_PATTERNS = {
    "U+ 인터넷":      r"U\+\s*인터넷[:\s]*([0-9]+)",
    "인터넷_요금제":   r"요금제[:\s]*([^\n]+)",
    "인터넷_약정시작":  r"약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "인터넷_약정종료":  r"약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "인터넷_단말":    r"단말[:\s]*([^\n]+)",

    "U+ TV (주)":     r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "TV주_요금제":     r"TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":   r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV주_약정종료":   r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":     r"TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ TV (부)":     r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":     r"TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":   r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV부_약정종료":   r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":     r"TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ 스마트홈":    r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":  r"스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작":r"스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "스마트홈_약정종료":r"스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":   r"스마트홈[\s\S]*?단말[:\s]*([^\n]+)",

    "고객희망일":     r"고객희망일[:\s]*([0-9\-]+)"
}

def parse_header(text: str) -> dict:
    return {
        k: (m.group(1).strip() if (m:=re.search(p, text)) else None)
        for k,p in HEADER_PATTERNS.items()
    }

def parse_others(text: str) -> dict:
    return {
        k: (m.group(1).strip() if (m:=re.search(p, text)) else None)
        for k,p in OTHER_PATTERNS.items()
    }

# ───── Streamlit UI ─────
st.set_page_config(page_title="OCR 영역별+전체 필드 추출", layout="wide")
st.title("📷 OCR → 헤더·전체·하단 영역별 필드 추출 → 엑셀")

files = st.file_uploader("이미지 업로드 (여러 장)", type=["jpg","jpeg","png"], accept_multiple_files=True)
if files:
    rows, prog = [], st.progress(0)
    for i,f in enumerate(files):
        img = Image.open(f).convert("RGB")
        try:
            header_txt  = ocr_header(img)
            header_data = parse_header(header_txt)

            full_txt   = ocr_google_vision(img)
            other_data = parse_others(full_txt)

            common_dev = extract_common_device(img)

            record = {
                **header_data,
                **other_data,
                "공용단말": common_dev,
                "파일명": f.name
            }
        except Exception as e:
            record = {"파일명": f.name, "오류": str(e)}

        rows.append(record)
        prog.progress((i+1)/len(files))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 엑셀 다운로드", buf.getvalue(),
                       file_name="ocr_result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
