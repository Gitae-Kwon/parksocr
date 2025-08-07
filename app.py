# app.py

import io
import re
import streamlit as st
import pandas as pd
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# ─── 1) 인증 설정 ─────────────────
service_account_info = st.secrets["gcp_service_account"]
creds  = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

# ─── 2) 전체 OCR 함수 ─────────────────
def ocr_google_vision(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

# ─── 3) parse_header: 전번 위의 마지막 이름 추출 ─────────────────
def parse_header(text: str) -> dict:
    # (1) 전번 레이블 앞까지만 분리
    head_until_phone = text.split("전번", 1)[0]
    # (2) 그 구간에서 모든 '이름:' 값을 찾아, 맨 마지막을 선택
    names = re.findall(r"이름[:\s]*([가-힣A-Za-z· ]+)", head_until_phone)
    name  = names[-1].strip() if names else None

    # (3) 나머지 필드는 전체 텍스트에서 검색
    m_phone  = re.search(r"전번[:\s]*([\d\s\-]+)", text)
    m_birth  = re.search(r"생년[:\s]*(\d{6,8})", text)
    m_bundle = re.search(r"결합[:\s]*([가-힣A-Za-z0-9]+)", text)
    m_addr   = re.search(r"주소[:\s]*(.+?)(?=\n)", text)

    return {
        "이름":   name,
        "전번":   m_phone.group(1).strip()   if m_phone  else None,
        "생년":   m_birth.group(1).strip()   if m_birth  else None,
        "결합":   m_bundle.group(1).strip()  if m_bundle else None,
        "주소":   m_addr.group(1).strip()    if m_addr   else None,
    }

# ─── 4) 기타 필드(인터넷·TV·스마트홈·고객희망일) 파싱 ─────────────────
OTHER_PATTERNS = {
    "U+ 인터넷":      r"U\+\s*인터넷[:\s]*([0-9]+)",
    "인터넷_요금제":   r"요금제[:\s]*([^\n]+)",
    "인터넷_약정시작":  r"약정기간[^(]*\((\d{4}-\d{2}-\d{2})\)",
    "인터넷_약정종료":  r"약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "인터넷_단말":    r"단말[:\s]*([^\n]+)",

    "U+ TV (주)":     r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "TV주_요금제":     r"TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":   r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})\)",
    "TV주_약정종료":   r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":     r"TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ TV (부)":     r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":     r"TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":   r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})\)",
    "TV부_약정종료":   r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":     r"TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ 스마트홈":    r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":  r"스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작":r"스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})\)",
    "스마트홈_약정종료":r"스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":   r"스마트홈[\s\S]*?단말[:\s]*([^\n]+)",

    "고객희망일":     r"고객희망일[:\s]*([0-9\-]+)"
}

def parse_others(text: str) -> dict:
    return {
        k: (m.group(1).strip() if (m := re.search(p, text)) else None)
        for k, p in OTHER_PATTERNS.items()
    }

# ─── 5) 공용단말(하단50%) 추출 ─────────────────
def extract_common_device(img: Image.Image) -> str:
    W, H = img.size
    crop = img.crop((0, H//2, W, H))
    txt  = ocr_google_vision(crop)
    m = re.search(r"WIFI\s*([^\n]+)", txt, re.IGNORECASE)
    return m.group(1).strip() if m else None

# ─── 6) 푸터(신청자명) 추출 ─────────────────
FOOTER_ROI = (0.00, 0.80, 1.00, 1.00)
def ocr_footer(img: Image.Image) -> str:
    W, H = img.size
    crop = img.crop((0, int(FOOTER_ROI[1]*H), W, H))
    return ocr_google_vision(crop)

def parse_footer_name(text: str) -> str:
    m = re.search(r"신청자명/?연락처\s*([가-힣]+)", text)
    return m.group(1).strip() if m else None

# ─── 7) Streamlit UI ─────────────────
st.set_page_config(page_title="OCR 종합 추출", layout="wide")
st.title("📷 OCR → 모든 영역 필드 추출 → 엑셀 저장")

uploaded = st.file_uploader(
    "이미지 업로드 (여러 장)", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

if uploaded:
    rows, prog = [], st.progress(0)
    for i, f in enumerate(uploaded):
        img = Image.open(f).convert("RGB")
        try:
            # 전체 텍스트 OCR
            full_txt = ocr_google_vision(img)

            # 1) 헤더: 이름(전번 앞의 마지막), 전번, 생년, 결합, 주소
            hdr = parse_header(full_txt)

            # 2) 기타
            oth = parse_others(full_txt)

            # 3) 공용단말
            com = extract_common_device(img)

            # 4) 신청자명
            ftr_txt = ocr_footer(img)
            ft_name = parse_footer_name(ftr_txt)

            record = {
                **hdr,
                **oth,
                "공용단말": com,
                "신청자명": ft_name,
                "파일명":   f.name
            }
        except Exception as e:
            record = {"파일명": f.name, "오류": str(e)}

        rows.append(record)
        prog.progress((i+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "📥 엑셀 다운로드",
        data=buf.getvalue(),
        file_name="ocr_all_fields.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
