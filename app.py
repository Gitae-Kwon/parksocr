# app.py

import io, re, streamlit as st, pandas as pd
from google.cloud import vision
from google.oauth2 import service_account

# ──────────── 1) 인증 ────────────
# Secrets UI에 TOML로 등록해 두신 [gcp_service_account] 섹션을
# 그대로 dict 로 꺼내 씁니다.
service_account_info = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

# ──────────── 2) 필드 패턴 ────────────
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
    "TV주_요금제":      r"U\+\s*TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":    r"U\+\s*TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV주_약정종료":    r"U\+\s*TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":       r"U\+\s*TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",
    "U+ TV (부)":      r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":      r"U\+\s*TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":    r"U\+\s*TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV부_약정종료":    r"U\+\s*TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":       r"U\+\s*TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",
    "U+ 스마트홈":     r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":   r"U\+\s*스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작": r"U\+\s*스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "스마트홈_약정종료": r"U\+\s*스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":    r"U\+\s*스마트홈[\s\S]*?단말[:\s]*([^\n]+)",
    "공용단말":         r"(?m)^[ \t]*공용단말[:\s]*([^\n]+)",
    "고객희망일":       r"고객희망일[:\s]*([0-9\-]+)"
}

def ocr_google_vision(image_bytes: bytes) -> str:
    image = vision.Image(content=image_bytes)
    resp  = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

def parse_all_fields(raw_text: str) -> dict:
    data = {}
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, raw_text)
        data[field] = m.group(1).strip() if m else None
    return data

# ──────────── Streamlit UI ────────────
st.set_page_config(page_title="OCR → Fields → Excel", layout="wide")
st.title("🧾 OCR → 항목별 추출 → 엑셀 저장")

uploaded = st.file_uploader("이미지 업로드 (여러 장)", type=["jpg","png","jpeg"], accept_multiple_files=True)
if uploaded:
    rows, prog = [], st.progress(0)
    for i, file in enumerate(uploaded):
        try:
            raw    = ocr_google_vision(file.read())
            parsed = parse_all_fields(raw)
            parsed["파일명"] = file.name
        except Exception as e:
            parsed = {"파일명": file.name, "오류": str(e)}
        rows.append(parsed)
        prog.progress((i+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 엑셀 다운로드", buf.getvalue(),
                       file_name="ocr_fields.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
