import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import re, pandas as pd, io

# load the JSON you pasted into “Secrets” and build creds
service_account_info = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

FIELD_PATTERNS = {
    # 기본정보
    "이름":             r"이름[:\s]*([가-힣A-Za-z· ]+)",
    "전번":             r"전번[:\s]*([\d\s\-]+)",
    "생년":             r"생년[:\s]*(\d{6,8})",
    "결합":             r"결합[:\s]*([가-힣A-Za-z0-9]+)",
    "주소":             r"주소[:\s]*(.+?)(?=\n)",

    # U+ 인터넷
    "U+ 인터넷":        r"U\+\s*인터넷[:\s]*([0-9]+)",
    "인터넷_요금제":     r"요금제[:\s]*([^\n]+)",
    "인터넷_약정시작":    r"약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "인터넷_약정종료":    r"약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    # 단말은 '단말' 제목 뒤에 오는 텍스트
    "인터넷_단말":      r"U\+\s*인터넷[\s\S]*?단말[:\s]*([^\n]+)",

    # U+ TV (주)
    "U+ TV (주)":       r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "TV주_요금제":       r"U\+\s*TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":     r"U\+\s*TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV주_약정종료":     r"U\+\s*TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":       r"U\+\s*TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",

    # U+ TV (부)
    "U+ TV (부)":       r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":       r"U\+\s*TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":     r"U\+\s*TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV부_약정종료":     r"U\+\s*TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":       r"U\+\s*TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",

    # U+ 스마트홈
    "U+ 스마트홈":      r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":    r"U\+\s*스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작":   r"U\+\s*스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "스마트홈_약정종료":   r"U\+\s*스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":     r"U\+\s*스마트홈[\s\S]*?단말[:\s]*([^\n]+)",

    # 기타
    "공용단말":         r"공용단말[:\s]*([^\n]+)",
    "고객희망일":       r"고객희망일[:\s]*([0-9-]+)"
}

def parse_all_fields(raw_text: str) -> dict:
    """
    OCR로 뽑은 raw_text 에서,
    FIELD_PATTERNS 에 정의된 항목만 꺼내서 dict로 리턴
    """
    out = {}
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, raw_text)
        out[field] = m.group(1).strip() if m else None
    return out

st.title("🧾 Batch OCR → Fields → Excel")
imgs = st.file_uploader("Upload images", type=["jpg","png"], accept_multiple_files=True)
if imgs:
    rows, prog = [], st.progress(0)
    for i, f in enumerate(imgs):
        try:
            txt = ocr_google(f.read())
            data = parse_fields(txt)
            data["파일명"] = f.name
        except Exception as e:
            data = {"파일명": f.name, "오류": str(e)}
        rows.append(data)
        prog.progress((i+1)/len(imgs))
    df = pd.DataFrame(rows)
    st.dataframe(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w, index=False)
    st.download_button("Download Excel", buf.getvalue(),
                       file_name="ocr_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
