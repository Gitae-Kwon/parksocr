# app.py

import io, re, streamlit as st, pandas as pd
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# 1) 인증
svc_info = st.secrets["gcp_service_account"]
creds    = service_account.Credentials.from_service_account_info(svc_info)
client   = vision.ImageAnnotatorClient(credentials=creds)

# 2) 전역 OCR (전체 이미지)
def ocr_google_vision(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

# 3) 패턴 정의 (공용단말 포함)
FIELD_PATTERNS = {
    "이름":      r"이름[:\s]*([가-힣A-Za-z· ]+)",
    "전번":      r"전번[:\s]*([\d\s\-]+)",
    "생년":      r"생년[:\s]*(\d{6,8})",
    "결합":      r"결합[:\s]*([가-힣A-Za-z0-9]+)",
    "주소":      r"주소[:\s]*(.+?)(?=\n)",

    "U+ 인터넷":     r"U\+\s*인터넷[:\s]*([0-9]+)",
    "인터넷_요금제":  r"요금제[:\s]*([^\n]+)",
    "인터넷_약정시작": r"약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "인터넷_약정종료": r"약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "인터넷_단말":   r"단말[:\s]*([^\n]+)",

    "U+ TV (주)":    r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "TV주_요금제":    r"TV\s*\(주\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV주_약정시작":  r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV주_약정종료":  r"TV\s*\(주\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV주_단말":    r"TV\s*\(주\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ TV (부)":    r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "TV부_요금제":    r"TV\s*\(부\)[\s\S]*?요금제[:\s]*([^\n]+)",
    "TV부_약정시작":  r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "TV부_약정종료":  r"TV\s*\(부\)[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "TV부_단말":    r"TV\s*\(부\)[\s\S]*?단말[:\s]*([^\n]+)",

    "U+ 스마트홈":   r"U\+\s*스마트홈[:\s]*([0-9]+)",
    "스마트홈_요금제":  r"스마트홈[\s\S]*?요금제[:\s]*([^\n]+)",
    "스마트홈_약정시작":r"스마트홈[\s\S]*?약정기간[^(]*\((\d{4}-\d{2}-\d{2})",
    "스마트홈_약정종료":r"스마트홈[\s\S]*?약정기간[^(]*\(\d{4}-\d{2}-\d{2}~(\d{4}-\d{2}-\d{2})\)",
    "스마트홈_단말":  r"스마트홈[\s\S]*?단말[:\s]*([^\n]+)",

    # 공용단말은 아래 함수에서 별도 처리
    "고객희망일":    r"고객희망일[:\s]*([0-9\-]+)"
}

def parse_all_fields(text: str) -> dict:
    data = {}
    # 전체 텍스트에서 공통 필드들 뽑기
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, text)
        data[field] = m.group(1).strip() if m else None
    return data

#
# 4) 공용단말만 '이미지의 하단 절반'에서 OCR→파싱
#
def extract_common_device(img: Image.Image) -> str:
    W, H = img.size
    # 하단 50% 영역만 crop
    crop = img.crop((0, H//2, W, H))
    txt  = ocr_google_vision(crop)
    # 공용단말 패턴: 줄 머리에서만 매칭
    m = re.search(r"(?m)^[ \t]*공용단말[:\s]*([^\n]+)", txt)
    return m.group(1).strip() if m else None

# ──────────── Streamlit UI ────────────
st.set_page_config(page_title="OCR → 필드 추출 → 엑셀", layout="wide")
st.title("📷 이미지 OCR → 지정 필드만 추출 → 엑셀 저장")

uploaded = st.file_uploader("이미지 업로드 (여러 장)", type=["jpg","jpeg","png"], accept_multiple_files=True)
if uploaded:
    rows, prog = [], st.progress(0)
    for i, f in enumerate(uploaded):
        img = Image.open(f).convert("RGB")
        try:
            raw_text = ocr_google_vision(img)
            parsed   = parse_all_fields(raw_text)
            # 공용단말만 하단 50%에서 재추출
            parsed["공용단말"] = extract_common_device(img)
            parsed["파일명"]   = f.name
        except Exception as e:
            parsed = {"파일명": f.name, "오류": str(e)}
        rows.append(parsed)
        prog.progress((i+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 엑셀 다운로드", buf.getvalue(),
                       file_name="ocr_extracted.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
