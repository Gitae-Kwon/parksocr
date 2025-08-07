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

# ─── 3) parse_header: 맨 위 10줄만 보고, 마지막 매칭값 사용 ─────────────────
def parse_header(full_text: str) -> dict:
    # 1) 헤더 영역만: 맨 위 10줄
    lines = full_text.splitlines()[:10]
    header_slice = "\n".join(lines)

    data = {}
    # 이름: 모든 매칭 중 레이블만 걸러내고 마지막
    raw_names = re.findall(r"이름[:：]\s*([^\n]+)", header_slice)
    names = [n.strip() for n in raw_names if n.strip() and n.strip() != "이름"]
    data["이름"] = names[-1] if names else None

    # 전번
    m = re.search(r"전번[:\s]*([\d\s\-]+)", header_slice)
    data["전번"] = m.group(1).strip() if m else None

    # 생년
    m = re.search(r"생년[:\s]*(\d{6,8})", header_slice)
    data["생년"] = m.group(1).strip() if m else None

    # 결합
    m = re.search(r"결합[:\s]*([^\n]+)", header_slice)
    data["결합"] = m.group(1).strip() if m else None

    # 주소
    m = re.search(r"주소[:\s]*(.+?)(?=\n)", header_slice)
    data["주소"] = m.group(1).strip() if m else None

    return data

# ─── 4) 기타 필드 파싱 (인터넷·TV·스마트홈·고객희망일) ─────────────────
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
        label: (m.group(1).strip() if (m := re.search(pat, text)) else None)
        for label, pat in OTHER_PATTERNS.items()
    }

# ─── 5) 공용단말 추출(하단50%) ─────────────────
def extract_common_device(img: Image.Image) -> str:
    W, H = img.size
    crop = img.crop((0, H//2, W, H))
    txt  = ocr_google_vision(crop)
    m = re.search(r"WIFI\s*([^\n]+)", txt, re.IGNORECASE)
    return m.group(1).strip() if m else None

# ─── 6) 푸터 신청자명 추출 ─────────────────
FOOTER_ROI = (0.00, 0.80, 1.00, 1.00)
def ocr_footer(img: Image.Image) -> str:
    W, H = img.size
    crop = img.crop((0, int(FOOTER_ROI[1]*H), W, H))
    return ocr_google_vision(crop)

def parse_footer_name(text: str) -> str:
    m = re.search(r"신청자명/?연락처\s*([가-힣]+)", text)
    return m.group(1).strip() if m else None

# ─── 7) Streamlit UI ─────────────────
st.set_page_config(page_title="OCR 통합 추출", layout="wide")
st.title("📷 OCR → 전체·헤더·하단·푸터 필드 추출 → 엑셀")

uploaded = st.file_uploader(
    "이미지 업로드 (여러 장)", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

if uploaded:
    rows, prog = [], st.progress(0)
    for idx, f in enumerate(uploaded):
        img = Image.open(f).convert("RGB")
        try:
            full_txt = ocr_google_vision(img)

            # 헤더(맨 위 10줄)
            hdr = parse_header(full_txt)
            # 기타
            oth = parse_others(full_txt)
            # 공용단말
            dev = extract_common_device(img)
            # 신청자명 (푸터)
            ftxt = ocr_footer(img)
            fname = parse_footer_name(ftxt)

            record = {
                **hdr,
                **oth,
                "공용단말": dev,
                "신청자명": fname,
                "파일명":   f.name
            }
        except Exception as e:
            record = {"파일명": f.name, "오류": str(e)}

        rows.append(record)
        prog.progress((idx+1)/len(uploaded))

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
