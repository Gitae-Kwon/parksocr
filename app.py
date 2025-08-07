import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import re, pandas as pd, io

# load the JSON you pasted into “Secrets” and build creds
service_account_info = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=creds)

FIELD_PATTERNS = {
    "이름":      r"이름[:\s]*([가-힣A-Za-z]+)",
    "U+ 인터넷":r"U\+\s*인터넷[:\s]*([0-9]+)",
    # … other patterns …
}

def ocr_google(image_bytes):
    resp = client.document_text_detection(image=vision.Image(content=image_bytes))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text

def parse_fields(text):
    out = {}
    for k, p in FIELD_PATTERNS.items():
        m = re.search(p, text)
        out[k] = m[1].strip() if m else None
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
