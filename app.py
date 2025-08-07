import streamlit as st
import pandas as pd
import io
from ocr_utils import extract_text_from_ocr_space, parse_specified_fields

st.set_page_config(page_title="이미지 OCR → 지정 필드 추출 → 엑셀", layout="wide")
st.title("🧾 지정 필드만 뽑아내는 OCR → 엑셀 변환기")

uploaded = st.file_uploader("📂 이미지 업로드 (여러 개 선택 가능)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    rows = []
    prog = st.progress(0)

    for i, file in enumerate(uploaded):
        try:
            img_bytes = file.read()
            raw = extract_text_from_ocr_space(img_bytes)
            parsed = parse_specified_fields(raw)
            parsed["파일명"] = file.name
            rows.append(parsed)
        except Exception as e:
            rows.append({"파일명": file.name, "오류": str(e)})
        prog.progress((i+1)/len(uploaded))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, index=False)
        st.download_button(
            "📥 엑셀 다운로드",
            data=buf.getvalue(),
            file_name="ocr_fields.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
