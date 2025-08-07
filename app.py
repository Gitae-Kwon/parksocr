import streamlit as st
import pandas as pd
import io
from ocr_utils import extract_text_from_ocr_space, parse_ocr_text

st.set_page_config(page_title="이미지 OCR → 항목별 정리 → 엑셀 저장", layout="wide")
st.title("🧾 이미지에서 텍스트 추출 → 항목별 정리 → 엑셀 저장")

uploaded = st.file_uploader(
    "📂 이미지 업로드 (여러 개 선택 가능)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded:
    results = []
    prog = st.progress(0)

    for i, file in enumerate(uploaded):
        try:
            img_bytes = file.read()
            raw = extract_text_from_ocr_space(img_bytes)
            parsed = parse_ocr_text(raw)
            parsed["파일명"] = file.name
            results.append(parsed)
        except Exception as e:
            results.append({"파일명": file.name, "오류": str(e)})
        prog.progress((i + 1) / len(uploaded))

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            "📥 엑셀 파일 다운로드",
            data=buf.getvalue(),
            file_name="ocr_parsed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
