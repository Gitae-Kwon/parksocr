# app.py
import streamlit as st
import pandas as pd
import io
from ocr_utils import extract_text_from_ocr_space

st.set_page_config(page_title="OCR.space 기반 이미지 → 엑셀", layout="wide")
st.title("🧾 이미지에서 텍스트 추출 → 엑셀 변환")

api_key = st.text_input("🔑 OCR.space API Key를 입력하세요", type="password")

uploaded_files = st.file_uploader("📁 이미지 업로드 (여러 개 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and api_key:
    results = []

    with st.spinner("🔍 OCR 처리 중..."):
        for file in uploaded_files:
            try:
                text = extract_text_from_ocr_space(file.read(), api_key)
                results.append({"파일명": file.name, "OCR 결과": text})
            except Exception as e:
                results.append({"파일명": file.name, "OCR 결과": f"❌ 오류: {str(e)}"})

    df = pd.DataFrame(results)
    st.dataframe(df)

    # 엑셀 다운로드
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 엑셀 파일 다운로드", data=towrite.getvalue(), file_name="ocr_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
