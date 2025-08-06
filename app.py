# app.py
import streamlit as st
from ocr_utils import extract_text
from PIL import Image
import pandas as pd
import io

st.set_page_config(page_title="이미지 OCR → 엑셀 변환기", layout="wide")
st.title("🧾 이미지에서 텍스트 추출 → 엑셀 저장")

uploaded_files = st.file_uploader("📂 이미지 업로드 (다중 선택 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []

    progress = st.progress(0, text="OCR 처리 중...")

    for idx, file in enumerate(uploaded_files):
        try:
            image = Image.open(file).convert("RGB")
            text = extract_text(image)
            results.append({"파일명": file.name, "OCR결과": text})
        except Exception as e:
            results.append({"파일명": file.name, "OCR결과": f"[ERROR] {str(e)}"})
        progress.progress((idx + 1) / len(uploaded_files), text=f"{idx + 1} / {len(uploaded_files)} 처리 중...")

    df = pd.DataFrame(results)

    st.success("✅ 모든 이미지 처리 완료!")
    st.dataframe(df)

    # 엑셀 다운로드
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button(
        label="📥 엑셀 파일 다운로드",
        data=excel_buffer.getvalue(),
        file_name="ocr_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
