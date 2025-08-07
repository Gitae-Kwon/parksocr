import streamlit as st
from ocr_utils import extract_text_from_ocr_space
from PIL import Image
import pandas as pd
import io

st.set_page_config(page_title="이미지 OCR → 엑셀 변환기", layout="wide")
st.title("🧾 이미지에서 텍스트 추출 → 엑셀 저장")

api_key = st.text_input("K89682508288957", type="password")

uploaded_files = st.file_uploader("📂 이미지 업로드 (여러 개 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and api_key:
    results = []
    progress = st.progress(0, text="OCR 처리 중...")

    for idx, file in enumerate(uploaded_files):
        try:
            image = Image.open(file).convert("RGB")
            image_bytes_io = io.BytesIO()
            image.save(image_bytes_io, format='JPEG')
            image_bytes = image_bytes_io.getvalue()

            text = extract_text_from_ocr_space(image_bytes, api_key)
            results.append({"파일명": file.name, "OCR 결과": text})
        except Exception as e:
            results.append({"파일명": file.name, "OCR 결과": f"❌ 오류: {str(e)}"})

        progress.progress((idx + 1) / len(uploaded_files), text="OCR 처리 중...")

    df = pd.DataFrame(results)
    st.dataframe(df)

    excel_bytes = io.BytesIO()
    with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 엑셀 파일 다운로드", data=excel_bytes.getvalue(), file_name="ocr_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
