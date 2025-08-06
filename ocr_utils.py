# ocr_utils.py
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# PaddleOCR 모델을 전역에서 한 번만 로딩 (속도 & 안정성 ↑)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 텍스트 정제용 오타 보정 사전
corrections = {
    "똑똑케어": "펫케어",
    "WIFI (무료WIFI)가": "WIFI (무료WIFI)기가",
    "기기인터넷": "기가인터넷",
    "무료": "무료",
    "욜정": "요금제"
}

def extract_text(image: Image.Image) -> str:
    image_np = np.array(image)
    result = ocr.ocr(image_np, cls=True)
    lines = [line[1][0] for line in result[0]]

    # 오타 보정
    text = "\n".join(lines)
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    return text
