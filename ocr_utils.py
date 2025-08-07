import requests
import re

API_KEY = "여기에_발급받은_API_키_문자열_형태로_삽입"

def extract_text_from_ocr_space(image_bytes):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': API_KEY,
        'isOverlayRequired': False,
        'language': 'kor',
    }
    files = {
        'file': ('image.jpg', image_bytes, 'image/jpeg')
    }

    res = requests.post(url, data=payload, files=files)
    result = res.json()
    if result.get("IsErroredOnProcessing"):
        raise ValueError(result["ErrorMessage"][0])
    return result['ParsedResults'][0]['ParsedText']

# 여기서 뽑을 항목과 패턴을 딕셔너리에 정의
FIELD_PATTERNS = {
    "이름":         r"이름[:\s]*([가-힣A-Za-z]+)",
    "U+ 인터넷":   r"U\+\s*인터넷[:\s]*([0-9]+)",
    "U+ TV (주)":   r"U\+\s*TV\s*\(주\)[:\s]*([0-9]+)",
    "U+ TV (부)":   r"U\+\s*TV\s*\(부\)[:\s]*([0-9]+)",
    "U+ 스마트홈":  r"U\+\s*스마트홈[:\s]*([0-9]+)",
    # 필요하다면 여기에 더 추가...
}

def parse_specified_fields(text):
    """
    미리 정의한 FIELD_PATTERNS에서 지정한 항목만 뽑아 dict 반환.
    """
    data = {}
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, text)
        data[field] = m.group(1).strip() if m else None
    return data
