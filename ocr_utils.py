import requests

# API 키를 여기 하드코딩
API_KEY = "여기에_발급받은_API_키를_입력하세요"

def extract_text_from_ocr_space(image_bytes):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': API_KEY,
        'isOverlayRequired': False,
        'language': 'kor'
    }
    files = {'filename': ('image.jpg', image_bytes)}

    response = requests.post(url, data=payload, files=files)
    result = response.json()

    if result.get("IsErroredOnProcessing"):
        raise ValueError(result.get("ErrorMessage", ["Unknown error"])[0])

    return result['ParsedResults'][0]['ParsedText']


def parse_ocr_text(text):
    """
    OCR 결과 텍스트를 key:value 항목으로 파싱
    예: 이름: 홍길동 → {"이름": "홍길동"}
    """
    lines = text.splitlines()
    parsed = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed
