import requests

API_KEY = "K89682508288957"  # 여기에 본인의 OCR.space API Key 하드코딩

def extract_text_from_ocr_space(image_bytes):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': API_KEY,
        'isOverlayRequired': False,
        'language': 'kor'
    }
    files = {'filename': image_bytes}

    response = requests.post(url, data=payload, files=files)
    result = response.json()

    if result.get("IsErroredOnProcessing"):
        raise ValueError(result.get("ErrorMessage", ["Unknown error"])[0])

    return result['ParsedResults'][0]['ParsedText']


def parse_ocr_text(text):
    """
    OCR로 추출한 텍스트를 key: value 형태로 파싱하여 dict로 반환
    """
    lines = text.splitlines()
    parsed = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed
