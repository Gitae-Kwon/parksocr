import requests

API_KEY = "K89682508288957"

def extract_text_from_ocr_space(image_bytes):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': API_KEY,
        'isOverlayRequired': False,
        'language': 'kor',
    }
    # 파일 파라미터를 'file' 키로, (이름, 바이트, MIME) 튜플 형식으로 넘겨야 인식됩니다
    files = {
        'file': ('image.jpg', image_bytes, 'image/jpeg')
    }

    response = requests.post(url, data=payload, files=files)
    result = response.json()

    if result.get("IsErroredOnProcessing"):
        raise ValueError(result.get("ErrorMessage", ["Unknown error"])[0])

    return result['ParsedResults'][0]['ParsedText']


def parse_ocr_text(text):
    """
    OCR 로 출력된 긴 텍스트를
    '키:값' 라인별로 분리하여 dict 반환
    """
    parsed = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            parsed[key.strip()] = val.strip()
    return parsed
