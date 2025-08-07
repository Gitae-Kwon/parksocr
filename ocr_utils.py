# ocr_utils.py
import requests

def extract_text_from_ocr_space(image_bytes, api_key):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'isOverlayRequired': False,
        'apikey': K89682508288957,
        'language': 'kor'  # 한국어
    }
    files = {'filename': image_bytes}
    response = requests.post(url, data=payload, files=files)
    result = response.json()
    if result.get("IsErroredOnProcessing"):
        raise ValueError(result.get("ErrorMessage", ["Unknown error"])[0])
    return result['ParsedResults'][0]['ParsedText']
