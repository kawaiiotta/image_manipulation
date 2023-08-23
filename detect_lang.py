import pytesseract
import re
# from polyglot.detect import Detector
from PIL import Image
import os
from langdetect import detect
import langid

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_image_lang(img_path):
    try:
        osd = pytesseract.image_to_osd(img_path)
        language_info = re.findall(r"Script: ([a-zA-Z]+)(?:\nScript confidence: (\d+\.?(\d+)?))?", osd)
        
        languages = {}
        for script, conf, _ in language_info:
            languages[script] = float(conf) if conf else 0.0
        return languages
    except Exception as e:
        print(e)
        return None
    

def detect_lang_tesseract(img_path):
    # Assuming 'image' is your input image
    # Set the config parameter to enable LSTM language model and language detection
    # custom_config = r'--oem 1 --psm 0'
    # custom_config = r'tesseract -l osd --psm 0 IMAGE -c tessedit_char_whitelist=0123456789'
    custom_config = r'tesseract -l osd --psm 0 IMAGE -'

    image = Image.open(img_path)

    # Perform OCR with the custom config to enable language detection without full text extraction
    detected_languages = pytesseract.image_to_osd(image, config=custom_config)

    # Extract the detected languages and their confidence scores from the output
    languages = {}
    for line in detected_languages.split('\n'):
        if 'Script:' in line:
            script_info = line.split(':')
            language = script_info[1].strip()
        elif 'Script confidence:' in line:
            confidence = float(line.split(':')[1].strip())
            languages[language] = confidence

    print("Detected Languages and Confidence Scores:")
    for language, confidence in languages.items():
        print(f"{language}: {confidence}")


def detect_image_lang_long(img_path):
    lang = pytesseract.get_languages(config='')

    # transfrom lang to have this format "eng+deu+ara+fas+fra+ita+jpn+kor+por+rus+spa+tha+vie"
    lang = "+".join(lang)
    print(lang)

    # ocr the image first time
    text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
    print(text)
    
"""    # Assuming 'image_text' is the text extracted from the image using pytesseract
    detected_languages = set(detect(text))

    # Assuming 'image_text' is the text extracted from the image using pytesseract
    language, confidence = langid.classify(text)

    print(detected_languages)
    # If you want to use a confidence threshold to filter out detections with low confidence
    if confidence >= 0.8:
        detected_languages = {language}
        print(language)
    else:
        detected_languages = set()  # Empty set if confidence is too low

    print(detected_languages)"""
    # for language in Detector(mixed_text).languages:
    #   print(language)


if __name__ == '__main__':
    path = "images/arabic_test/"
    # detect_image_lang_long(path + 'arabic_1.jpg')
    detect_lang_tesseract(path + 'arabic_1.jpg')
    languages = detect_image_lang(path + 'arabic_1.jpg')
    print(languages)
