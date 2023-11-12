from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# function to read the text from image
def ocr_core(filepath, lang='deu'):
    """
    This function will handle the core OCR processing of images.
    """
    # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = pytesseract.image_to_string(Image.open(filepath), lang=lang)
    return text


# go through each file
def ocr_core_folder(path):
    
    # go through each result file in "images/result_images/" that has this format: "test_image_result.png", "test_image2_result.png", "test_image3_result.png"
    for filename in os.listdir(path):
        if filename.endswith("result.png"):
            print(filename)
            print(ocr_core(path + filename, lang="deu+fra"))


if __name__ == '__main__':
    # path
    path = "images/result_images/"

    # print(ocr_core(path + 'arabic_1.jpg', lang="deu+ara"))
    ocr_core_folder(path)
