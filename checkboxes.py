import cv2
import os

# Load image, convert to grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('images/result_images_2/test_image2_cleaned.png')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours and filter using contour area filtering to remove noise
cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
AREA_THRESHOLD = 10
for c in cnts:
    area = cv2.contourArea(c)
    if area < AREA_THRESHOLD:
        cv2.drawContours(thresh, [c], -1, 0, -1)

# Repair checkbox horizontal and vertical walls
repair_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
repair = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, repair_kernel1, iterations=1)
repair_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, repair_kernel2, iterations=1)

# Detect checkboxes using shape approximation and aspect ratio filtering
checkbox_contours = []
cnts, _ = cv2.findContours(repair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.035 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    if len(approx) == 4 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2):
        cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 3)
        checkbox_contours.append(c)

print('Checkboxes:', len(checkbox_contours))
# cv2.imshow('thresh', thresh)
# cv2.imshow('repair', repair)
# cv2.imshow('original', original)
# cv2.waitKey()

# function that will see all files in the folder and returns the name of the latest file + 1 (qcksve_1.png, qcksve_2.png, etc.)
def get_next_file_name(folder_path="images/quicksaves"):
    # get all files in the folder
    files = os.listdir(folder_path)
    # get the last file name
    try:
        last_file = max(files, key=lambda x: int(x.split(".")[0].split("_")[1]))
    except ValueError:
        last_file = "qcksve_0.png"
    # get the number of the last file
    last_file_number = int(last_file.split(".")[0].split("_")[1])
    # return the next file name
    return "images/quicksaves/qcksve_" + str(last_file_number + 1) + ".png"

# save the three to images/quicksaves
cv2.imwrite(get_next_file_name(), thresh)
cv2.imwrite(get_next_file_name(), repair)
cv2.imwrite(get_next_file_name(), original)
