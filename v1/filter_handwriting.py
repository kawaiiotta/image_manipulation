import cv2
import numpy as np
import os


import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary_image

def detect_text_contours(binary_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and aspect ratio
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 100 < w < 3000 and 1 < aspect_ratio < 10:  # Adjust these values based on your use case
            filtered_contours.append(contour)

    return filtered_contours

def classify_text_contours(contours, model):
    # You can use a pre-trained model or build a custom classifier
    # For example, you can use scikit-learn or train a deep learning model

    # Example using scikit-learn:
    # X = extract_features_from_contours(contours)
    # predictions = model.predict(X)

    # Example using a deep learning model (requires model loading and preprocessing)
    # predictions = model.predict(contours)

    # Replace the above line with the appropriate classifier or model prediction
    # For this example, let's assume the result is a list of boolean values representing
    # whether each contour is handwritten or not.
    predictions = [True] * len(contours)

    return predictions

def main(path):
    binary_image = preprocess_image(path)
    contours = detect_text_contours(binary_image)

    # Load your classifier or pre-trained model
    # classifier = load_classifier_or_model()

    # Classify contours
    # predictions = classify_text_contours(contours, classifier)

    # For the example, we'll assume everything is handwritten (True) for demonstration purposes
    predictions = [True] * len(contours)

    # create black background of same image shape
    img = cv2.imread(path)
    black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # Now, you have the list of boolean values indicating whether each contour is handwritten or not
    for i, contour in enumerate(contours):
        if predictions[i]:
            # Handle handwritten text
            # save contour
            black = cv2.drawContours(black,[contour],0,(255,255,255),2)



def split_handwriting(path):
    # read image, convert to grayscale and apply Otsu threshold
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # create black background of same image shape
    black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # find contours from threshold image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # draw contours whose area is above certain value
    # mean of all contour areas
    mean_contours = np.mean([cv2.contourArea(c) for c in contours])
    print(mean_contours)
    area_threshold = mean_contours
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_threshold:
            black = cv2.drawContours(black,[c],0,(255,255,255),2)

    # save file to /images/quicksaves/
    save_image(get_next_file_name(), black)

def save_image(path, image):
    cv2.imwrite(path, image)

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


def parce_all_images(path):
    for filename in os.listdir(path):
        if filename.endswith("result.png"):
            split_handwriting(path + filename)


if __name__ == '__main__':
    path = "images/result_images/"
    parce_all_images(path)