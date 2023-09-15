from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def get_polygon_corners_path(image_path):
    # Load the PNG image using OpenCV with the alpha channel (including transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image has an alpha channel, split the channels and use only the alpha channel as the mask
    if image.shape[2] == 4:
        _, _, _, alpha = cv2.split(image)
        mask = alpha
    else:
        # If the image does not have an alpha channel, create a binary mask based on the white background
        _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    if len(mask.shape) > 2:
        # Convert the image to grayscale
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the polygon we want to find the corners of
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the polygon by reducing the number of points using the Ramer-Douglas-Peucker algorithm
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert the approximated corners to a list of tuples (x, y)
    corners = [tuple(point[0]) for point in approx_corners]

    return corners

def get_polygon_corners(image):
    # If the image has an alpha channel, split the channels and use only the alpha channel as the mask
    if image.shape[2] == 4:
        _, _, _, alpha = cv2.split(image)
        mask = alpha
    else:
        # If the image does not have an alpha channel, create a binary mask based on the white background
        _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    if len(mask.shape) > 2:
        # Convert the image to grayscale
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the polygon we want to find the corners of
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the polygon by reducing the number of points using the Ramer-Douglas-Peucker algorithm
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert the approximated corners to a list of tuples (x, y)
    corners = [tuple(point[0]) for point in approx_corners]

    return corners

def sort_corners(coordinates):
    # Find the minimum and maximum x and y values
    min_x = min(coordinates, key=lambda c: c[0])[0]
    max_x = max(coordinates, key=lambda c: c[0])[0]
    min_y = min(coordinates, key=lambda c: c[1])[1]
    max_y = max(coordinates, key=lambda c: c[1])[1]

    # Sort the corners based on their relative positions
    top_left = min(coordinates, key=lambda c: c[0] + c[1])
    bottom_left = max(coordinates, key=lambda c: c[1] - c[0])
    bottom_right = max(coordinates, key=lambda c: c[0] + c[1])
    top_right = min(coordinates, key=lambda c: c[1] - c[0])

    return [top_left, bottom_left, bottom_right, top_right]

def transform_image_path(input_image_path, output_image_path, corners, a4_corners):
    # Use find_coeffs function to calculate the transformation matrix
    coeffs = find_coeffs(corners, a4_corners)

    # Determine the output size based on the sorted A4 dimensions
    a4_width = max(a4_corners[1][0], a4_corners[2][0]) - min(a4_corners[0][0], a4_corners[3][0])
    a4_height = max(a4_corners[2][1], a4_corners[3][1]) - min(a4_corners[0][1], a4_corners[1][1])

    # Load the input image
    input_image = Image.open(input_image_path)

    # Apply the affine transformation using the calculated coefficients and the determined output size
    transformed_image = input_image.transform((a4_width, a4_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Save the transformed image to the output path
    transformed_image.save(output_image_path, format='PNG')

def transform_image(image, corners, a4_corners):
    # Use find_coeffs function to calculate the transformation matrix
    coeffs = find_coeffs(corners, a4_corners)

    # Determine the output size based on the sorted A4 dimensions
    a4_width = max(a4_corners[1][0], a4_corners[2][0]) - min(a4_corners[0][0], a4_corners[3][0])
    a4_height = max(a4_corners[2][1], a4_corners[3][1]) - min(a4_corners[0][1], a4_corners[1][1])

    # Apply the affine transformation using the calculated coefficients and the determined output size
    transformed_image = image.transform((a4_width, a4_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    return transformed_image

def detect_document_bounding_box_path(image_path):
    # Load the PNG image using OpenCV with the alpha channel (including transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image has an alpha channel, split the channels and use only the alpha channel as the mask
    if image.shape[2] == 4:
        _, _, _, alpha = cv2.split(image)
        mask = alpha
    else:
        # If the image does not have an alpha channel, create a binary mask based on the white background
        _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    if len(mask.shape) > 2:
        # Convert the image to grayscale
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the document we want to find the bounding box of
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the document
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return the coordinates of the bounding box
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

def detect_document_bounding_box(image):

    # If the image has an alpha channel, split the channels and use only the alpha channel as the mask
    if image.shape[2] == 4:
        _, _, _, alpha = cv2.split(image)
        mask = alpha
    else:
        # If the image does not have an alpha channel, create a binary mask based on the white background
        _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    if len(mask.shape) > 2:
        # Convert the image to grayscale
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the document we want to find the bounding box of
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the document
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return the coordinates of the bounding box
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

def calculate_aspect_ratio(bounding_box):
    # Calculate the width and height of the bounding box
    width = bounding_box[2][0] - bounding_box[0][0]
    height = bounding_box[2][1] - bounding_box[0][1]

    # Calculate the aspect ratio
    aspect_ratio = height / width

    return aspect_ratio

def process_all_images(test_image_folder, result_image_folder):
    # List all files in the test_image_folder
    image_files = os.listdir(test_image_folder)

    for file_name in image_files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            input_image_path = os.path.join(test_image_folder, file_name)
            output_image_name = file_name.split('.')[0] + '_result.png'
            output_image_path = os.path.join(result_image_folder, output_image_name)

            # Get the corners of the skewed image
            corners = get_polygon_corners_path(input_image_path)

            # Coordinates of the corners of the skewed image (in the order: top-left, top-right, bottom-right, bottom-left)
            skewed_corners = corners
            skewed_corners = sort_corners(skewed_corners)

            # Coordinates of the corners of the A4 size image (in the order: top-left, top-right, bottom-right, bottom-left)
            # pixels at 300 DPI: 2480 x 3508 pixels
            a4_corners = [(0, 0), (2480, 0), (2480, 3508), (0, 3508)]
            a4_corners = sort_corners(a4_corners)

            # Transform and save the image
            transform_image_path(input_image_path, output_image_path, skewed_corners, a4_corners)

def fetch_transformed(image):
    # Get the corners of the skewed image
    corners = get_polygon_corners(image)

    # Coordinates of the corners of the skewed image (in the order: top-left, top-right, bottom-right, bottom-left)
    skewed_corners = corners
    skewed_corners = sort_corners(skewed_corners)

    # Coordinates of the corners of the A4 size image (in the order: top-left, top-right, bottom-right, bottom-left)
    # pixels at 300 DPI: 2480 x 3508 pixels
    a4_corners = [(0, 0), (2480, 0), (2480, 3508), (0, 3508)]
    a4_corners = sort_corners(a4_corners)

    # Transform and save the image
    transformed_image = transform_image(image, skewed_corners, a4_corners)

    return transformed_image

# function to read the text from image
def fetch_ocr(image, lang='deu'):
    """
    This function will handle the core OCR processing of images.
    """
    # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = pytesseract.image_to_string(image, lang=lang)
    return text

if __name__ == '__main__':
    test_image_folder = 'images/test_images'
    result_image_folder = 'images/result_images'

    bounding_box = detect_document_bounding_box_path(test_image_folder + '/test_image.png')
    print("Bounding Box Coordinates:", bounding_box)
    aspect_ratio = calculate_aspect_ratio(bounding_box)
    print("Aspect Ratio:", aspect_ratio)
    # with this info and a threshhold we can determine the size that the image will most likely have. 
    # we will match them to a hardcoded list of sizes to general sizes (A4, A5, A6, postcard, letter, etc.) in 300 DPI both vertical and horizontal

    if not os.path.exists(result_image_folder):
        os.makedirs(result_image_folder)

    process_all_images(test_image_folder, result_image_folder)


