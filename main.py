# imports
from PIL import Image, ImageDraw

import cv2
import numpy as np
import os


def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def get_polygon_corners(image_path):
    # Load the PNG image using OpenCV with the alpha channel (including transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image has an alpha channel, split the channels and use only the alpha channel as the mask
    if image.shape[2] == 4:
        _, _, _, alpha = cv2.split(image)
        mask = alpha
    else:
        # If the image does not have an alpha channel, create a binary mask based on the white background
        _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

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

def transform_image(input_image_path, corners, a4_corners):
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
    return transformed_image

def detect_document_bounding_box(image_path):
    # Load the PNG image using OpenCV with the alpha channel (including transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image has an alpha channel, split the channels and use only the alpha channel as the mask
    if image.shape[2] == 4:
        _, _, _, alpha = cv2.split(image)
        mask = alpha
    else:
        # If the image does not have an alpha channel, create a binary mask based on the white background
        _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

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

def clean_up_image(image):

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=65)

    # Convert the image to grayscale
    adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    save_image(get_next_file_name(), adjusted)

    # Apply adaptive thresholding to obtain a binary image
    _, binary_image = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations (erosion followed by dilation) to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Apply Gaussian blur to smooth the image
    cleaned_blurr_image = cv2.GaussianBlur(cleaned_image, (3, 3), 0)

    # Convert the binary image back to 3-channel (RGB) for visualization
    cleaned_image_rgb = cv2.cvtColor(cleaned_blurr_image, cv2.COLOR_GRAY2RGB)

    # Find contours in the cleaned binary image
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the paper region
    paper_mask = np.zeros_like(binary_image)

    # Check if a contour pixel is within 10 pixels from the border
    border_range = 5
    height, width = paper_mask.shape
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if x < border_range or x >= width - border_range or y < border_range or y >= height - border_range:
                cv2.drawContours(paper_mask, [contour], -1, 255, thickness=cv2.FILLED)
                break

    # Invert the paper_mask to select the black parts (previously white)
    paper_mask_inverted = cv2.bitwise_not(paper_mask)

    # Apply the inverted paper_mask to the original image to add ONLY the black parts to the cleaned_image
    cleaned_image = cv2.bitwise_and(cleaned_image, cleaned_image, mask=paper_mask_inverted)


    # invert the image and save 
    cleaned_image = cv2.bitwise_not(cleaned_image)
    # Apply Gaussian blur to smooth the image
    cleaned_image = cv2.GaussianBlur(cleaned_image, (3, 3), 0)

    save_image(get_next_file_name(), paper_mask)   
    save_image(get_next_file_name(), cleaned_image)
    return cleaned_image

# quicksave an image
def save_image(save_path, image):
    cv2.imwrite(save_path, image)   

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

def process_all_images(test_image_folder, result_image_folder):
    # List all files in the test_image_folder
    image_files = os.listdir(test_image_folder)

    amount_files = len(image_files)

    for file_name in image_files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            input_image_path = os.path.join(test_image_folder, file_name)
            output_image_name_step_1 = file_name.split('.')[0] + '_unskewed.png'
            output_image_name_step_2 = file_name.split('.')[0] + '_result.png'
            output_image_path_step_1 = os.path.join(result_image_folder, output_image_name_step_1)
            output_image_path_step_2 = os.path.join(result_image_folder, output_image_name_step_2)

            # Get the corners of the skewed image
            corners = get_polygon_corners(input_image_path)
            # Coordinates of the corners of the skewed image (in the order: top-left, top-right, bottom-right, bottom-left)
            skewed_corners = sort_corners(corners)
            # Coordinates of the corners of the A4 size image (in the order: top-left, top-right, bottom-right, bottom-left)
            # pixels at 300 DPI: 2480 x 3508 pixels
            a4_corners = [(0, 0), (2480, 0), (2480, 3508), (0, 3508)]
            a4_corners = sort_corners(a4_corners)

            # Transform and save the image
            transformed_image = transform_image(input_image_path, skewed_corners, a4_corners)
            transformed_image.save(output_image_path_step_1, format='PNG')

            # reload transformed image
            transformed_image = cv2.imread(output_image_path_step_1, cv2.IMREAD_UNCHANGED)

            # Clean up the image
            cleaned_image = clean_up_image(transformed_image)
            save_image(output_image_path_step_2, cleaned_image)
            save_image(get_next_file_name(), transformed_image)

        # print progress
        print("Progress: " + str(image_files.index(file_name) + 1) + "/" + str(amount_files))


if __name__ == '__main__':
    test_image_folder = 'images/test_images'
    result_image_folder = 'images/result_images'

    bounding_box = detect_document_bounding_box(test_image_folder + '/test_image.png')
    print("Bounding Box Coordinates:", bounding_box)
    aspect_ratio = calculate_aspect_ratio(bounding_box)
    print("Aspect Ratio:", aspect_ratio)
    # with this info and a threshhold we can determine the size that the image will most likely have. 
    # we will match them to a hardcoded list of sizes to general sizes (A4, A5, A6, postcard, letter, etc.) in 300 DPI both vertical and horizontal

    if not os.path.exists(result_image_folder):
        os.makedirs(result_image_folder)

    process_all_images(test_image_folder, result_image_folder)
