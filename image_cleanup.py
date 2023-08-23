import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance


def clean_up_image(image_path, save_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=65)

    # Convert the image to grayscale
    adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # save the gray image
    cv2.imwrite(save_path + "gray.png", adjusted)

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
    cv2.imwrite(save_path + "paper_mask.png", paper_mask)   
    cv2.imwrite(save_path + "cleaned.png", cleaned_image)
    """

    # Combine the cleaned-up image with the original image using the mask
    final_image = cv2.addWeighted(mask, 1, cleaned_image_rgb, 0.5, 0)

    return final_image"""


def process_all_clean_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for image_file in image_files:
        # Create the full paths for input and output images
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # Clean up the image
        cleaned_image = clean_up_image(output_folder + "/" + image_file.split(".")[0] + "_result.png", output_folder + "/" + image_file.split(".")[0] + "_")

        # Save the cleaned-up image
        # cv2.imwrite(output_image_path + "cleaned_result.png", cleaned_image)
        print("image saved")

if __name__ == '__main__':
    input_folder = 'images/test_images'
    output_folder = 'images/result_images'

    process_all_clean_images(input_folder, output_folder)