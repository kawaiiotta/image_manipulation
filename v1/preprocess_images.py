import os
import cv2
import numpy as np
from tqdm import tqdm

def remove_noise(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:  # Check if the image is not grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Make sure the image is of type CV_8UC1
    if image.dtype != 'uint8':
        image = image.astype('uint8')

    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image

def scale_image(image, target_size):
    scaled_image = cv2.resize(image, (target_size, target_size))
    return scaled_image

def contour_image(image):
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 0, 0), 1)
    return contour_image

def process_images_in_folders(input_folders):
    total_images = sum(len(files) for folder in input_folders for _, _, files in os.walk(folder))
    progress_bar = tqdm(total=total_images, desc="Processing images", unit="image")

    for folder in input_folders:
        for root, _, files in os.walk(folder):
            # find the smallest size of all images
            target_size = min([cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED).shape[0] for file in files])
            # make target_size 28 if it is less
            if int(target_size) < 28:
                target_size = 28
            print(type(target_size))
            for file in files:
                input_path = os.path.join(root, file)

                try:
                    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

                    # Remove noise
                    denoised_image = remove_noise(image)

                    # Scale image
                    scaled_image = scale_image(denoised_image, target_size)

                    # Create contour image
                    contour_image_result = contour_image(scaled_image)

                    # Save the processed images (overwriting existing files)
                    cv2.imwrite(input_path, contour_image_result)

                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f"Processed: {progress_bar.n}/{progress_bar.total}")

                except Exception as e:
                    # Delete the original image if an error occurs
                    os.remove(input_path)
                    print(f"Error processing image: {input_path}")
                    print(f"Error message: {str(e)}")
                    continue
            
    progress_bar.close()

if __name__ == "__main__":
    input_folders_list = ["images/handwritten/","images/printed/", "images/signatures/"]
    process_images_in_folders(input_folders_list)