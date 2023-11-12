
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import re

"""
    print("got images")
    for i, image in enumerate(images):
        clean_up_image(image, printed_letters_directory+"/"+image_names[i], image_types[i])
        # delete original image
        os.remove(printed_letters_directory+"/"+image_names[i]+"."+image_types[i])

    images, image_names, image_types = get_images(handwritten_letters_directory)
    for i, image in enumerate(images):
        clean_up_image(image, handwritten_letters_directory+"/"+image_names[i], image_types[i])
        # delete original image
        os.remove(handwritten_letters_directory+"/"+image_names[i]+"."+image_types[i])
        
            # clean up images
    images, image_names, image_types = get_images(printed_letters_directory)
    # go through each image and find the amount of white pixels in them, if there are less than 5 remove the image
    for i, image in enumerate(images):
        white_pixels = cv2.countNonZero(image)
        if white_pixels < 5:
            os.remove(f"{printed_letters_directory}/{image_names[i]}.{image_types[i]}")

    images, image_names, image_types = get_images(handwritten_letters_directory)
    # go through each image and find the amount of white pixels in them, if there are less than 5 remove the image
    for i, image in enumerate(images):
        white_pixels = cv2.countNonZero(image)
        if white_pixels < 5:
            os.remove(f"{handwritten_letters_directory}/{image_names[i]}.{image_types[i]}")
"""
def clean_up_image(image, image_name, image_type):
    # find all contours in the image and save each contour as an image in the folder
    # Apply adaptive thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # shorten it to max 10 contours
    contours = contours[:10]
    print(len(contours))
    
    for i, contour in enumerate(contours):
        # create a blank image
        contour_image = np.zeros(image.shape, dtype=np.uint8)
        # draw the contour on the blank image
        cv2.drawContours(contour_image, [contour], 0,(255,255,255),1)
        # save the image
        cv2.imwrite(f"{image_name}_contour_{i}.{image_type}", contour_image)

def create_perspective_transform_matrix(src, dst):
    # ... (The implementation of the create_perspective_transform_matrix function)
    """ Creates a perspective transformation matrix which transforms points
        in quadrilateral ``src`` to the corresponding points on quadrilateral
        ``dst``.

        Will raise a ``np.linalg.LinAlgError`` on invalid input.
        """
    # See:
    # * http://xenia.media.mit.edu/~cwren/interpolator/
    # * http://stackoverflow.com/a/14178717/71522
    in_matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3))

def create_perspective_transform(src, dst, round=False, splat_args=False):
    # ... (The implementation of the create_perspective_transform function)
    """ Returns a function which will transform points in quadrilateral
        ``src`` to the corresponding points on quadrilateral ``dst``::

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ... )
            >>> transform((5, 5))
            (74.99999999999639, 74.999999999999957)

        If ``round`` is ``True`` then points will be rounded to the nearest
        integer and integer values will be returned.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     round=True,
            ... )
            >>> transform((5, 5))
            (75, 75)

        If ``splat_args`` is ``True`` the function will accept two arguments
        instead of a tuple.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     splat_args=True,
            ... )
            >>> transform(5, 5)
            (74.99999999999639, 74.999999999999957)

        If the input values yield an invalid transformation matrix an identity
        function will be returned and the ``error`` attribute will be set to a
        description of the error::

            >>> tranform = create_perspective_transform(
            ...     np.zeros((4, 2)),
            ...     np.zeros((4, 2)),
            ... )
            >>> transform((5, 5))
            (5.0, 5.0)
            >>> transform.error
            'invalid input quads (...): Singular matrix
        """
    try:
        transform_matrix = create_perspective_transform_matrix(src, dst)
        error = None
    except np.linalg.LinAlgError as e:
        transform_matrix = np.identity(3, dtype=np.float)
        error = "invalid input quads (%s and %s): %s" %(src, dst, e)
        error = error.replace("\n", "")

    # Modified section to work in Python 3.x
    to_eval = "def perspective_transform(%s):\n" % (
        splat_args and "*pt" or "pt",
    )
    to_eval += "  res = np.dot(transform_matrix, np.array(((pt[0], ), (pt[1], ), (1,))))\n"
    to_eval += "  res = res / res[2]\n"
    if round:
        to_eval += "  return (int(round(res[0][0])), int(round(res[1][0])))\n"
    else:
        to_eval += "  return (res[0][0], res[1][0])\n"
    locals_dict = {
        "transform_matrix": transform_matrix,
    }
    globals_dict = globals()
    exec(to_eval, globals_dict, locals_dict)
    res = locals_dict["perspective_transform"]
    res.matrix = transform_matrix
    res.error = error
    return res

def transform_image(input_image_path, output_image_path, corners, a4_corners):
    # Use create_perspective_transform function to calculate the transformation
    perspective_transform = create_perspective_transform(corners, a4_corners)

    # Load the input image
    input_image = Image.open(input_image_path)

    # Apply the perspective transformation using the calculated transform function
    transformed_image = input_image.transform(input_image.size, Image.PERSPECTIVE, perspective_transform.matrix.flatten(), Image.BICUBIC)

    # Save the transformed image to the output path
    transformed_image.save(output_image_path, format='PNG')



"""def transform_image(input_image_path, output_image_path, corners, a4_corners):
    # Use find_coeffs function to calculate the transformation matrix
    coeffs = find_coeffs(corners, a4_corners)

    # Determine the output size based on the sorted A4 dimensions
    a4_width = max(a4_corners[1][0], a4_corners[2][0]) - min(a4_corners[0][0], a4_corners[3][0])
    a4_height = max(a4_corners[2][1], a4_corners[3][1]) - min(a4_corners[0][1], a4_corners[1][1])
    print(a4_width, a4_height)

    # determine the output size based on the sorted A4 dimensions
    # a4_width = abs(a4_corners[0][0] - a4_corners[-1][0])
    # a4_height = abs(a4_corners[0][1] - a4_corners[1][1])

    # Load the input image
    input_image = Image.open(input_image_path)

    # Apply the affine transformation using the calculated coefficients and the determined output size
    transformed_image = input_image.transform((a4_width, a4_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Save the transformed image to the output path
    transformed_image.save(output_image_path, format='PNG')"""

def create_perspective_transform_matrix(src, dst):
    # ... (The implementation of the create_perspective_transform_matrix function)
    """ Creates a perspective transformation matrix which transforms points
        in quadrilateral ``src`` to the corresponding points on quadrilateral
        ``dst``.

        Will raise a ``np.linalg.LinAlgError`` on invalid input.
        """
    # See:
    # * http://xenia.media.mit.edu/~cwren/interpolator/
    # * http://stackoverflow.com/a/14178717/71522
    in_matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3))

def create_perspective_transform(src, dst, round=False, splat_args=False):
    # ... (The implementation of the create_perspective_transform function)
    """ Returns a function which will transform points in quadrilateral
        ``src`` to the corresponding points on quadrilateral ``dst``::

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ... )
            >>> transform((5, 5))
            (74.99999999999639, 74.999999999999957)

        If ``round`` is ``True`` then points will be rounded to the nearest
        integer and integer values will be returned.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     round=True,
            ... )
            >>> transform((5, 5))
            (75, 75)

        If ``splat_args`` is ``True`` the function will accept two arguments
        instead of a tuple.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     splat_args=True,
            ... )
            >>> transform(5, 5)
            (74.99999999999639, 74.999999999999957)

        If the input values yield an invalid transformation matrix an identity
        function will be returned and the ``error`` attribute will be set to a
        description of the error::

            >>> tranform = create_perspective_transform(
            ...     np.zeros((4, 2)),
            ...     np.zeros((4, 2)),
            ... )
            >>> transform((5, 5))
            (5.0, 5.0)
            >>> transform.error
            'invalid input quads (...): Singular matrix
        """
    try:
        transform_matrix = create_perspective_transform_matrix(src, dst)
        error = None
    except np.linalg.LinAlgError as e:
        transform_matrix = np.identity(3, dtype=np.float)
        error = "invalid input quads (%s and %s): %s" %(src, dst, e)
        error = error.replace("\n", "")

    # Modified section to work in Python 3.x
    to_eval = "def perspective_transform(%s):\n" % (
        splat_args and "*pt" or "pt",
    )
    to_eval += "  res = np.dot(transform_matrix, np.array(((pt[0], ), (pt[1], ), (1,))))\n"
    to_eval += "  res = res / res[2]\n"
    if round:
        to_eval += "  return (int(round(res[0][0])), int(round(res[1][0])))\n"
    else:
        to_eval += "  return (res[0][0], res[1][0])\n"
    locals_dict = {
        "transform_matrix": transform_matrix,
    }
    globals_dict = globals()
    exec(to_eval, globals_dict, locals_dict)
    res = locals_dict["perspective_transform"]
    res.matrix = transform_matrix
    res.error = error
    return res

"""def transform_image(input_image_path, output_image_path, corners, a4_corners):
    # Use create_perspective_transform_matrix function to calculate the transformation
    perspective_transform_matrix = create_perspective_transform_matrix(corners, a4_corners)

    # Load the input image
    input_image = Image.open(input_image_path)
    input_width, input_height = input_image.size

    # Create a new image with A4 dimensions
    # Determine the output size based on the sorted A4 dimensions
    a4_width = max(a4_corners[1][0], a4_corners[2][0]) - min(a4_corners[0][0], a4_corners[3][0])
    a4_height = max(a4_corners[2][1], a4_corners[3][1]) - min(a4_corners[0][1], a4_corners[1][1])
    
    output_image = Image.new('RGBA', (a4_width, a4_height), (255, 255, 255, 0))

    # Invert the perspective_transform_matrix to get the inverse transformation
    inverse_transform_matrix = np.linalg.inv(perspective_transform_matrix)

    # Iterate through each pixel in the output image and find its corresponding position in the input image
    for out_y in range(a4_height):
        for out_x in range(a4_width):
            # Apply the inverse transformation to find the corresponding position in the input image
            in_x, in_y, _ = np.dot(inverse_transform_matrix, (out_x, out_y, 1))
            in_x = int(in_x / in_y)
            in_y = int(in_y / in_y)

            # Check if the calculated position is within the input image bounds
            if 0 <= in_x < input_width and 0 <= in_y < input_height:
                # Get the pixel value from the input image and paste it onto the output image
                pixel_value = input_image.getpixel((in_x, in_y))
                output_image.putpixel((out_x, out_y), pixel_value)
                print("Pixel value: ", pixel_value)
                print("Out x: ", out_x, "Out y: ", out_y)

    # Save the transformed image to the output path
    output_image.save(output_image_path, format='PNG')"""


def find_coeffs(pa, pb):
    # ... (The implementation of the find_coeffs function from the previous code)
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# function that will see all files in the folder and returns the name of the latest file + 1 (qcksve_1.png, qcksve_2.png, etc.)
def get_next_file_name(folder_path="images/quicksaves"):
    # get all files in the folder
    files = os.listdir(folder_path)
    # get the last file name
    last_file = files[-1]
    # get the number of the last file
    last_file_number = int(last_file.split(".")[0].split("_")[1])
    # return the next file name
    return "qcksve_" + str(last_file_number + 1)

print(get_next_file_name())


def detect_image_lang(img_path):
    try:
        osd = pytesseract.image_to_osd(img_path)
        script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
        conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd).group(1)
        return script, float(conf)
    except Exception as e:
        print(e)
        return None, 0.0
    
def detect_image_lang(img_path):
    try:
        osd = pytesseract.image_to_osd(img_path)
        script_match = re.search(r"Script: ([a-zA-Z]+)\n", osd)
        conf_match = re.search(r"Script confidence: (\d+\.?(\d+)?)", osd)

        script = script_match.group(1) if script_match else None
        conf = float(conf_match.group(1)) if conf_match else 0.0

        return script, conf
    except Exception as e:
        print(e)
        return None, 0.0