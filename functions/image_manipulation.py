# implementation of a simple floyd-steinberg dithering algorithm
from PIL import Image
import numpy as np

def simple_threshold(image):
    
    
    """
    Applies a simple thresholding to an image, converting each RGB component
    of each pixel to either 0 or 255 based on its original value.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The thresholded image.
    """

    # Convert the image to RGB mode if it's not already

    # Convert the image to a NumPy array for easier manipulation
    pixels = np.array(image, dtype=np.uint8)

    # Get the width and height of the image
    height, width, _ = pixels.shape # Correct order for numpy array shape

    # Iterate over each pixel and apply thresholding to each color component
    for y in range(height):
        for x in range(width):
            for c in range(3):  # Iterate over the three color channels (R, G, B)
                old_pixel = pixels[y, x, c]
                new_pixel = 255 if old_pixel > 127 else 0
                pixels[y, x, c] = new_pixel

    # Convert the NumPy array back to a PIL image
    thresholded_image = Image.fromarray(pixels)

    return thresholded_image

def floyd_steinberg(image):
    """
    Applies Floyd-Steinberg dithering to an image.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The dithered image.
    """
    # Convert the image to grayscale (if it's not already)
    image = image.convert("L")

    # Convert the image to a NumPy array for easier manipulation
    pixels = np.array(image, dtype=float)

    # Get the width and height of the image
    width, height = image.size

    # Iterate over each pixel
    for y in range(height - 1):  # Stop at height - 1 to avoid index errors
        for x in range(1, width - 1):  # Stop at width - 1 to avoid index errors
            # Get the current pixel value
            old_pixel = pixels[y, x]

            # Quantize the pixel to 0 or 255
            new_pixel = 255 if old_pixel > 127 else 0

            # Set the new pixel value
            pixels[y, x] = new_pixel

            # Calculate the quantization error
            error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels
            pixels[y, x + 1] += error * 7 / 16  # Right
            pixels[y + 1, x - 1] += error * 3 / 16  # Bottom-left
            pixels[y + 1, x] += error * 5 / 16  # Bottom
            pixels[y + 1, x + 1] += error * 1 / 16  # Bottom-right

    # Clip pixel values to ensure they are within the valid range [0, 255]
    pixels = np.clip(pixels, 0, 255)

    # Convert the NumPy array back to a PIL image
    dithered_image = Image.fromarray(pixels.astype(np.uint8))

    return dithered_image