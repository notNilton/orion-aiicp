import numba
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import os
import cv2

@numba.jit(nopython=True, nogil=True)
def floyd_steinberg(image, palette):
    Lx, Ly, Lc = image.shape
    for i in range(Lx):
        for j in range(Ly):
            old_pixel = image[i, j]
            new_pixel = quantize_to_palette(old_pixel, palette)
            image[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            if i < Lx - 1:
                image[i + 1, j] += quant_error * 7 / 16
            if j < Ly - 1:
                image[i, j + 1] += quant_error * 5 / 16
            if i > 0 and j < Ly - 1:
                image[i - 1, j + 1] += quant_error * 1 / 16
            if i < Lx - 1 and j < Ly - 1:
                image[i + 1, j + 1] += quant_error * 3 / 16
    return image

@numba.jit(nopython=True, nogil=True)
def simple_rounding(image, palette):
    Lx, Ly, Lc = image.shape
    for i in range(Lx):
        for j in range(Ly):
            old_pixel = image[i, j]
            new_pixel = quantize_to_palette(old_pixel, palette)
            image[i, j] = new_pixel
    return image

@numba.jit(nopython=True, nogil=True)
def quantize_to_palette(pixel, palette):
    distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
    return palette[np.argmin(distances)]

def reduce_palette(image_data, n_colors):
    """
    Reduce the number of colors in an image using KMeans clustering.

    Parameters:
    image_data (np.ndarray): The image data.
    n_colors (int): The number of colors to reduce to.

    Returns:
    np.ndarray: The reduced color palette.
    """
    pixels = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    return kmeans.cluster_centers_

def create_custom_palette(use_custom_palette=True):
    """
    Create a custom color palette or detect palette from image data.

    Parameters:
    use_custom_palette (bool): Whether to use the custom palette.

    Returns:
    np.ndarray: The custom color palette or detected palette.
    """
    if use_custom_palette:
        black = np.array([0, 0, 0])
        red1 = np.array([1, 0, 0])
        red2 = np.array([0.5, 0, 0])
        white = np.array([1, 1, 1])
        return np.array([black, red1, red2, white])
    else:
        # Return detected palette from image data
        return None  # This will be handled in the process_image function

def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=12):
    """
    Apply an unsharp mask to an image.

    Parameters:
    image (np.ndarray): The input image.
    kernel_size (tuple): The size of the Gaussian kernel.
    sigma (float): The standard deviation of the Gaussian kernel.
    strength (float): The strength of the sharpening effect.

    Returns:
    np.ndarray: The sharpened image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(strength + 1) * image - float(strength) * blurred
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened.astype(np.uint8)

def process_image(image_path, pixelation_size, n_colors, contrast, use_floyd_steinberg, apply_unsharp, use_custom_palette):
    """
    Process an image by enhancing contrast, reducing its palette, and applying dithering.

    Parameters:
    image_path (str): Path to the input image.
    pixelation_size (int): The size to which the image will be reduced.
    n_colors (int): The number of colors in the reduced palette.
    contrast (float): The contrast enhancement factor.
    use_floyd_steinberg (bool): Whether to use Floyd-Steinberg dithering.
    apply_unsharp (bool): Whether to apply unsharp masking.
    use_custom_palette (bool): Whether to use the custom predefined palette.

    Returns:
    Image: The processed image.
    """
    image = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    if apply_unsharp:
        image_data = np.array(image)
        image_data = apply_unsharp_mask(image_data)
        image = Image.fromarray(image_data)

    reduced_image = image.resize((pixelation_size, pixelation_size), Image.BILINEAR)
    reduced_image_data = np.array(reduced_image).astype(np.float32) / 255.0

    if use_custom_palette:
        palette = create_custom_palette(use_custom_palette=True)
        palette_info = "CustomPalette"
    else:
        # Automatically detect palette from image data
        palette = reduce_palette(reduced_image_data, n_colors)
        palette_info = f"{n_colors}bitPalette"
    
    if use_floyd_steinberg:
        dithered_image = floyd_steinberg(reduced_image_data, palette)
    else:
        dithered_image = simple_rounding(reduced_image_data, palette)
    
    dithered_image = (dithered_image * 255).astype(np.uint8)
    result_image = Image.fromarray(dithered_image)
    pixel_image = result_image.resize(image.size, Image.NEAREST)
    
    return pixel_image, palette_info

# Original Image
image_original = r"C:\Development\AIICP\Images\Untreated\mountain.jpg"

# Remove the extension
old_filename = os.path.splitext(os.path.basename(image_original))[0]

# Desired pixelation size
pixelation_size = 2**7
n_colors = 2**8
image_contrast = 1

# Choose dithering method
use_floyd_steinberg = True  # Set to True for Floyd-Steinberg, False for simple rounding

# Choose whether to use custom palette or detect from image
use_custom_palette = False  # Set to True to use custom predefined palette, False to detect from image

# Process the image with unsharp mask
apply_unsharp = True  # Set to True to apply unsharp masking
processed_image_with_unsharp, palette_info = process_image(image_original, pixelation_size, n_colors, image_contrast, use_floyd_steinberg, apply_unsharp, use_custom_palette)

# Process the image without unsharp mask
apply_unsharp = False  # Set to False to not apply unsharp masking
processed_image_without_unsharp, palette_info = process_image(image_original, pixelation_size, n_colors, image_contrast, use_floyd_steinberg, apply_unsharp, use_custom_palette)

# Construct the filenames
new_filename_with_unsharp = f"{old_filename}_{pixelation_size}_{palette_info}_{image_contrast}'Contrast_{use_floyd_steinberg}_unsharp.png"
new_filename_without_unsharp = f"{old_filename}_{pixelation_size}_{palette_info}_{image_contrast}Contrast_{use_floyd_steinberg}_nosharp.png"

# Save paths
save_path_with_unsharp = rf"C:\Development\aiicp\Images\Treated\{new_filename_with_unsharp}"
save_path_without_unsharp = rf"C:\Development\aiicp\Images\Treated\{new_filename_without_unsharp}"

# Save the images
processed_image_with_unsharp.save(save_path_with_unsharp)
processed_image_without_unsharp.save(save_path_without_unsharp)

print(f"It's Done! Files saved:\n{new_filename_with_unsharp}\n{new_filename_without_unsharp}")
