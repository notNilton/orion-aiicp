import numpy as np
import matplotlib.pyplot as plt
import numba
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import os
import cv2

@numba.jit(nopython=True, nogil=True)
def quantize_to_palette(pixel, palette):
    """
    Finds the closest color in the palette to the given pixel.

    Parameters:
    pixel (np.ndarray): RGB pixel color.
    palette (np.ndarray): Color palette.

    Returns:
    np.ndarray: Closest color from the palette.
    """
    distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
    return palette[np.argmin(distances)]

@numba.jit(nopython=True, nogil=True)
def floyd_steinberg_dithering(image, palette):
    """
    Applies Floyd-Steinberg dithering to an image to reduce colors to a given palette.

    This function iterates over each pixel in the image and quantizes it to the nearest
    color in the provided palette. The quantization error is then diffused to neighboring
    pixels according to the Floyd-Steinberg error diffusion coefficients.

    Parameters:
    image (np.ndarray): Input image as a NumPy array (float in [0, 1]).
    palette (np.ndarray): Color palette as a NumPy array (float in [0, 1]).

    Returns:
    np.ndarray: Dithered image as a NumPy array.
    """
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = quantize_to_palette(old_pixel, palette)
            image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Error diffusion coefficients for Floyd-Steinberg
            if y < height - 1:
                image[y + 1, x] += quant_error * 7 / 16
            if x < width - 1:
                image[y, x + 1] += quant_error * 5 / 16
            if y < height - 1 and x > 0:
                image[y + 1, x - 1] += quant_error * 3 / 16
            if y < height - 1 and x < width - 1:
                image[y + 1, x + 1] += quant_error * 1 / 16
    return image

@numba.jit(nopython=True, nogil=True)
def simple_rounding_quantization(image, palette):
    """
    Applies simple rounding quantization to reduce colors to a given palette.

    This function iterates over each pixel in the image and directly quantizes it
    to the nearest color in the provided palette without error diffusion.

    Parameters:
    image (np.ndarray): Input image as a NumPy array (float in [0, 1]).
    palette (np.ndarray): Color palette as a NumPy array (float in [0, 1]).

    Returns:
    np.ndarray: Quantized image using simple rounding.
    """
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = quantize_to_palette(old_pixel, palette)
            image[y, x] = new_pixel
    return image

def reduce_palette_kmeans(image_data, n_colors):
    """
    Reduces the number of colors in an image using KMeans clustering.

    Parameters:
    image_data (np.ndarray): The image data.
    n_colors (int): The number of colors to reduce to.

    Returns:
    np.ndarray: The reduced color palette.
    """
    pixels = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10).fit(pixels) # explicitly set n_init
    return kmeans.cluster_centers_

def create_custom_palette(use_custom_palette=True):
    """
    Creates a custom color palette or indicates palette detection from image data will be used.

    Parameters:
    use_custom_palette (bool): Whether to use the predefined custom palette.

    Returns:
    np.ndarray or None: The custom color palette or None if palette detection will be used.
    """
    if use_custom_palette:
        black = np.array([0, 0, 0])
        red1 = np.array([1, 0, 0])
        red2 = np.array([0.5, 0, 0])
        white = np.array([1, 1, 1])
        return np.array([black, red1, red2, white])
    else:
        return None  # Palette will be detected dynamically in process_image

def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=3):
    """
    Applies an unsharp mask to sharpen the image.

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
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def process_image(image_path, pixelation_size, n_colors, contrast, use_floyd_steinberg, apply_unsharp, use_custom_palette):
    """
    Processes an image by enhancing contrast, reducing its palette, and applying dithering.

    Parameters:
    image_path (str): Path to the input image.
    pixelation_size (int): Size to which the image will be pixelated (reduced and then upscaled).
    n_colors (int): Number of colors in the reduced palette if not using custom palette.
    contrast (float): Contrast enhancement factor.
    use_floyd_steinberg (bool): True to use Floyd-Steinberg dithering, False for simple rounding.
    apply_unsharp (bool): True to apply unsharp masking.
    use_custom_palette (bool): True to use a predefined custom palette, False to detect palette from image.

    Returns:
    tuple: A tuple containing the processed PIL Image and palette information string.
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
        palette = reduce_palette_kmeans(reduced_image_data, n_colors)
        palette_info = f"{n_colors}ColorPalette"

    if use_floyd_steinberg:
        dithered_image = floyd_steinberg_dithering(reduced_image_data, palette)
        dithering_method = "FloydSteinberg"
    else:
        dithered_image = simple_rounding_quantization(reduced_image_data, palette)
        dithering_method = "SimpleRounding"

    dithered_image = (dithered_image * 255).astype(np.uint8)
    result_image = Image.fromarray(dithered_image)
    pixel_image = result_image.resize(image.size, Image.NEAREST)

    return pixel_image, palette_info, dithering_method

if __name__ == "__main__":
    # Define project directories relative to the script's location
    project_root = os.path.dirname(os.path.abspath(__file__))
    image_dir_untreated = os.path.join(project_root, 'data', 'untreated')
    image_dir_treated = os.path.join(project_root, 'data', 'treated')

    # Ensure treated directory exists
    os.makedirs(image_dir_treated, exist_ok=True)

    # Get the path to the original image (example, you might want to iterate over images in image_dir_untreated)
    image_original_path = os.path.join(image_dir_untreated, 'capy.webp')

    old_filename = os.path.splitext(os.path.basename(image_original_path))[0]

    # --- Processing Parameters ---
    pixelation_size = 2**7 # Reduced image size (e.g., 256x256)
    n_colors = 8 # Number of colors for palette reduction (if not using custom palette)
    image_contrast = 1.2 # Contrast enhancement factor
    use_floyd_steinberg = True # Use Floyd-Steinberg dithering (True) or simple rounding (False)
    use_custom_palette = False # Use custom palette (True) or detect from image (False)
    apply_unsharp = True # Apply unsharp masking (True) or not (False)

    # --- Process and Save Images ---
    process_params = [
        (True, True, "Unsharp_FS"), # Unsharp mask, Floyd-Steinberg
        (False, True, "NoUnsharp_FS"), # No unsharp mask, Floyd-Steinberg
        (True, False, "Unsharp_SR"), # Unsharp mask, Simple Rounding
        (False, False, "NoUnsharp_SR")  # No unsharp mask, Simple Rounding
    ]

    print("Processing images...")
    for apply_unsharp_param, use_floyd_steinberg_param, suffix in process_params:
        processed_image, palette_info, dithering_method = process_image(
            image_original_path, pixelation_size, n_colors, image_contrast,
            use_floyd_steinberg_param, apply_unsharp_param, use_custom_palette
        )

        new_filename = (
            f"{old_filename}_{pixelation_size}_{palette_info}_{image_contrast}Contrast_"
            f"{dithering_method}_{suffix}.png"
        )
        save_path = os.path.join(image_dir_treated, new_filename)
        processed_image.save(save_path)
        print(f"Saved: {new_filename}")

    print("Image processing completed!")