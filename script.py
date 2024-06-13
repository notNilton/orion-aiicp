import numba
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import os
import cv2

def edge_detection(image_path, save_directory):
    """
    Detect edges in an image and save the result in the specified directory.

    Parameters:
    image_path (str): Path to the input image.
    save_directory (str): Directory where the edge-detected image will be saved.

    Returns:
    str: Path to the saved edge-detected image.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    edge_filename = os.path.join(save_directory, f"{base_filename}_edges.jpg")
    cv2.imwrite(edge_filename, edges)
    print(f"Edge detection done! Edge file: {edge_filename}")
    return edge_filename

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

def create_custom_palette():
    """
    Create a custom color palette.

    Returns:
    np.ndarray: The custom color palette.
    """
    black = np.array([0, 0, 0])
    red = np.array([1, 0, 0])
    white = np.array([1, 1, 1])
    return np.array([black, red, white])

def process_image(image_path, pixelation_size, n_colors, contrast, use_floyd_steinberg):
    """
    Process an image by enhancing contrast, reducing its palette, and applying dithering.

    Parameters:
    image_path (str): Path to the input image.
    pixelation_size (int): The size to which the image will be reduced.
    n_colors (int): The number of colors in the reduced palette.
    contrast (float): The contrast enhancement factor.
    use_floyd_steinberg (bool): Whether to use Floyd-Steinberg dithering.

    Returns:
    Image: The processed image.
    """
    image = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    reduced_image = image.resize((pixelation_size, pixelation_size), Image.BILINEAR)
    reduced_image_data = np.array(reduced_image).astype(np.float32) / 255.0

    palette = reduce_palette(reduced_image_data, n_colors)
    
    if use_floyd_steinberg:
        dithered_image = floyd_steinberg(reduced_image_data, palette)
    else:
        dithered_image = simple_rounding(reduced_image_data, palette)
    
    dithered_image = (dithered_image * 255).astype(np.uint8)
    result_image = Image.fromarray(dithered_image)
    pixel_image = result_image.resize(image.size, Image.NEAREST)
    
    return pixel_image

def overlay_images(original_image_path, edge_image_path, save_directory):
    """
    Overlay the edge-detected image on the original image and save the result.

    Parameters:
    original_image_path (str): Path to the original image.
    edge_image_path (str): Path to the edge-detected image.
    save_directory (str): Directory where the overlaid image will be saved.
    """
    original_image = Image.open(original_image_path).convert("RGB")
    edge_image = Image.open(edge_image_path).convert("L")  # Convert to grayscale

    # Ensure the edge image is the same size as the original image
    edge_image = edge_image.resize(original_image.size, Image.NEAREST)

    # Create a mask from the edge image where the edges are white
    mask = edge_image.point(lambda p: p > 0 and 255)

    # Create an image with red edges
    red_edges = Image.new("RGB", original_image.size, (255, 0, 0))
    red_edges.paste(original_image, mask=Image.fromarray(255 - np.array(mask)))

    # Overlay the red edges onto the original image
    overlaid_image = Image.composite(original_image, red_edges, mask)

    # Save the overlaid image
    base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
    overlay_filename = os.path.join(save_directory, f"{base_filename}_overlay.jpg")
    overlaid_image.save(overlay_filename)
    print(f"Overlay done! Overlay file: {overlay_filename}")

    # Save a copy of the original image in the same directory
    original_copy_filename = os.path.join(save_directory, f"{base_filename}_original.jpg")
    original_image.save(original_copy_filename)
    print(f"Original image copy saved! File: {original_copy_filename}")

# Original Image
image_original = r"C:\Development\AIICP\Images\Untreated\mountain.jpg"

# Remove the extension
old_filename = os.path.splitext(os.path.basename(image_original))[0]

# Desired pixelation size (e.g., 256x256 pixels)
pixelation_size = 2**7
n_colors = 2**4
image_contrast = 1

# Choose dithering method
use_floyd_steinberg = False  # Set to True for Floyd-Steinberg, False for simple rounding

# Process the image
processed_image = process_image(image_original, pixelation_size, n_colors, image_contrast, use_floyd_steinberg)

# Construct the filename using pixelation_size and n_colors
new_filename = f"{old_filename}_{pixelation_size}_{n_colors}bit_{image_contrast}Contrast_{use_floyd_steinberg}.png"
save_path = rf"C:\Development\aiicp\Images\Treated\{new_filename}"
save_directory = os.path.dirname(save_path)
processed_image.save(save_path)

# Perform edge detection
# edge_image_path = edge_detection(image_original, save_directory)

# Overlay images
# overlay_images(image_original, edge_image_path, save_directory)

print(f"It's Done! Name file: {new_filename}")
