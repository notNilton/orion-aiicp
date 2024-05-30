import numba
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import os

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
    print(distances)
    return palette[np.argmin(distances)]

def reduce_palette(image_data, n_colors):
    # Flatten the image array and fit KMeans
    pixels = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    # print(image_data)
    return kmeans.cluster_centers_

def create_custom_palette():
    # Define the colors
    black = np.array([0, 0, 0])
    red = np.array([1, 0, 0])
    white = np.array([1, 1, 1])
    
    # Create the palette array
    palette = np.array([black, red, white])
    
    return palette

def process_image(image_path, pixelation_size, n_colors, use_floyd_steinberg):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Reduce the size (pixelate)
    reduced_image = image.resize((pixelation_size, pixelation_size), Image.BILINEAR)
    
    # Convert the reduced image to a NumPy array and normalize
    reduced_image_data = np.array(reduced_image).astype(np.float32) / 255.0
    
    # Reduce the palette of the image
    palette = reduce_palette(reduced_image_data, n_colors)
    # palette = create_custom_palette()
    
    # Apply the chosen dithering method with the reduced palette
    if use_floyd_steinberg:
        dithered_image = floyd_steinberg(reduced_image_data, palette)
    else:
        dithered_image = simple_rounding(reduced_image_data, palette)
    
    # Convert back to 8-bit per channel
    dithered_image = (dithered_image * 255).astype(np.uint8)
    
    # Convert NumPy array back to an image
    result_image = Image.fromarray(dithered_image)
    
    # Resize back to original size using nearest neighbor to maintain pixelation effect
    pixel_image = result_image.resize(image.size, Image.NEAREST)
    
    return pixel_image


# Path to your imagem
image_path = r"C:\Development\aiicp\Images\Untreated\river.jpg"
# filename = os.path.basename(image_path)

# Remove the extension
old_filename = os.path.splitext(os.path.basename(image_path))[0]

# Desired pixelation size (e.g., 256x256 pixels)
pixelation_size = 2**8
n_colors = 2**4
image_contrast = 2

# Choose dithering method
use_floyd_steinberg = False  # Set to True for Floyd-Steinberg, False for simple rounding

# Process the image
processed_image = process_image(image_path, pixelation_size, n_colors, use_floyd_steinberg)

# Construct the filename using pixelation_size and n_colors
new_filename = f"{old_filename}_{pixelation_size}_{n_colors}bit_{image_contrast}Contrast_{use_floyd_steinberg}.png"
save_path = rf"C:\Development\aiicp\Images\Treated\{new_filename}"
processed_image.save(save_path)

print(f"It's Done! Name file: {new_filename}")