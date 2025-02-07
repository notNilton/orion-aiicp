from PIL import Image
from functions.image_manipulation import floyd_steinberg, simple_threshold
from functions.utils import load_images, save_image
import os

if __name__ == "__main__":
    original_image_path = "./data/untreated/cat_image.jpg"
    processed_image_path = "./data/treated/"  # Directory path
    use_grayscale = True  # Set to True for grayscale, False for RGB
    use_floydsteinberg = True  # Set to True for Floyd-Steinberg, False for simple threshold

    try:
        input_image = Image.open(original_image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {original_image_path}")
        exit()

    # Process the image based on the flags
    if use_grayscale:
        input_image_processed = input_image.convert("L")
        if use_floydsteinberg:
            output_image = floyd_steinberg(input_image)
        else:
            output_image = simple_threshold(input_image)
        output_image_processed = output_image.convert("L")
    else:
        input_image_processed = input_image.copy()
        if use_floydsteinberg:
            output_image = floyd_steinberg(input_image)  # Use RGB Floyd-Steinberg
        else:
            output_image = simple_threshold(input_image)  # Use RGB color thresholding
        output_image_processed = output_image.copy()


    # Save processed image - Provide a filename!
    filename = os.path.basename(original_image_path)
    name, ext = os.path.splitext(filename)
    save_image(output_image_processed, processed_image_path, f"{name}_processed{ext}") #Dynamic name

    # Load and display the images
    load_images(input_image_processed, output_image_processed)