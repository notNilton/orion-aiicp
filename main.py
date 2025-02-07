from PIL import Image
from functions.image_manipulation import simple_threshold, simple_threshold
from functions.utils import load_images

if __name__ == "__main__":
    original_image_path = "./data/untreated/cat_image.jpg"
    use_grayscale = True  # Set to True for grayscale, False for RGB

    try:
        input_image = Image.open(original_image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {original_image_path}")
        exit()

    # Process the image based on the grayscale flag
    input_image_processed = input_image.convert("L") if use_grayscale else input_image.copy()
    output_image = simple_threshold(input_image) if use_grayscale else simple_threshold(input_image)
    output_image_processed = output_image.convert("L") if use_grayscale else output_image.copy()

    # Load and display the images
    load_images(input_image_processed, output_image_processed)