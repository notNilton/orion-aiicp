from PIL import Image
import os

imageLocation = "./landscape-river.jpg"

# Extract filename without extension
filename, ext = os.path.splitext(os.path.basename(imageLocation))

try:
    # Image Generation with n as a power of 2
    originalImage = Image.open(imageLocation)
    n = 2 ** 6
    reducedImage = originalImage.resize((n, n), Image.BILINEAR)
    pixelImage = reducedImage.resize(originalImage.size, Image.NEAREST)

    # Convert image to 'P' mode (each pixel's color value is unique)
    unique_color_image = pixelImage.convert("P", palette=Image.ADAPTIVE, colors=256)

    # Get the color values
    color_counts = unique_color_image.getcolors()

    # Get the image palette
    palette = unique_color_image.getpalette()

    # Create a dictionary to store RGB values for each color index
    color_index_to_rgb = {}

    # Loop through the color counts and map color indices to RGB values
    for count, index in color_counts:
        # Get RGB values from the palette
        rgb = palette[index * 3: index * 3 + 3]
        # Store RGB values in the dictionary
        color_index_to_rgb[index] = rgb

    # Print the RGB values corresponding to the most concurrent distinct colors
    print("Four most concurrent distinct colors (RGB values):")
    for count, index in sorted(color_counts, key=lambda x: x[0], reverse=True)[:8]:
        rgb = color_index_to_rgb[index]
        print(f"Count: {count}, RGB: {rgb}")

except Exception as e:
    print(f"An error occurred: {e}")
