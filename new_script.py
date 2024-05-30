import numpy as np
import matplotlib.pyplot as plt

def floyd_steinberg_dithering(image):
    # Convert the input image to floating point format with values in [0, 1]
    Iacc = image.astype(float) / 255.0

    # Create the output binary image
    height, width = Iacc.shape
    Iout = np.zeros((height, width))

    # Define the error diffusion coefficients
    coefficients = [
        (1, 0, 7 / 16),
        (1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (-1, 1, 1 / 16)
    ]

    # Process each pixel in a zig-zag order
    for y in range(height):
        if y % 2 == 0:
            # Left to right
            for x in range(width):
                old_pixel = Iacc[y, x]
                new_pixel = 1 if old_pixel >= 0.5 else 0
                Iout[y, x] = new_pixel

                error = old_pixel - new_pixel

                for dx, dy, coef in coefficients:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        Iacc[ny, nx] += error * coef
        else:
            # Right to left
            for x in range(width - 1, -1, -1):
                old_pixel = Iacc[y, x]
                new_pixel = 1 if old_pixel >= 0.5 else 0
                Iout[y, x] = new_pixel

                error = old_pixel - new_pixel

                for dx, dy, coef in coefficients:
                    nx, ny = x - dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        Iacc[ny, nx] += error * coef

    return (Iout * 255).astype(np.uint8)

if __name__ == "__main__":
    # Load the grayscale image
    image = plt.imread(r"C:\Development\aiicp\Images\Untreated\river.jpg")

    # Ensure the image is in grayscale
    if image.ndim == 3:
        image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

    # Apply the Floyd-Steinberg dithering algorithm
    dithered_image = floyd_steinberg_dithering(image)

    # Display the original and dithered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Dithered Image")
    plt.imshow(dithered_image, cmap='gray')

    plt.show()
