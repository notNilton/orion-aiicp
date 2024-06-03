import cv2
import numpy as np
from matplotlib import pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=0.3, threshold=0):
    """
    Apply unsharp mask to an image.

    Parameters:
    - image: input image
    - kernel_size: size of the Gaussian kernel
    - sigma: standard deviation of the Gaussian kernel
    - amount: amount of sharpening
    - threshold: threshold for minimum amount of difference

    Returns:
    - Sharpened image
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    difference = cv2.subtract(image, blurred_image)
    sharpened_image = cv2.addWeighted(image, 1 + amount, difference, amount, 0)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image

# Read the color image
image = cv2.imread(r"C:\Development\aiicp\Images\Untreated\river.jpg")

# Convert the image from BGR to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the image into its respective R, G, B channels
r, g, b = cv2.split(image)

# Apply unsharp mask to each channel
r_sharpened = unsharp_mask(r)
g_sharpened = unsharp_mask(g)
b_sharpened = unsharp_mask(b)

# Merge the sharpened channels back into a single image
sharpened_image = cv2.merge([r_sharpened, g_sharpened, b_sharpened])
processed_image.save("C:\Development\aiicp\Images\Treated\river.png")



# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sharpened Image')
plt.imshow(sharpened_image)
plt.axis('off')

plt.show()
