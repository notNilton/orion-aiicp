# from colorthief import ColorThief
# import matplotlib.pyplot as plt

# #Palette Aquisition
# paletteSize = 8
# ct = ColorThief(imageLocation)
# palette = ct.get_palette(color_count=paletteSize+2)
# plt.imshow([[palette[i] for i in range(paletteSize)]])
# plt.show()

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

    # Save original image with "-raw" suffix
    raw_filename = f"./{filename}-raw{ext}"
    if os.path.exists(raw_filename):
        os.remove(raw_filename)
    originalImage.save(raw_filename)

    # Save pixelated image with "-pixel-treated" suffix
    treated_filename = f"./{filename}-pixel-treated{ext}"
    if os.path.exists(treated_filename):
        os.remove(treated_filename)
    pixelImage.save(treated_filename)

    # Print console log if no errors occurred
    print(f"Images processed successfully with n={n}.")

except Exception as e:
    print(f"An error occurred: {e}")


