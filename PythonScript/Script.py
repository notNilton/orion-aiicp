from colorthief import ColorThief
import matplotlib.pyplot as plt
from PIL import Image
import os

imageLocation = "./PythonScript/aiGeneratedPixelArt-1-P.jpeg"

#Palette Aquisition
paletteSize = 8
ct = ColorThief(imageLocation)
palette = ct.get_palette(color_count=paletteSize+2)
plt.imshow([[palette[i] for i in range(paletteSize)]])
plt.show()

# Image Generation
n = 64
originalImage = Image.open(imageLocation)
reducedImage = originalImage.resize((n, n), Image.BILINEAR)
pixelImage = reducedImage.resize(originalImage.size, Image.NEAREST)
if os.path.exists(imageLocation):
  os.remove(imageLocation)
  pixelImage.save(imageLocation)
else: 
  pixelImage.save(imageLocation)