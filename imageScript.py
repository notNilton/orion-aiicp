from PIL import Image

n = 256

originalImage = Image.open("./aiGeneratedPixelArt-2.jpg")
reducedImage = originalImage.resize((n, n), Image.BILINEAR)
pixelImage = reducedImage.resize(originalImage.size, Image.NEAREST)
pixelImage.save('newfDragggg.png')