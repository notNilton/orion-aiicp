from PIL import Image

n = 256+256

originalImage = Image.open("./aiGeneratedPixelArt-1.jpeg")
reducedImage = originalImage.resize((n, n), Image.BILINEAR)
pixelImage = reducedImage.resize(originalImage.size, Image.NEAREST)
pixelImage.save('aiGeneratedPixelArt-1-T.png')