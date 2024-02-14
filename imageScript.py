from PIL import Image

originalImage = Image.open('paCreated.jpg')
reducedImage = originalImage.resize((512, 512), Image.BILINEAR)
pixelImage = reducedImage.resize(originalImage.size, Image.NEAREST)
pixelImage.save('newImage.png')