from colorthief import ColorThief
import matplotlib.pyplot as plt
from PIL import Image


#Palette Aquisition
paletteSize = 8
ct = ColorThief("./PythonScript/testImage1.webp")
palette = ct.get_palette(color_count=6)
plt.imshow([[palette[i] for i in range(6)]])
plt.show()

# Image Generation
n = 256+256
originalImage = Image.open("./PythonScript/testImage1.webp")
reducedImage = originalImage.resize((n, n), Image.BILINEAR)
pixelImage = reducedImage.resize(originalImage.size, Image.NEAREST)
pixelImage.save("ai1.png")


# img = Image.open("./PythonScript/16bitImage.png")

# img_arr = np.array(img)
# # Print shape of array (dimensions of matrix)
# print(img_arr.shape)

# # Print first 5 rows and columns of array (first 5x5 pixels of image)
# print(img_arr[:5, :5])

# def get_image_palette(image_path, num_colors):
#     # Open the image using Pillow
#     image = Image.open(image_path)

#     # Resize the image to a small size to speed up clustering
#     image = image.resize((100, 100))

#     # Convert the image to a numpy array
#     image_array = np.array(image)

#     # Reshape the array to a list of RGB values
#     pixels = image_array.reshape((-1, 3))

#     # Use KMeans clustering to find dominant colors
#     kmeans = KMeans(n_clusters=num_colors)
#     kmeans.fit(pixels)

#     # Get the RGB values of the cluster centers
#     dominant_colors = kmeans.cluster_centers_.astype(int)

#     # Convert the dominant colors to a list of tuples
#     dominant_colors = [tuple(color) for color in dominant_colors]

#     return dominant_colors

# # def display_color_palette(image_path, num_colors):
# #     # Create a window
# #     window = tk.Tk()
# #     window.title("Color Palette Viewer")

# #     # Get the dominant colors in the image
# #     palette = get_image_palette(image_path, num_colors)

# #     # Display the color palette
# #     for i, color in enumerate(palette):
# #         hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        
# #         # Create a canvas to display the color
# #         canvas = Canvas(window, width=50, height=50, bg=hex_color)
# #         canvas.grid(row=0, column=i, padx=5, pady=5)

# #         # Display the color in RGB format
# #         label = Label(window, text=f"RGB: {color}", padx=5, pady=5)
# #         label.grid(row=1, column=i)

# #     # Run the application
# #     window.mainloop()

# if __name__ == "__main__":
#     # Provide the path to the image file
#     image_path = "./PythonScript/testImage1.webp"

#     # Specify the number of colors you want in the palette
#     num_colors = 5

#     # Get the dominant colors in the image
#     palette = get_image_palette(image_path, num_colors)

#     # Print the palette
#     print("Image Palette:", palette)
