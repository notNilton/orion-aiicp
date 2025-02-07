import tkinter as tk
from PIL import Image, ImageTk

def load_images(image1, image2):  # Accept PIL Image objects
    """Loads and displays two images in a Tkinter window, resized to a maximum of 600x600."""
    try:
        window = tk.Tk()
        window.title("Image Viewer")

        # Resize images if necessary
        max_size = (600, 600)

        image1.thumbnail(max_size) # In-place modification
        image2.thumbnail(max_size) # In-place modification

        tk_image1 = ImageTk.PhotoImage(image1)  # Create Tkinter images from PIL images
        tk_image2 = ImageTk.PhotoImage(image2)

        label1 = tk.Label(window, image=tk_image1)
        label1.pack(side=tk.LEFT)
        label1.image = tk_image1  # Keep a reference

        label2 = tk.Label(window, image=tk_image2)
        label2.pack(side=tk.LEFT)
        label2.image = tk_image2

        window.mainloop()
    except Exception as e:
        print(f"Error displaying images: {e}")