import tkinter as tk
from PIL import Image, ImageTk
import os  # Import the os module

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
        
        
import os

def save_image(image, save_path, filename):
    """Saves a PIL Image object to the specified directory with a given filename."""
    try:
        os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist

        full_path = os.path.join(save_path, filename)  # Join path and filename

        image.save(full_path)
        print(f"Image saved successfully at {full_path}")
    except Exception as e:
        print(f"Error saving the image: {e}")