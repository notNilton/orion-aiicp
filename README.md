# Artificial Intelligence Image Pipeline

This Python script uses artificial intelligence techniques to process and transform an input image. The pipeline includes pixelation, palette reduction, dithering, and contrast enhancement.

## Overview

1. **Pixelation**: Resizes the input image to a specified size to create a pixelated effect.
2. **Palette Reduction**: Reduces the number of colors in the image palette using KMeans clustering.
3. **Dithering**: Applies Floyd-Steinberg dithering or simple rounding to distribute colors and reduce artifacts.
4. **Contrast Enhancement**: Enhances the contrast of the processed image.
5. **Saving**: Saves the processed image with a descriptive filename indicating the processing parameters used.

## Requirements

- Python 3.x
- NumPy
- Numba
- Pillow (Python Imaging Library)
- scikit-learn (for KMeans clustering)

## Usage

1. Clone the repository or download the script.
2. Ensure all dependencies are installed using `pip install -r requirements.txt`.
3. Replace the `image_path` variable in the script with the path to your input image.
4. Optionally adjust the `pixelation_size`, `n_colors`, and `use_floyd_steinberg` variables to customize the processing.
5. Run the script using `python image_pipeline.py`.

## Parameters

- `image_path`: Path to the input image.
- `pixelation_size`: Desired size for pixelation.
- `n_colors`: Number of colors to reduce the palette to.
- `use_floyd_steinberg`: Set to `True` for Floyd-Steinberg dithering, `False` for simple rounding.

## Output

The processed image is saved with a filename indicating the processing parameters used.

Example filename: `river_256_16bit_2Contrast_False.png`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact [nilton.naab@gmail.com](mailto:nilton.naab@gmail.com).
