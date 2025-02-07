import numpy as np
import numba
from sklearn.cluster import KMeans
from PIL import Image, ImageEnhance # Import PIL modules here as they are used by process_image

@numba.jit(nopython=True, nogil=True)
def quantize_to_palette(pixel, palette):
    """
    Encontra a cor mais próxima na paleta para o pixel dado.

    Parâmetros:
    pixel (np.ndarray): Cor do pixel RGB.
    palette (np.ndarray): Paleta de cores.

    Retorna:
    np.ndarray: Cor mais próxima da paleta.
    """
    distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
    return palette[np.argmin(distances)]

@numba.jit(nopython=True, nogil=True)
def floyd_steinberg_dithering(image, palette):
    """
    Aplica dithering Floyd-Steinberg a uma imagem para reduzir cores a uma paleta dada.

    Esta função itera sobre cada pixel na imagem e quantiza para a cor mais próxima
    na paleta fornecida. O erro de quantização é então difundido para pixels vizinhos
    de acordo com os coeficientes de difusão de erro de Floyd-Steinberg.

    Parâmetros:
    image (np.ndarray): Imagem de entrada como um array NumPy (float em [0, 1]).
    palette (np.ndarray): Paleta de cores como um array NumPy (float em [0, 1]).

    Retorna:
    np.ndarray: Imagem com dithering como um array NumPy.
    """
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = quantize_to_palette(old_pixel, palette)
            image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Coeficientes de difusão de erro para Floyd-Steinberg
            if y < height - 1:
                image[y + 1, x] += quant_error * 7 / 16
            if x < width - 1:
                image[y, x + 1] += quant_error * 5 / 16
            if y < height - 1 and x > 0:
                image[y + 1, x - 1] += quant_error * 3 / 16
            if y < height - 1 and x < width - 1:
                image[y + 1, x + 1] += quant_error * 1 / 16
    return image

@numba.jit(nopython=True, nogil=True)
def simple_rounding_quantization(image, palette):
    """
    Aplica quantização de arredondamento simples para reduzir cores a uma paleta dada.

    Esta função itera sobre cada pixel na imagem e diretamente quantiza para o mais próximo
    cor na paleta fornecida sem difusão de erro.

    Parâmetros:
    image (np.ndarray): Imagem de entrada como um array NumPy (float em [0, 1]).
    palette (np.ndarray): Paleta de cores como um array NumPy (float em [0, 1]).

    Retorna:
    np.ndarray: Imagem quantizada usando arredondamento simples.
    """
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = quantize_to_palette(old_pixel, palette)
            image[y, x] = new_pixel
    return image

def reduce_palette_kmeans(image_data, n_colors):
    """
    Reduz o número de cores em uma imagem usando o clustering KMeans.

    Parâmetros:
    image_data (np.ndarray): Os dados da imagem.
    n_colors (int): O número de cores para reduzir.

    Retorna:
    np.ndarray: A paleta de cores reduzida.
    """
    pixels = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10).fit(pixels) # explicitly set n_init
    return kmeans.cluster_centers_

def create_custom_palette(use_custom_palette=True):
    """
    Cria uma paleta de cores personalizada ou indica que a paleta será detectada dos dados da imagem.

    Parâmetros:
    use_custom_palette (bool): Se deve usar a paleta personalizada predefinida.

    Retorna:
    np.ndarray ou None: A paleta de cores personalizada ou None se a paleta for detectada da imagem.
    """
    if use_custom_palette:
        black = np.array([0, 0, 0])
        red1 = np.array([1, 0, 0])
        red2 = np.array([0.5, 0, 0])
        white = np.array([1, 1, 1])
        return np.array([black, red1, red2, white])
    else:
        return None  # Paleta será detectada dinamicamente na função process_image

def process_image(image_path, pixelation_size, n_colors, contrast, use_floyd_steinberg, use_custom_palette):
    """
    Processa uma imagem melhorando o contraste, reduzindo sua paleta e aplicando dithering.

    Parâmetros:
    image_path (str): Caminho para a imagem de entrada.
    pixelation_size (int): Tamanho para o qual a imagem será pixelada (reduzida e depois ampliada).
    n_colors (int): Número de cores na paleta reduzida se não estiver usando paleta personalizada.
    contrast (float): Fator de melhoria de contraste.
    use_floyd_steinberg (bool): True para usar dithering Floyd-Steinberg, False para arredondamento simples.
    use_custom_palette (bool): True para usar uma paleta personalizada predefinida, False para detectar paleta da imagem.

    Retorna:
    tuple: Uma tupla contendo a imagem PIL processada e a string de informação da paleta.
    """
    image = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Redimensionar a imagem para o tamanho de pixelização mantendo a razão de aspecto
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        resized_width = pixelation_size
        resized_height = int(pixelation_size / aspect_ratio)
    else:
        resized_height = pixelation_size
        resized_width = int(pixelation_size * aspect_ratio)


    reduced_image = image.resize((resized_width, resized_height), Image.BILINEAR)
    reduced_image_data = np.array(reduced_image).astype(np.float32) / 255.0

    if use_custom_palette:
        palette = create_custom_palette(use_custom_palette=True)
        palette_info = "CustomPalette"
    else:
        palette = reduce_palette_kmeans(reduced_image_data, n_colors)
        palette_info = f"{n_colors}ColorPalette"

    if use_floyd_steinberg:
        dithered_image = floyd_steinberg_dithering(reduced_image_data, palette)
        dithering_method = "FloydSteinberg"
    else:
        dithered_image = simple_rounding_quantization(reduced_image_data, palette)
        dithering_method = "SimpleRounding"

    dithered_image = (dithered_image * 255).astype(np.uint8)
    result_image = Image.fromarray(dithered_image)
    pixel_image = result_image.resize(image.size, Image.NEAREST)

    return pixel_image, palette_info, dithering_method