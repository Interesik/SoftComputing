import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os

def add_noise(image: np.ndarray, noise_level: int) -> np.ndarray:
    noisy_image = np.array(image)
    noise = np.random.randint(0, 256, noisy_image.shape).astype(np.uint8)
    noisy_image = np.where(np.random.rand(*noisy_image.shape) < noise_level / 100, noise, noisy_image)
    return noisy_image

def generate_font_image(w: int, h: int, x: int, y: int, font_file: str, letter: str, noise_level: int, output_directory: str):
    # Creating image with provided size and letter drawing
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, size=60)
    draw.text((x, y), letter, font=font, fill=0)
    
    image_np = np.array(image)
    
    # Adding noise
    noisy_image = add_noise(image_np, noise_level)
    
    # Save file as .png
    os.makedirs(output_directory, exist_ok=True)
    output_image_path = os.path.join(output_directory, f"{letter}.png")
    Image.fromarray(noisy_image).save(output_image_path)
    
    # Save letter description in description.txt file
    description_path = os.path.join(output_directory, "description.txt")
    with open(description_path, "a") as f:
        f.write(f"{letter}.png: letter {letter}, noise level {noise_level}%\n")
    
    print(f"Generated image for letter '{letter}' with noise level {noise_level}%.")


# font_generator.py

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def add_noise(image: Image.Image, noise_level: int) -> Image.Image:
    """
    Dodaje szum do obrazu na podstawie poziomu szumu.
    :param image: Obraz PIL.
    :param noise_level: Poziom szumu w procentach (0-100).
    :return: Obraz z dodanym szumem.
    """
    if noise_level == 0:
        return image

    np_image = np.array(image)
    total_pixels = np_image.size
    num_noisy_pixels = int(total_pixels * noise_level / 100)

    for _ in range(num_noisy_pixels):
        x = random.randint(0, image.width - 1)
        y = random.randint(0, image.height - 1)
        # Toggle pixel value: 0 -> 255, 255 -> 0
        np_image[y, x] = 0 if np_image[y, x] == 255 else 255

    return Image.fromarray(np_image)

def generate_letter_image(w: int, h: int, x: int, y: int, font_path: str, letter: str, noise_level: int) -> Image.Image:
    """
    Generuje obraz litery z określonymi parametrami.
    :param w: Szerokość obrazu.
    :param h: Wysokość obrazu.
    :param x: Pozycja litery na osi X.
    :param y: Pozycja litery na osi Y.
    :param font_path: Ścieżka do pliku .ttf czcionki.
    :param letter: Litera do wygenerowania.
    :param noise_level: Poziom szumu w procentach.
    :return: Obraz PIL z wygenerowaną literą i dodanym szumem.
    """
    # Utwórz pusty obraz biały
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, size=int(h * 0.8))  # Dopasuj rozmiar czcionki
    except IOError:
        print(f"Nie można otworzyć pliku czcionki: {font_path}")
        sys.exit(1)

    # Rysuj literę czarnym kolorem
    draw.text((x, y), letter, font=font, fill=0)

    # Dodaj szum
    noisy_image = add_noise(image, noise_level)

    return noisy_image



