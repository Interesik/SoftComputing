import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os


def add_noise(image: np.ndarray, noise_level: int) -> np.ndarray:
    noisy_image = np.array(image)
    noise = np.random.randint(0, 256, noisy_image.shape).astype(np.uint8)
    noisy_image = np.where(np.random.rand(*noisy_image.shape) < noise_level / 100, noise, noisy_image)
    return noisy_image


def generate_font_image(w, h, x, y, font_file, letter, noise_level, output_directory):
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, size=60)
    draw.text((x, y), letter, font=font, fill=0)

    image_np = np.array(image)

    # Dodanie szumu
    noisy_image = add_noise(image_np, noise_level)

    # Zapisz obraz jako plik .png
    os.makedirs(output_directory, exist_ok=True)
    output_image_path = os.path.join(output_directory, f"{letter}.png")
    Image.fromarray(noisy_image).save(output_image_path)

    # Zapisz opis litery w pliku description.txt
    description_path = os.path.join(output_directory, "description.txt")
    with open(description_path, "a") as f:
        f.write(f"{letter}.png: letter {letter}, noise level {noise_level}%\n")

    print(f"Generated image for letter '{letter}' with noise level {noise_level}%.")

