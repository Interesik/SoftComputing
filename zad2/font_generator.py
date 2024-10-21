import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os


def add_noise(image: np.ndarray, noise_level: int) -> np.ndarray:
    """Adds random noise to the image based on the noise level."""
    noisy_image = np.array(image)
    noise = np.random.randint(0, 256, noisy_image.shape).astype(np.uint8)
    noisy_image = np.where(np.random.rand(*noisy_image.shape) < noise_level / 100, noise, noisy_image)
    return noisy_image


def generate_font_image(w, h, x, y, font_file, letter, noise_level, output_directory):
    """Generates an image of a letter with noise and saves it as a .png file."""
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, size=60)
    draw.text((x, y), letter, font=font, fill=0)

    image_np = np.array(image)

    # Add noise to the image
    noisy_image = add_noise(image_np, noise_level)

    # Save the image as a .png file
    os.makedirs(output_directory, exist_ok=True)
    output_image_path = os.path.join(output_directory, f"{letter}.png")
    Image.fromarray(noisy_image).save(output_image_path)

    # Save the letter description in the description.txt file
    description_path = os.path.join(output_directory, "description.txt")
    with open(description_path, "a") as f:
        f.write(f"{letter}.png: letter {letter}, noise level {noise_level}%\n")

    print(f"Generated image for letter '{letter}' with noise level {noise_level}%.")

