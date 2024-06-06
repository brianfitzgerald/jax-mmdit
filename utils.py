from typing import List
from PIL import Image
from pathlib import Path
import shutil
import jax.numpy as jnp
from jax import Array


def image_grid(image_list: List[List[Image.Image]]) -> Image.Image:
    """
    Generate a single grid image from a 2D list of images.
    Empty / None cells are filled with a black image.
    """
    if len(image_list) == 0 or image_list[0][0] == None:
        raise ValueError("No images found in grid.")
    num_rows, num_cols = len(image_list), max(len(row) for row in image_list)
    image_width, image_height = image_list[0][0].size

    grid_width = num_cols * image_width
    grid_height = num_rows * image_height

    grid_image = Image.new("RGB", (grid_width, grid_height))

    for row in range(num_rows):
        for col in range(num_cols):
            x_offset = col * image_width
            y_offset = row * image_height
            if row >= len(image_list) or col >= len(image_list[row]):
                img = None
            else:
                img = image_list[row][col]
            if not img:
                img = Image.new("RGB", (image_width, image_height))
            grid_image.paste(img, (x_offset, y_offset))
    return grid_image


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)


def normalize_images(images: Array, mean: float = 0.5, std: float = 0.5):
    """
    Normalize a batch of images from [0, 255] to [-1, 1].
    """
    mean_array = jnp.array(mean).reshape(1, 1, 1, -1)
    std_array = jnp.array(std).reshape(1, 1, 1, -1)
    normalized_images = (images / 255 - mean_array) / std_array
    return normalized_images


def denormalize_images(images: Array, mean: float = 0.5, std: float = 0.5):
    """
    Denormalize a batch of images from [-1, 1] to [0, 255].
    """
    mean_array = jnp.array(mean).reshape(1, 1, 1, -1)
    std_array = jnp.array(std).reshape(1, 1, 1, -1)
    denormalized_images = images * std_array + mean_array
    denormalized_images = jnp.clip(denormalized_images * 255, 0, 255).astype(jnp.uint8)
    return denormalized_images


def center_crop(image: Image.Image, crop_width: int, crop_height: int):
    width, height = image.size
    left = int((width - crop_width) / 2)
    top = int((height - crop_height) / 2)
    right = int((width + crop_width) / 2)
    bottom = int((height + crop_height) / 2)

    img_cropped = image.crop((left, top, right, bottom))
    return img_cropped
