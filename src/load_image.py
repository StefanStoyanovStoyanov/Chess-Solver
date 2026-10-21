from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(image_path: str | Path) -> Image.Image:
    return load_image(image_path).convert("L")


def load_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.copy()


def resize_image(
    image: Image.Image | np.ndarray,
    max_size: tuple[int, int] = (2000, 2000),
    max_fail_size: tuple[int, int] = (2000, 2000),
) -> Image.Image | None:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = image.size
    if width > max_fail_size[0] or height > max_fail_size[1]:
        return None

    scale = min(max_size[0] / width, max_size[1] / height, 1)
    if scale < 1:
        image = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.BILINEAR,
        )
    return image
