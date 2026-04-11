from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


def read_and_validate_image(contents: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        return image
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image file.") from exc


def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> np.ndarray:
    image = image.resize((image_size, image_size))
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def apply_tta(batch: np.ndarray) -> np.ndarray:
    # Minimal deterministic TTA set: original, horizontal flip, mild brightness.
    x = batch[0]
    x_flip = np.fliplr(x)
    x_bright = np.clip(x * 1.05, 0.0, 1.0)
    return np.stack([x, x_flip, x_bright], axis=0)
