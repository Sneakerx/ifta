"""I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_grayscale(path: str | Path) -> np.ndarray:
    """Load an image file as a grayscale float array in [0, 1]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def save_image_grayscale(arr: np.ndarray, path: str | Path) -> None:
    """Save a [0, 1] float array as 8-bit grayscale image."""
    a = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="L")
    img.save(path)
