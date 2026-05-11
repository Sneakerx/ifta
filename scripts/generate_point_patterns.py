"""Generate the Klasse-2 point-pattern test set.

Each image is a 256x256 grayscale PNG with a black background and
~5-50 bright pixels (gaussian-shaped spots) at pseudorandom positions.
These targets are the classical kinoform application of
Prongue (1992) / Bengtsson (1994) and are sparser than the Klasse-1
natural-image set.

Usage:
    python scripts/generate_point_patterns.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


SIZE = 256
N_IMAGES = 10
COUNTS = [5, 8, 12, 16, 20, 25, 32, 40, 48, 50]  # one per image
SPOT_SIGMA = 1.0  # pixels


def gaussian_spot(canvas: np.ndarray, x: int, y: int, sigma: float = 1.0) -> None:
    """Add a small isotropic Gaussian centered at (x, y)."""
    h, w = canvas.shape
    r = int(np.ceil(3 * sigma))
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    spot = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], spot)


def make_image(n_spots: int, rng: np.random.Generator) -> np.ndarray:
    """One point-pattern image, intensity in [0, 1]."""
    img = np.zeros((SIZE, SIZE), dtype=np.float64)
    # Keep spots away from the border so DFT shifts don't wrap them.
    margin = 16
    coords = rng.integers(margin, SIZE - margin, size=(n_spots, 2))
    for y, x in coords:
        gaussian_spot(img, int(x), int(y), sigma=SPOT_SIGMA)
    return img


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "klasse2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i, n in enumerate(COUNTS[:N_IMAGES]):
        img = make_image(n, rng)
        u8 = (img * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(out_dir / f"points_{n:03d}.png")
        print(f"wrote {out_dir / f'points_{n:03d}.png'}  ({n} spots)")


if __name__ == "__main__":
    main()
