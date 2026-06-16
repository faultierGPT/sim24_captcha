"""Color-invariant preprocessing.

The captchas use many colour schemes (purple-on-pink, gold-on-beige,
green-on-white, blue-on-grey, ...). Training a tiny model to be invariant to
all of those from only a few dozen images is wasteful. Instead we collapse
every image to a single-channel "foreground map": how far each pixel's colour
is from the background colour. Text (and noise) become bright; the flat
background becomes ~0 — regardless of the actual colours used.

This is the single biggest sample-efficiency win for this dataset.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .config import IMG_H, IMG_W


def load_rgb(path) -> Image.Image:
    """Load an image as RGB at the native captcha resolution."""
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_W, IMG_H):
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    return img


def estimate_bg_color(arr: np.ndarray) -> np.ndarray:
    """Estimate the background colour from the image border.

    The border is almost always pure background, so its median colour is a
    robust estimate that ignores the text in the middle.
    """
    h, w, _ = arr.shape
    b = max(2, h // 12)  # border thickness
    border = np.concatenate(
        [
            arr[:b, :, :].reshape(-1, 3),
            arr[-b:, :, :].reshape(-1, 3),
            arr[:, :b, :].reshape(-1, 3),
            arr[:, -b:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border, axis=0)


def foreground_map(img: Image.Image) -> np.ndarray:
    """Return an (H, W) float32 array in [0, 1]; foreground (text) is bright.

    Steps: distance of each pixel's colour from the estimated background,
    then robust normalisation by the 99th percentile so a single very-bright
    noise line cannot compress the text's dynamic range.
    """
    arr = np.asarray(img, dtype=np.float32)
    bg = estimate_bg_color(arr)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))  # (H, W) euclidean colour dist

    # Robust per-image normalisation to use the full [0,1] range.
    hi = np.percentile(dist, 99.0)
    if hi < 1e-3:
        return np.zeros_like(dist, dtype=np.float32)
    out = np.clip(dist / hi, 0.0, 1.0)
    return out.astype(np.float32)


def to_uint8_l(fg: np.ndarray) -> Image.Image:
    """Convert a [0,1] foreground map to an 'L' (grayscale) PIL image."""
    return Image.fromarray((np.clip(fg, 0.0, 1.0) * 255).astype(np.uint8), mode="L")


# Standardisation stats for the foreground map. Foreground is sparse so the
# mean is low; these constants keep inputs roughly zero-mean/unit-variance and
# are fixed (not data-dependent) so train and inference always match.
_MEAN = 0.15
_STD = 0.30


def standardize(fg: np.ndarray) -> np.ndarray:
    """Standardise a [0,1] foreground map to ~zero-mean/unit-variance."""
    return ((fg - _MEAN) / _STD).astype(np.float32)


def preprocess_path(path) -> np.ndarray:
    """Full deterministic preprocessing used at inference time.

    Returns a standardised (1, H, W) float32 array ready for the model.
    """
    fg = foreground_map(load_rgb(path))
    return standardize(fg)[None, :, :]
