"""On-the-fly data augmentation for the foreground maps.

With only a few dozen real captchas, the model would memorise them instantly.
Strong, captcha-specific augmentation turns each original into an effectively
unlimited stream of variants so the network learns the *font glyphs* and how
they warp, rather than the specific pixels of 45 images.

Augmentation is scaled by a `strength` in [0, 1] so training can use a
curriculum: warm up on clean images (strength 0) so the CTC model first learns
to read glyphs, then ramp strength up to 1 for robustness. At strength 0 this
module is a no-op.

Everything operates on a single-channel float map in [0, 1] (foreground bright,
background ~0), the same representation produced by `preprocess.foreground_map`.
All randomness flows through a passed-in numpy Generator for reproducibility.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import IMG_H, IMG_W


def _smooth_field(rng: np.random.Generator, amp: float, coarse=(6, 18)) -> np.ndarray:
    """A smooth random displacement field of shape (H, W), values ~[-amp, amp]."""
    ch, cw = coarse
    low = rng.uniform(-1.0, 1.0, size=(ch, cw)).astype(np.float32)
    field = np.asarray(
        Image.fromarray(low, mode="F").resize((IMG_W, IMG_H), Image.BILINEAR),
        dtype=np.float32,
    )
    return field * amp


def _bilinear_sample(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Sample `img` (H,W) at fractional coords (map_x, map_y). OOB -> 0."""
    h, w = img.shape
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx = map_x - x0
    wy = map_y - y0

    def gather(yy, xx):
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        xc = np.clip(xx, 0, w - 1)
        yc = np.clip(yy, 0, h - 1)
        return np.where(valid, img[yc, xc], 0.0)

    top = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
    bot = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def _warp(rng: np.random.Generator, fg: np.ndarray, s: float) -> np.ndarray:
    """Combined affine + sinusoidal wave + elastic warp in a single resample.

    Magnitudes scale with strength `s`; at s=0 the transform is the identity.
    """
    h, w = fg.shape
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    # --- affine params (these describe dest -> src sampling) ---
    theta = np.deg2rad(rng.uniform(-12, 12) * s)
    scale = 1.0 + (rng.uniform(0.85, 1.12) - 1.0) * s
    shear_x = rng.uniform(-0.20, 0.20) * s
    shear_y = rng.uniform(-0.08, 0.08) * s
    tx = rng.uniform(-8, 8) * s
    ty = rng.uniform(-5, 5) * s
    ct, st = np.cos(theta), np.sin(theta)

    dx = xx - cx
    dy = yy - cy
    sx = (ct * dx - st * dy) / scale + shear_x * dy
    sy = (st * dx + ct * dy) / scale + shear_y * dx
    src_x = cx + sx + tx
    src_y = cy + sy + ty

    # --- sinusoidal wave (mimics the captcha's wavy baseline) ---
    if rng.random() < 0.8 * s:
        amp_y = rng.uniform(1.5, 4.0) * s
        period = rng.uniform(40, 120)
        phase = rng.uniform(0, 2 * np.pi)
        src_y = src_y + amp_y * np.sin(2 * np.pi * xx / period + phase)
    if rng.random() < 0.4 * s:
        amp_x = rng.uniform(1.0, 3.0) * s
        period = rng.uniform(20, 60)
        phase = rng.uniform(0, 2 * np.pi)
        src_x = src_x + amp_x * np.sin(2 * np.pi * yy / period + phase)

    # --- elastic distortion ---
    if rng.random() < 0.6 * s:
        src_x = src_x + _smooth_field(rng, amp=rng.uniform(1.0, 3.0) * s)
        src_y = src_y + _smooth_field(rng, amp=rng.uniform(1.0, 2.5) * s)

    return _bilinear_sample(fg, src_x, src_y)


def _add_lines(rng: np.random.Generator, fg: np.ndarray, s: float) -> np.ndarray:
    """Draw 0-N bright wavy noise lines, like the strike-through curves."""
    max_lines = 1 + int(round(3 * s))
    n = rng.integers(0, max_lines)
    if n == 0:
        return fg
    img = Image.fromarray((np.clip(fg, 0, 1) * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    for _ in range(int(n)):
        amp = rng.uniform(2, 10)
        period = rng.uniform(40, 160)
        phase = rng.uniform(0, 2 * np.pi)
        y0 = rng.uniform(IMG_H * 0.2, IMG_H * 0.8)
        thickness = int(rng.integers(1, 3))
        val = int(rng.integers(150, 256))
        xs = np.arange(0, IMG_W, 2)
        ys = y0 + amp * np.sin(2 * np.pi * xs / period + phase)
        pts = list(zip(xs.tolist(), ys.tolist()))
        draw.line(pts, fill=val, width=thickness, joint="curve")
    return np.asarray(img, dtype=np.float32) / 255.0


def augment(rng: np.random.Generator, fg: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply the augmentation pipeline to a [0,1] foreground map.

    `strength` in [0,1] scales every effect; 0 returns the map unchanged.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return fg.astype(np.float32)

    fg = _warp(rng, fg, s)
    fg = _add_lines(rng, fg, s)

    # brightness / contrast jitter
    if rng.random() < 0.7 * s:
        gain = 1.0 + (rng.uniform(0.7, 1.3) - 1.0) * s
        bias = rng.uniform(-0.08, 0.08) * s
        fg = np.clip(fg * gain + bias, 0, 1)

    # blur
    if rng.random() < 0.3 * s:
        radius = rng.uniform(0.4, 1.0) * s
        img = Image.fromarray((fg * 255).astype(np.uint8), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        fg = np.asarray(img, dtype=np.float32) / 255.0

    # additive gaussian noise
    if rng.random() < 0.6 * s:
        sigma = rng.uniform(0.01, 0.06) * s
        fg = np.clip(fg + rng.normal(0, sigma, size=fg.shape), 0, 1)

    # random erasing (occlusion)
    if rng.random() < 0.3 * s:
        ew = int(rng.integers(6, 26))
        eh = int(rng.integers(6, 22))
        ex = int(rng.integers(0, max(1, IMG_W - ew)))
        ey = int(rng.integers(0, max(1, IMG_H - eh)))
        fg[ey : ey + eh, ex : ex + ew] = rng.uniform(0, 0.2)

    return fg.astype(np.float32)
