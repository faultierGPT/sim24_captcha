"""Synthetic captcha generation for pretraining.

Forty-five (now ~75) real images are too few to learn general glyph shapes — the
model memorises them (high train, near-random val). To fix that we pretrain on
thousands of *synthetic* captchas rendered in the same visual style (solid light
background, five warped single-colour glyphs, wavy noise lines).

The single most important factor for synthetic->real transfer is the **font**:
the real captchas are rendered in **Trebuchet MS Bold** (identified by matching
distinctive humanist glyphs — the curly `f`, curved-bottom `t`, looped `k`,
straight-descender `y`, open `e`). So the generator renders predominantly in
Trebuchet MS Bold, with a handful of similar clean sans fonts mixed in for
robustness. Glyph size, spacing and the wavy strike-through lines are matched to
measurements of the real images.

Because `preprocess.foreground_map` is colour-invariant, the text/line *colours*
don't matter after preprocessing — only shape, size, warp and line geometry do,
which is what we match here.

Realistic warp (affine + sine wave + elastic) and extra noise lines are applied
**on the fly** during pretraining (via `augment`), so the network learns to read
*distorted* Trebuchet directly rather than clean glyphs — closing most of the
gap to the real, warped captchas before fine-tuning even starts.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .augment import augment
from .config import ALPHABET, IMG_H, IMG_W, LabelCodec, NUM_CHARS
from .preprocess import foreground_map, standardize

_T = "/usr/share/fonts/truetype"

# The real captcha font. Rendered for ~half of all synthetic samples.
_PRIMARY_FONT = f"{_T}/msttcorefonts/Trebuchet_MS_Bold.ttf"

# Similar clean sans fonts, mixed in for robustness so the recogniser doesn't
# overfit to a single font's quirks. Kept deliberately readable (no decorative
# / display faces) so synthetic stays learnable and transfers.
_SIMILAR_FONTS = [
    f"{_T}/msttcorefonts/Trebuchet_MS.ttf",
    f"{_T}/msttcorefonts/Verdana_Bold.ttf",
    f"{_T}/msttcorefonts/Verdana.ttf",
    f"{_T}/msttcorefonts/Arial_Bold.ttf",
    f"{_T}/msttcorefonts/Comic_Sans_MS_Bold.ttf",
    f"{_T}/dejavu/DejaVuSans-Bold.ttf",
    f"{_T}/dejavu/DejaVuSans.ttf",
    f"{_T}/liberation/LiberationSans-Bold.ttf",
    f"{_T}/ubuntu/Ubuntu-B.ttf",
    f"{_T}/ubuntu/Ubuntu-M.ttf",
    f"{_T}/freefont/FreeSansBold.ttf",
    f"{_T}/freefont/FreeSans.ttf",
    f"{_T}/noto/NotoSans-Bold.ttf",
    f"{_T}/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
    f"{_T}/ttf-bitstream-vera/VeraBd.ttf",
]


def available_fonts() -> list[str]:
    """The synthetic font pool: primary (real) font first, then similar fonts."""
    fonts = [f for f in [_PRIMARY_FONT, *_SIMILAR_FONTS] if os.path.exists(f)]
    if not fonts:
        # Fall back to anything truetype on the system (keeps the tool working
        # if the msttcorefonts package isn't installed).
        import glob
        for d in ("/usr/share/fonts", "/usr/local/share/fonts"):
            fonts += glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True)[:20]
    return fonts


_FONTS = available_fonts()
_PRIMARY_AVAILABLE = os.path.exists(_PRIMARY_FONT)
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_SIZE_CACHE: dict[str, int] = {}


def _get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(path, size)
        _FONT_CACHE[key] = f
    return f


def _base_size(path: str, target_glyph_h: int = 37) -> int:
    """Pixel font size so a digit ('8') renders ~`target_glyph_h` px tall.

    Matches the measured real glyph size (~37 px tall digits/caps) regardless of
    each font's internal metrics, so every font in the pool draws at a
    consistent, real-looking scale.
    """
    cached = _SIZE_CACHE.get(path)
    if cached is not None:
        return cached
    f = ImageFont.truetype(path, 50)
    bb = f.getbbox("8")
    h = max(1, bb[3] - bb[1])
    size = max(18, round(50 * target_glyph_h / h))
    _SIZE_CACHE[path] = size
    return size


def _pick_font(rng: np.random.Generator) -> str:
    """Sample a font, biased heavily toward the real (Trebuchet) font."""
    if _PRIMARY_AVAILABLE and rng.random() < 0.5:
        return _PRIMARY_FONT
    return _FONTS[int(rng.integers(0, len(_FONTS)))]


def random_text(rng: np.random.Generator, alphabet: str = ALPHABET) -> str:
    chars = list(alphabet)
    return "".join(rng.choice(chars) for _ in range(NUM_CHARS))


def render_captcha(rng: np.random.Generator, text: str | None = None,
                   alphabet: str = ALPHABET) -> tuple[Image.Image, str]:
    """Render one clean synthetic captcha (RGB, 200x50) in the dataset's style.

    Geometry is matched to the real images: bold ~37 px glyphs filling the
    height, mild per-glyph rotation, glyphs spaced to span the width with the
    occasional light overlap, on a near-white background, crossed by a few thin
    wavy lines. Warp/extra noise is added later (on the fly) by `augment`.
    """
    if text is None:
        text = random_text(rng, alphabet)

    # Background: near-white, lightly tinted (real bg mean RGB ~247).
    base = int(rng.integers(236, 256))
    bg = tuple(int(np.clip(base + rng.integers(-12, 6), 220, 255)) for _ in range(3))
    img = Image.new("RGB", (IMG_W, IMG_H), bg)

    # Text colour: a single medium/dark colour (irrelevant after the
    # colour-invariant foreground map, but kept realistic).
    text_color = tuple(int(rng.integers(20, 150)) for _ in range(3))

    font_path = _pick_font(rng)
    size = _base_size(font_path) + int(rng.integers(-2, 4))
    font = _get_font(font_path, size)

    # Lay out glyphs left-to-right to span the width with mild jitter/overlap.
    n = len(text)
    margin = 6
    avg_w = (IMG_W - 2 * margin) / n
    x = margin + rng.uniform(-1, 2)
    for ch in text:
        tile = Image.new("RGBA", (size + 16, size + 20), (0, 0, 0, 0))
        gd = ImageDraw.Draw(tile)
        gd.text((8, 6), ch, font=font, fill=text_color + (255,))
        angle = rng.uniform(-14, 14)
        tile = tile.rotate(angle, expand=True, resample=Image.BILINEAR)
        # Trim to ink so vertical centring uses the real glyph extent.
        bbox = tile.getbbox()
        if bbox:
            tile = tile.crop(bbox)
        gw, gh = tile.size
        y = int((IMG_H - gh) / 2 + rng.uniform(-3, 3))
        img.paste(tile, (int(x), y), tile)
        x += avg_w * rng.uniform(0.9, 1.05)

    # Thin wavy strike-through lines (real captchas have ~1-3; augment adds more).
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(0, 3))):
        amp = rng.uniform(2, 8)
        period = rng.uniform(60, 180)
        phase = rng.uniform(0, 2 * np.pi)
        y0 = rng.uniform(IMG_H * 0.3, IMG_H * 0.7)
        line_color = tuple(int(rng.integers(20, 150)) for _ in range(3))
        xs = np.arange(0, IMG_W, 2)
        ys = y0 + amp * np.sin(2 * np.pi * xs / period + phase)
        draw.line(list(zip(xs.tolist(), ys.tolist())), fill=line_color,
                  width=1, joint="curve")

    return img, text


def make_clean(seed_i: int, alphabet: str):
    """Render+preprocess one synthetic sample to a *clean* foreground map.

    Returns (fg float32 [H,W] in [0,1], text). Warp/noise is applied later on
    the fly by the dataset, so the same clean render yields endless variants.
    """
    rng = np.random.default_rng(seed_i)
    img, text = render_captcha(rng, alphabet=alphabet)
    return foreground_map(img).astype(np.float32), text


# Backwards-compatible helper (clean, standardized tensor + label).
def make_sample(seed_i: int, alphabet: str, codec: LabelCodec):
    fg, text = make_clean(seed_i, alphabet)
    x = standardize(fg)[None, :, :]
    return torch.from_numpy(x), torch.tensor(codec.encode(text), dtype=torch.long)


class SyntheticCaptchaDataset(Dataset):
    """`size` synthetic captchas; clean maps rendered once, warped on the fly.

    Rendering dominates cost, so the `size` clean foreground maps are rendered
    serially up front and cached. Each `__getitem__` then applies fresh random
    `augment` warp + noise lines, so every epoch sees newly distorted variants
    of the same glyph set — infinite augmentation variety for the price of
    `size` renders. This matches the warped, line-crossed real captchas, so
    pretraining learns to read distorted Trebuchet directly.
    """

    def __init__(self, size: int, alphabet: str, codec: LabelCodec,
                 seed: int = 0, workers: int = 0, log=None,
                 strength_max: float = 0.75):
        self.size = int(size)
        self.seed = seed
        self.strength_max = float(strength_max)
        # Clean renders + encoded labels (process pools proved fragile on this
        # platform — Py3.14 + PIL — so generation is serial; ~1.8 s / 1000).
        self._clean: list[tuple[np.ndarray, torch.Tensor]] = []
        for i in range(self.size):
            fg, text = make_clean(seed * 1_000_003 + i, alphabet)
            self._clean.append((fg, torch.tensor(codec.encode(text), dtype=torch.long)))
        self._counter = 0
        if log:
            log(f"  generated {self.size} synthetic samples "
                f"({len(alphabet)}-char alphabet, on-the-fly warp)")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int):
        fg, label = self._clean[i]
        # Vary the warp per access (per epoch) for endless augmentation variety.
        self._counter += 1
        rng = np.random.default_rng((self.seed + 7919, i, self._counter))
        if rng.random() < 0.12:
            s = 0.0  # keep some clean samples so easy glyphs stay anchored
        else:
            s = float(rng.uniform(0.2, self.strength_max))
        aug = augment(rng, fg, s)
        return torch.from_numpy(standardize(aug)[None, :, :]), label
