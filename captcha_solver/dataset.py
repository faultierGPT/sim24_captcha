"""Dataset discovery and the torch Dataset that feeds the model.

Labels come from filenames: ``2sbv3.jpg`` -> label ``2sbv3``. Anything whose
stem is not a valid 5-char alphanumeric label is skipped (with a count), so the
training folder can also hold unrelated files safely.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment
from .config import NUM_CHARS, LabelCodec, encode_label, is_valid_label
from .preprocess import foreground_map, load_rgb, standardize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def discover_samples(dirs) -> tuple[list[tuple[str, str]], int]:
    """Find (path, label) pairs under one or more directories.

    Returns (samples, n_skipped). Labels are lower-cased.
    """
    if isinstance(dirs, (str, os.PathLike)):
        dirs = [dirs]
    samples: list[tuple[str, str]] = []
    skipped = 0
    seen: set[str] = set()
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = p.stem
            if is_valid_label(stem):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    samples.append((str(p), stem.lower()))
            else:
                skipped += 1
    return samples, skipped


def make_splits(samples, val_frac: float, seed: int):
    """Shuffle and split samples into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = int(round(len(samples) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [samples[i] for i in range(len(samples)) if i not in val_idx]
    val = [samples[i] for i in range(len(samples)) if i in val_idx]
    return train, val


class CaptchaDataset(Dataset):
    """Yields (image_tensor[1,H,W], label_tensor[NUM_CHARS])."""

    def __init__(self, samples, train: bool, seed: int = 0, repeat: int = 1,
                 codec: LabelCodec | None = None):
        self.train = train
        self.repeat = max(1, int(repeat))
        self.samples = list(samples)
        encode = codec.encode if codec is not None else encode_label
        # Foreground maps are tiny (50x200 float); precompute & cache once.
        self._fg = [foreground_map(load_rgb(p)) for p, _ in self.samples]
        self._labels = [torch.tensor(encode(lbl), dtype=torch.long)
                        for _, lbl in self.samples]
        self._rng = np.random.default_rng(seed + (1 if train else 7))
        self.strength = 1.0  # augmentation strength; ramped by the training loop

    def set_strength(self, s: float) -> None:
        self.strength = float(s)

    def __len__(self) -> int:
        # Oversample so each epoch yields many gradient updates (and, when
        # training, many fresh augmented variants) from few real images.
        return len(self.samples) * self.repeat

    def label_text(self, i: int) -> str:
        return self.samples[i % len(self.samples)][1]

    def __getitem__(self, i: int):
        i = i % len(self.samples)
        fg = self._fg[i]
        if self.train:
            fg = augment(self._rng, fg.copy(), strength=self.strength)
        x = standardize(fg)[None, :, :]
        return torch.from_numpy(x), self._labels[i]
