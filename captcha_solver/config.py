"""Project-wide constants and the label alphabet.

The captchas are 5 characters, case-insensitive, drawn from digits + letters.
We fold everything to lowercase, giving 36 classes.
"""

from __future__ import annotations

import string

# Alphabet: 0-9 then a-z. Index in this string == class id.
ALPHABET: str = string.digits + string.ascii_lowercase  # "0123456789abc...z"
NUM_CLASSES: int = len(ALPHABET)  # 36
NUM_CHARS: int = 5                # every captcha is exactly 5 characters

# Native captcha resolution (width x height). Images are resized to this.
IMG_W: int = 200
IMG_H: int = 50

# Maps for fast encode/decode.
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}


def encode_label(text: str) -> list[int]:
    """Turn a label string into a list of class indices (case-insensitive)."""
    text = text.strip().lower()
    if len(text) != NUM_CHARS:
        raise ValueError(
            f"label {text!r} has {len(text)} chars, expected {NUM_CHARS}"
        )
    try:
        return [CHAR_TO_IDX[c] for c in text]
    except KeyError as e:
        raise ValueError(f"label {text!r} contains out-of-alphabet char {e}") from e


def decode_indices(indices) -> str:
    """Turn a sequence of class indices back into a string (full alphabet)."""
    return "".join(IDX_TO_CHAR[int(i)] for i in indices)


def is_valid_label(text: str) -> bool:
    """True if `text` is a usable 5-char alphanumeric label."""
    text = text.strip().lower()
    return len(text) == NUM_CHARS and all(c in CHAR_TO_IDX for c in text)


def alphabet_from_labels(labels) -> str:
    """The sorted set of characters actually used across the given labels.

    This captcha type omits the ambiguous characters (0 1 g i l o q), so the
    effective alphabet learned from data is smaller than the full 36. Deriving
    it from the data both shrinks the classification problem and lets the
    alphabet grow automatically when new characters appear in added examples.
    """
    chars = set()
    for lbl in labels:
        chars.update(lbl.strip().lower())
    return "".join(sorted(chars))


class LabelCodec:
    """Maps between label strings and class indices for a specific alphabet.

    The CTC blank is index ``num_classes`` (one past the last real class).
    """

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        self.idx_to_char = {i: c for i, c in enumerate(alphabet)}
        self.num_classes = len(alphabet)
        self.blank = len(alphabet)

    def encode(self, text: str) -> list[int]:
        text = text.strip().lower()
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices) -> str:
        return "".join(self.idx_to_char[int(i)] for i in indices
                       if int(i) in self.idx_to_char)
