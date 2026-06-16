"""Captcha solver for 5-character (case-insensitive alphanumeric) image captchas.

A small, retrainable deep-learning tool. Drop more labelled examples into the
training folder (filename without extension = the solution) and retrain to
improve accuracy.
"""

from .config import ALPHABET, NUM_CHARS, IMG_W, IMG_H

__all__ = ["ALPHABET", "NUM_CHARS", "IMG_W", "IMG_H"]
__version__ = "0.1.0"
