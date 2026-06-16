"""Inference: load a trained checkpoint and solve captchas."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import LabelCodec
from .model import CRNN, ctc_greedy_decode
from .preprocess import foreground_map, load_rgb, standardize
from .train import DEFAULT_MODEL_PATH


class Solver:
    """Loads a checkpoint once; solves single images or batches."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device: str = "auto"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Train one first:  python captcha.py train"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.alphabet = ckpt["alphabet"]
        self.codec = LabelCodec(self.alphabet)
        self.num_chars = ckpt["num_chars"]
        self.metrics = ckpt.get("metrics", {})
        self.model = CRNN(
            num_classes=ckpt["num_classes"],
            dropout=ckpt.get("dropout", 0.3),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _to_tensor(self, image_path) -> torch.Tensor:
        fg = foreground_map(load_rgb(image_path))
        return torch.from_numpy(standardize(fg)[None, :, :])  # (1,H,W)

    @torch.no_grad()
    def solve(self, image_path) -> tuple[str, float]:
        """Return (predicted_text, confidence in [0,1]) for one image."""
        x = self._to_tensor(image_path).unsqueeze(0).to(self.device)  # (1,1,H,W)
        seqs, confs = ctc_greedy_decode(self.model(x), self.model.blank)
        return self.codec.decode(seqs[0]), confs[0]

    @torch.no_grad()
    def solve_batch(self, image_paths) -> list[tuple[str, str, float]]:
        """Return [(path, text, confidence), ...]."""
        results: list[tuple[str, str, float]] = []
        tensors, valid_paths = [], []
        for p in image_paths:
            try:
                tensors.append(self._to_tensor(p))
                valid_paths.append(p)
            except Exception as e:  # unreadable image
                results.append((str(p), f"<error: {e}>", 0.0))
        if tensors:
            x = torch.stack(tensors).to(self.device)  # (N,1,H,W)
            seqs, confs = ctc_greedy_decode(self.model(x), self.model.blank)
            for i, p in enumerate(valid_paths):
                results.append((str(p), self.codec.decode(seqs[i]), confs[i]))
        return results
