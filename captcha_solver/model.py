"""The captcha recognition network: a CRNN trained with CTC.

Why CRNN+CTC rather than 5 independent softmax heads? With only a few dozen
training images, independent per-position heads each see ~1 example per class
and overfit instantly (train ~99% / val ~10%). A CRNN reads the image as a
left-to-right sequence and classifies every timestep with a *single shared*
recogniser, so all ~5 character instances per image — ~180 across the dataset —
train the same 36-way classifier. That sharing is the key data-efficiency win,
and it naturally handles the variable horizontal positions produced by warping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import NUM_CLASSES

# CTC blank occupies the last logit index.
BLANK_IDX = NUM_CLASSES  # 36


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.blank = num_classes

        # CNN: collapse height to 1, keep ~50 width timesteps.
        self.cnn = nn.Sequential(
            _conv(1, 64), nn.MaxPool2d(2),            # 50x200 -> 25x100
            _conv(64, 128), nn.MaxPool2d(2),          # -> 12x50
            _conv(128, 256), _conv(256, 256),
            nn.MaxPool2d((2, 1)),                     # -> 6x50  (height only)
            _conv(256, 256), nn.MaxPool2d((2, 1)),    # -> 3x50
            nn.Dropout2d(min(dropout, 0.1)),          # light conv dropout
        )
        self.rnn = nn.LSTM(256, rnn_hidden, num_layers=2, bidirectional=True,
                           dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes + 1)  # +1 = CTC blank
        # Discourage the degenerate "predict all blanks" minimum at init.
        nn.init.constant_(self.fc.bias[self.blank], -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (T, B, num_classes+1)."""
        f = self.cnn(x)            # (B, C, H', W')
        f = f.mean(dim=2)          # collapse height -> (B, C, T)
        f = f.permute(2, 0, 1)     # (T, B, C)
        f, _ = self.rnn(f)         # (T, B, 2*hidden)
        f = self.drop(f)
        return self.fc(f)          # (T, B, num_classes+1)


def ctc_loss(logits: torch.Tensor, targets: torch.Tensor, blank: int) -> torch.Tensor:
    """CTC loss. logits: (T,B,K)  targets: (B, S) all of equal length S."""
    t, b, _ = logits.shape
    s = targets.shape[1]
    log_probs = logits.log_softmax(2)
    input_lengths = torch.full((b,), t, dtype=torch.long)
    target_lengths = torch.full((b,), s, dtype=torch.long)
    return nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=blank, zero_infinity=True,
    )


@torch.no_grad()
def ctc_greedy_decode(logits: torch.Tensor, blank: int):
    """Greedy CTC decode (collapse repeats, drop blanks).

    Returns (list_of_index_lists, list_of_confidences). Confidence is the mean
    of the chosen timesteps' max-probabilities.
    """
    probs = logits.softmax(2)              # (T,B,K)
    maxp, idx = probs.max(2)               # (T,B)
    # Pull to CPU once (no-op on CPU) to avoid per-element GPU syncs in the loop.
    idx = idx.permute(1, 0).cpu()          # (B,T)
    maxp = maxp.permute(1, 0).cpu()        # (B,T)
    seqs, confs = [], []
    for b in range(idx.shape[0]):
        out, used_probs, prev = [], [], -1
        row = idx[b].tolist()
        for t, c in enumerate(row):
            if c != prev and c != blank:
                out.append(c)
                used_probs.append(float(maxp[b, t]))
            prev = c
        conf = float(sum(used_probs) / len(used_probs)) if used_probs else 0.0
        seqs.append(out)
        confs.append(conf)
    return seqs, confs
