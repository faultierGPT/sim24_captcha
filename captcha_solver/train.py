"""Training loop.

Two-stage training for data efficiency:

  1. **Pretrain** on thousands of synthetic captchas (same visual style, many
     fonts) so the CRNN learns the general task of reading five warped glyphs.
  2. **Fine-tune** on the real labelled images with an augmentation curriculum
     to adapt to the specific real font.

Throughout, model selection and the (durable, incremental) checkpoint are driven
by accuracy on the *real* held-out validation split — so the saved model is the
best real-data model seen across both stages. Set `synthetic=0` to train on the
real images only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import (IMG_H, IMG_W, NUM_CHARS, LabelCodec, alphabet_from_labels)
from .dataset import CaptchaDataset, discover_samples, make_splits
from .model import CRNN, ctc_greedy_decode, ctc_loss

DEFAULT_MODEL_PATH = Path("models/captcha_net.pt")


@dataclass
class TrainConfig:
    data_dirs: list = field(default_factory=lambda: ["examples"])
    out: str = str(DEFAULT_MODEL_PATH)
    epochs: int = 150              # fine-tune epochs on real data
    batch_size: int = 32
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.2
    seed: int = 1337
    dropout: float = 0.3
    repeat: int = 12               # oversample factor for the real images
    synthetic: int = 5000          # number of (precomputed) synthetic images (0 = off)
    pretrain_epochs: int = 30
    synth_workers: int = 8
    finetune_lr_scale: float = 0.5  # fine-tune LR = lr * this
    mix_synth: int = 2000          # synthetic samples mixed into fine-tuning (0 = off)
    repretrain: bool = False       # force re-running synthetic pretraining
    device: str = "auto"           # "auto" -> CUDA if available, else CPU
    full: bool = False             # train on ALL real data (no held-out val)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, float]:
    """Return (char_acc, string_acc, mean_loss) over a loader."""
    model.eval()
    n_chars = n_strings = 0
    correct_chars = correct_strings = 0
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += ctc_loss(logits, y, model.blank).item()
        n_batches += 1
        seqs, _ = ctc_greedy_decode(logits, model.blank)
        for i in range(y.shape[0]):
            gold = y[i].tolist()
            pred = seqs[i]
            n_strings += 1
            correct_strings += int(pred == gold)
            for pos in range(len(gold)):
                n_chars += 1
                if pos < len(pred) and pred[pos] == gold[pos]:
                    correct_chars += 1
    if n_strings == 0:
        return 0.0, 0.0, 0.0
    return (correct_chars / n_chars, correct_strings / n_strings,
            total_loss / max(1, n_batches))


def resolve_device(name: str) -> torch.device:
    """Map "auto" to CUDA when available, else honour the explicit choice."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def train(cfg: TrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        log(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        log("Device: cpu")

    samples, skipped = discover_samples(cfg.data_dirs)
    if not samples:
        raise SystemExit(
            f"No labelled captchas found in {cfg.data_dirs}. "
            "Filenames must be the 5-char solution, e.g. 2sbv3.jpg"
        )
    log(f"Found {len(samples)} labelled images "
        f"({skipped} non-matching files skipped) in {cfg.data_dirs}")

    if cfg.full or len(samples) < 8:
        train_samples, val_samples = samples, []
        log("Training on ALL real samples (no validation split)."
            if cfg.full else "Few samples: training on all (no validation split).")
    else:
        train_samples, val_samples = make_splits(samples, cfg.val_frac, cfg.seed)
    log(f"Real train: {len(train_samples)}  Real val: {len(val_samples)}")

    # The active alphabet is derived from the data: this captcha omits the
    # ambiguous characters (0 1 g i l o q), so the model only needs the classes
    # that actually occur. Adding examples with new characters expands it.
    alphabet = alphabet_from_labels([lbl for _, lbl in samples])
    codec = LabelCodec(alphabet)
    log(f"Active alphabet ({codec.num_classes} classes): {alphabet!r}")

    train_ds = CaptchaDataset(train_samples, train=True, seed=cfg.seed,
                              repeat=cfg.repeat, codec=codec)
    train_eval_ds = CaptchaDataset(train_samples, train=False, seed=cfg.seed, codec=codec)
    val_ds = (CaptchaDataset(val_samples, train=False, seed=cfg.seed, codec=codec)
              if val_samples else None)

    drop_last = (len(train_ds) % cfg.batch_size == 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=drop_last, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False) if val_ds else None

    model = CRNN(num_classes=codec.num_classes, dropout=cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: CRNN+CTC, {n_params/1e6:.2f}M params, input {IMG_H}x{IMG_W}, "
        f"{codec.num_classes}+1 classes")

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    select_on = "val" if val_loader else "train"

    def save_checkpoint(state, metrics):
        torch.save({
            "arch": "crnn_ctc",
            "model_state": state,
            "alphabet": alphabet,
            "num_chars": NUM_CHARS,
            "num_classes": codec.num_classes,
            "img_w": IMG_W,
            "img_h": IMG_H,
            "dropout": cfg.dropout,
            "metrics": metrics,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "select_on": select_on,
        }, out_path)

    best = {"score": -1.0, "state": None, "metrics": {}}

    def consider(epoch: int, stage: str):
        """Evaluate on REAL data, log, and durably save if it's the best so far."""
        tr_char, tr_str, _ = evaluate(model, train_eval_loader, device)
        if val_loader:
            va_char, va_str, _ = evaluate(model, val_loader, device)
            score = va_str + 0.01 * va_char
            metrics = dict(train_char=tr_char, train_str=tr_str,
                           val_char=va_char, val_str=va_str, epoch=epoch, stage=stage)
            log(f"[{stage}] ep {epoch:4d}  "
                f"train {tr_str*100:5.1f}%/{tr_char*100:5.1f}%  "
                f"val {va_str*100:5.1f}%/{va_char*100:5.1f}%")
        else:
            score = tr_str + 0.01 * tr_char
            metrics = dict(train_char=tr_char, train_str=tr_str, epoch=epoch, stage=stage)
            log(f"[{stage}] ep {epoch:4d}  train {tr_str*100:5.1f}%/{tr_char*100:5.1f}%")
        if score >= best["score"]:
            best["score"] = score
            best["state"] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
            best["metrics"] = metrics
            save_checkpoint(best["state"], best["metrics"])
        return score

    def _step(x, y, opt):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = ctc_loss(model(x), y, model.blank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        return loss.item()

    def run_epoch(loader, opt):
        model.train()
        ep_loss, nb = 0.0, 0
        for x, y in loader:
            ep_loss += _step(x, y, opt)
            nb += 1
        return ep_loss / max(1, nb)

    def run_mixed_epoch(real_loader, synth_loader, opt):
        """Interleave one synthetic batch after each real batch (≈1:1).

        Continuing to train on (infinite) synthetic Trebuchet during fine-tuning
        stops the model memorising the few dozen real images: with only real
        data it hits train 100% within ~10 epochs and held-out accuracy becomes
        noisy/overfit. The synthetic stream keeps the shared classifier anchored
        to the general task while the real images adapt it to the exact rendering.
        """
        model.train()
        ep_loss, nb = 0.0, 0
        sit = iter(synth_loader)
        for x, y in real_loader:
            ep_loss += _step(x, y, opt); nb += 1
            try:
                sx, sy = next(sit)
            except StopIteration:
                sit = iter(synth_loader)
                sx, sy = next(sit)
            ep_loss += _step(sx, sy, opt); nb += 1
        return ep_loss / max(1, nb)

    t0 = time.time()

    # ---------------- Stage 1: synthetic pretraining ----------------
    # Synthetic pretraining is independent of the user's real images, so its
    # weights are cached. Subsequent retrains (e.g. after adding examples) reuse
    # them and skip straight to fine-tuning — unless the alphabet changed or
    # --repretrain is given.
    pretrained_path = out_path.parent / "pretrained.pt"

    def save_pretrained(done_epochs: int):
        # Saved periodically so a crash mid-pretraining (this machine has) can
        # resume instead of restarting all of stage 1.
        torch.save({"alphabet": alphabet, "num_classes": codec.num_classes,
                    "pretrain_epochs_done": done_epochs,
                    "target_epochs": cfg.pretrain_epochs,
                    "model_state": {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}},
                   pretrained_path)

    if cfg.synthetic > 0:
        from .synth import SyntheticCaptchaDataset, available_fonts
        done = 0
        if pretrained_path.exists() and not cfg.repretrain:
            try:
                c = torch.load(pretrained_path, map_location=device, weights_only=False)
                if c.get("alphabet") == alphabet:
                    model.load_state_dict(c["model_state"])
                    # Old caches lack the counter; treat them as complete.
                    done = int(c.get("pretrain_epochs_done", cfg.pretrain_epochs))
            except Exception:
                done = 0

        if done >= cfg.pretrain_epochs:
            log(f"Loaded cached synthetic-pretrained weights from {pretrained_path} "
                f"(skipping pretraining; use --repretrain to redo)")
            consider(0, "pretrain")
        else:
            if done > 0:
                log(f"Resuming pretraining from cached epoch {done}/{cfg.pretrain_epochs}")
            log(f"Pretraining on synthetic captchas "
                f"({cfg.synthetic} images, epochs {done+1}..{cfg.pretrain_epochs}, "
                f"{len(available_fonts())} fonts)")
            synth_ds = SyntheticCaptchaDataset(size=cfg.synthetic, alphabet=alphabet,
                                               codec=codec, seed=cfg.seed,
                                               workers=cfg.synth_workers, log=log)
            synth_loader = DataLoader(synth_ds, batch_size=64, shuffle=True,
                                      num_workers=0, drop_last=True)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
            # Cosine over the remaining epochs (optimizer state isn't persisted).
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, cfg.pretrain_epochs - done))
            for epoch in range(done + 1, cfg.pretrain_epochs + 1):
                loss = run_epoch(synth_loader, opt)
                sched.step()
                log(f"[pretrain] ep {epoch:3d}/{cfg.pretrain_epochs}  synth-loss {loss:.3f}")
                consider(epoch, "pretrain")
                if epoch % 5 == 0 or epoch == cfg.pretrain_epochs:
                    save_pretrained(epoch)
            log(f"Cached synthetic-pretrained weights -> {pretrained_path}")
            del synth_loader, synth_ds

    # ---------------- Stage 2: fine-tune on real data ----------------
    warmup = max(5, cfg.epochs // 20) if cfg.synthetic == 0 else 0
    ramp = max(1, cfg.epochs // 4)

    def strength_at(ep: int) -> float:
        if ep <= warmup:
            return 0.0
        return min(1.0, (ep - warmup) / ramp)

    ft_lr = cfg.lr * (cfg.finetune_lr_scale if cfg.synthetic > 0 else 1.0)
    # Mix synthetic batches into fine-tuning to curb memorisation of the few real
    # images (the biggest single-split-variance / overfitting fix).
    mix_loader = None
    if cfg.mix_synth > 0 and cfg.synthetic > 0:
        from .synth import SyntheticCaptchaDataset
        mix_ds = SyntheticCaptchaDataset(size=cfg.mix_synth, alphabet=alphabet,
                                         codec=codec, seed=cfg.seed + 101)
        mix_loader = DataLoader(mix_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
        log(f"Fine-tuning on real+synthetic ({cfg.epochs} epochs, lr {ft_lr:.1e}, "
            f"mixing {cfg.mix_synth} synthetic)")
    else:
        log(f"Fine-tuning on real data ({cfg.epochs} epochs, lr {ft_lr:.1e})")
    opt = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    for epoch in range(1, cfg.epochs + 1):
        train_ds.set_strength(strength_at(epoch))
        if mix_loader is not None:
            loss = run_mixed_epoch(train_loader, mix_loader, opt)
        else:
            loss = run_epoch(train_loader, opt)
        sched.step()
        if epoch % 5 == 0 or epoch == cfg.epochs:
            log(f"[finetune] ep {epoch:4d}  loss {loss:.3f}  "
                f"aug {strength_at(epoch):.2f}  lr {sched.get_last_lr()[0]:.1e}")
            consider(epoch, "finetune")

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_checkpoint(best["state"], best["metrics"])

    elapsed = time.time() - t0
    log(f"Done in {elapsed:.0f}s. Best ({select_on}): {best['metrics']}")
    log(f"Saved model -> {out_path}")
    return best["metrics"]
