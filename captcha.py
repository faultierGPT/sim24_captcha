#!/usr/bin/env python3
"""Captcha solver — command line interface.

Subcommands:
  train   Train (or retrain) the model on labelled example images.
  solve   Solve one or more captcha images.
  eval    Measure accuracy of the saved model on a labelled folder.
  add     Add new labelled images to the training set (then retrain).

Run `python captcha.py <subcommand> -h` for details.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from captcha_solver.config import is_valid_label
from captcha_solver.train import DEFAULT_MODEL_PATH


def cmd_train(args) -> int:
    from captcha_solver.train import TrainConfig, train

    cfg = TrainConfig(
        data_dirs=args.data,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        dropout=args.dropout,
        repeat=args.repeat,
        synthetic=args.synthetic,
        pretrain_epochs=args.pretrain_epochs,
        mix_synth=args.mix_synth,
        synth_workers=args.synth_workers,
        repretrain=args.repretrain,
        device=args.device,
        full=args.full,
    )
    train(cfg)
    return 0


def _gather_images(paths) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in exts))
        else:
            out.append(p)
    return out


def cmd_solve(args) -> int:
    from captcha_solver.infer import Solver

    solver = Solver(model_path=args.model, device=args.device)
    images = _gather_images(args.images)
    if not images:
        print("No images given.", file=sys.stderr)
        return 2

    results = solver.solve_batch(images)
    if args.json:
        print(json.dumps(
            [{"path": p, "text": t, "confidence": round(c, 4)} for p, t, c in results],
            indent=2,
        ))
    elif args.quiet:
        for _, t, _ in results:
            print(t)
    else:
        for p, t, c in results:
            print(f"{p}: {t}  (conf {c:.2f})")
    return 0


def cmd_eval(args) -> int:
    from captcha_solver.dataset import discover_samples
    from captcha_solver.infer import Solver

    solver = Solver(model_path=args.model, device=args.device)
    samples, skipped = discover_samples(args.data)
    if not samples:
        print(f"No labelled images in {args.data}", file=sys.stderr)
        return 2

    results = solver.solve_batch([p for p, _ in samples])
    n = len(samples)
    correct = 0
    char_correct = char_total = 0
    mistakes = []
    for (path, gold), (_, pred, conf) in zip(samples, results):
        ok = pred == gold
        correct += ok
        for a, b in zip(gold, pred.ljust(len(gold))):
            char_total += 1
            char_correct += (a == b)
        if not ok:
            mistakes.append((path, gold, pred, conf))

    print(f"Evaluated {n} images ({skipped} skipped)")
    print(f"  full-string accuracy: {correct}/{n} = {correct/n*100:.1f}%")
    print(f"  per-char    accuracy: {char_correct}/{char_total} = "
          f"{char_correct/char_total*100:.1f}%")
    if mistakes and not args.quiet:
        print(f"\n  {len(mistakes)} mistakes (gold -> pred):")
        for path, gold, pred, conf in mistakes:
            print(f"    {Path(path).name:14s} {gold} -> {pred}  (conf {conf:.2f})")
    return 0


def cmd_add(args) -> int:
    dest = Path(args.to)
    dest.mkdir(parents=True, exist_ok=True)
    images = [Path(p) for p in args.images]

    if args.label is not None:
        if len(images) != 1:
            print("--label can only be used with a single image.", file=sys.stderr)
            return 2
        if not is_valid_label(args.label):
            print(f"Invalid label {args.label!r}: need 5 alphanumeric chars.",
                  file=sys.stderr)
            return 2
        target = dest / f"{args.label.lower()}{images[0].suffix.lower()}"
        shutil.copy2(images[0], target)
        print(f"Added {target}")
        return 0

    added = skipped = 0
    for img in images:
        if not is_valid_label(img.stem):
            print(f"  skip {img.name}: filename is not a valid 5-char label",
                  file=sys.stderr)
            skipped += 1
            continue
        target = dest / f"{img.stem.lower()}{img.suffix.lower()}"
        shutil.copy2(img, target)
        added += 1
    print(f"Added {added} image(s) to {dest} ({skipped} skipped). "
          f"Retrain with:  python captcha.py train")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="captcha.py", description="Solve 5-char alphanumeric image captchas."
    )
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="train / retrain the model")
    pt.add_argument("--data", nargs="+", default=["examples"],
                    help="one or more folders of labelled images (default: examples)")
    pt.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="output model path")
    pt.add_argument("--epochs", type=int, default=150, help="fine-tune epochs on real data")
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1.5e-3)
    pt.add_argument("--val-frac", type=float, default=0.2)
    pt.add_argument("--seed", type=int, default=1337)
    pt.add_argument("--dropout", type=float, default=0.3)
    pt.add_argument("--repeat", type=int, default=12,
                    help="oversample factor (gradient updates per epoch ~ n*repeat/batch)")
    pt.add_argument("--synthetic", type=int, default=5000,
                    help="number of precomputed synthetic captchas (0 disables pretraining)")
    pt.add_argument("--pretrain-epochs", type=int, default=30)
    pt.add_argument("--mix-synth", type=int, default=2000,
                    help="synthetic samples mixed into fine-tuning to curb "
                         "overfitting (0 disables)")
    pt.add_argument("--synth-workers", type=int, default=8,
                    help="parallel workers generating synthetic data")
    pt.add_argument("--repretrain", action="store_true",
                    help="force re-running synthetic pretraining (ignore cached weights)")
    pt.add_argument("--device", default="auto",
                    help="auto (CUDA if available), cuda, or cpu")
    pt.add_argument("--full", action="store_true",
                    help="train on ALL data (no validation split) for the final model")
    pt.set_defaults(func=cmd_train)

    ps = sub.add_parser("solve", help="solve captcha image(s)")
    ps.add_argument("images", nargs="+", help="image files or folders")
    ps.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ps.add_argument("--device", default="auto")
    ps.add_argument("--json", action="store_true", help="output JSON")
    ps.add_argument("--quiet", action="store_true",
                    help="print only the predicted text, one per line")
    ps.set_defaults(func=cmd_solve)

    pe = sub.add_parser("eval", help="evaluate accuracy on a labelled folder")
    pe.add_argument("--data", nargs="+", default=["examples"])
    pe.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    pe.add_argument("--device", default="auto")
    pe.add_argument("--quiet", action="store_true", help="don't list individual mistakes")
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("add", help="add labelled images to the training set")
    pa.add_argument("images", nargs="+", help="image files (filename = solution)")
    pa.add_argument("--label", default=None,
                    help="explicit label for a single image whose filename isn't the solution")
    pa.add_argument("--to", default="examples", help="destination folder (default: examples)")
    pa.set_defaults(func=cmd_add)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
