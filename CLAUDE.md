# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A retrainable solver for one specific captcha style: 200×50 px JPEGs, solid
light background, **exactly 5 case-insensitive alphanumeric characters**,
warped single-colour glyphs crossed by wavy noise lines. Training labels come
from filenames — `examples/2sbv3.jpg` → label `2sbv3` (lower-cased;
non-alphanumeric or non-5-char stems are skipped).

The **active alphabet is data-driven**, not the full 36. This captcha omits the
ambiguous characters `0 1 g i l o q`, so the 75 bundled examples (in `examples/`
+ `examples1/`) use only **29 classes** (`23456789abcdefhjkmnprstuvwxyz`). `alphabet_from_labels()` recomputes it from
the training labels each run, the model sizes its output to it, the synthetic
generator only draws those glyphs, and the checkpoint stores it. Adding examples
with a new character automatically expands the alphabet on the next train.
`config.LabelCodec` maps strings↔indices for whatever alphabet is in force;
`is_valid_label` (full 0-9a-z) is used only to filter filenames during discovery.

## Environment & commands

PyTorch is used in a project venv (the system Python is PEP-668 managed — do not
`pip install` globally). Python here is 3.14; `torchvision` is **not** installed
(augmentation is hand-rolled in numpy/PIL), so don't add imports of it.

This machine has an **NVIDIA RTX 4060 Ti (CUDA 13.2 driver)**; the venv has the
**CUDA build** `torch==2.12.0+cu130` (installed via the `cu130` index — see
`setup.sh`'s `TORCH_CUDA`). Training/inference default to `--device auto` (CUDA
if available, else CPU); GPU training is ~10× faster (the full run below is
~14 min on GPU vs ~tens of minutes on CPU). The `multiprocessing.Pool` ban
still holds (Py3.14+PIL crashes) — keep synth generation and DataLoaders at
`num_workers=0`.

```bash
TORCH_CUDA=cu130 ./setup.sh                 # GPU build (omit TORCH_CUDA for CPU-only)
source .venv/bin/activate                   # or prefix commands with .venv/bin/

python captcha.py train --data examples examples1   # the 75-image set; auto-GPU
python captcha.py train --synthetic 0       # real-images-only (skip pretraining)
python captcha.py train --repretrain        # ignore cached pretrained weights, redo stage 1
python captcha.py train --mix-synth 0       # disable mixing synthetic into fine-tune
python captcha.py train --full              # train on ALL real data, no val split
python captcha.py solve path/img.jpg        # solve; add --quiet (text only) or --json
python captcha.py eval                       # accuracy of saved model over ./examples
python captcha.py add img_x7k2p.jpg          # copy a labelled image into examples/, then retrain
```

Synthetic pretraining is **cached** to `models/pretrained.pt`, so the first
`train` is slow (~tens of minutes on CPU; the 2-layer BiLSTM over 5000 synthetic
samples dominates) but later retrains reuse it and jump straight to fine-tuning.
The cache is invalidated automatically if the alphabet changes (or with
`--repretrain`).

There is no test suite or linter. Validate changes by training and reading the
`[pretrain]`/`[finetune]` accuracy lines (reported as `full-string% / per-char%`)
and by `python captcha.py eval`. Quick end-to-end sanity check (fast):
`python captcha.py train --synthetic 256 --pretrain-epochs 1 --epochs 2 --out /tmp/m.pt`.

## Architecture (the big picture)

The central problem is **a few dozen real images is far too few** to learn from
directly. Several design choices exist specifically to overcome that — keep them
in mind before changing anything, because each was the fix for a concrete
failure mode:

1. **Color-invariant preprocessing** (`preprocess.py`). Every image is reduced to
   a single-channel *foreground map* = per-pixel colour distance from the
   estimated background colour, robustly normalised. Text becomes bright,
   background ~0, independent of the (highly varied) colour scheme. Both real and
   synthetic images go through this identical pipeline, so they share one
   representation. `preprocess_path()` / `standardize()` define the exact
   inference-time transform — training must match it.

2. **Two-stage training** (`train.py`). (a) **Pretrain** on thousands of
   synthetic captchas (`synth.py`) rendered mostly in the **real font (Trebuchet
   MS Bold)**, warped on the fly → the network learns to read 5 warped glyphs in
   the actual target font. (b) **Fine-tune** on the real images, *interleaving
   synthetic batches* (mixed fine-tune) so the few reals don't get memorised.
   Model selection and the checkpoint are driven by accuracy on the *real*
   validation split across BOTH stages, so the saved model is the best real-data
   model seen. Runs on GPU automatically (`--device auto`).

3. **CRNN + CTC** (`model.py`). A CNN reads the image as a left-to-right sequence;
   a BiLSTM + a **single shared** classifier (active-alphabet size + 1 CTC blank;
   currently 29+1) labels each timestep. The shared classifier is the key
   data-efficiency win over independent per-position heads (which overfit
   instantly: train ~99% / val ~10%). Decoding is greedy CTC collapse; length is
   not hard-constrained to 5. The CTC blank is index `num_classes` (last).

Fine-tuning also uses **heavy augmentation** (`augment.py`: affine + sinusoidal
wave + elastic warp in one resample, plus noise lines/blur/noise/erasing) ramped
by a **curriculum** (`strength` 0→1; warm up on clean glyphs first), and the real
dataset is **oversampled** (`CaptchaDataset(repeat=...)`) so each epoch yields
many gradient updates from few images.

`captcha.py` is the CLI dispatcher; `infer.py`'s `Solver` loads a checkpoint and
solves (decoding with the checkpoint's alphabet via `LabelCodec`). Checkpoints
(`models/captcha_net.pt`) bundle the model + alphabet + image size + metrics and
are **git-ignored**; they are saved **incrementally on every improvement** so an
interrupted run (this machine has crashed mid-training) keeps the best model so
far. Model selection (and the saved checkpoint) is driven by accuracy on the
*real* validation split across BOTH stages.

## Gotchas that bit us (don't re-break)

- **CTC gets stuck predicting all-blanks** unless it gets enough early gradient
  updates and a head start. Both matter: keep `repeat` oversampling (≈13
  updates/epoch, not ~2) and the negative blank-bias init in `CRNN.__init__`.
- **Synthetic must stay learnable.** Over-aggressive synthetic distortion
  (large rotation/overlap/thick lines) makes synth-loss plateau and kills
  transfer. Geometry is deliberately kept close to the real captchas.
- **Match the font — it's the #1 transfer lever.** The real captchas are
  **Trebuchet MS Bold**; `synth.py` renders mostly in it (plus similar sans for
  robustness). This single change took held-out per-char from ~35% to ~76–88%
  and made the model read real captchas *during pretraining*. Preprocessing is
  colour-invariant, so only glyph SHAPE/size/warp must match, not colour. If you
  ever face a new captcha style, re-identify its font before anything else.
- **Pretraining warps synthetic on the fly.** `SyntheticCaptchaDataset` renders
  clean maps once, then applies `augment()` per `__getitem__`, so pretraining
  reads *distorted* Trebuchet (matching the real warp) with endless variety.
- **Synthetic data is generated serially and cached** in `SyntheticCaptchaDataset`
  (~1.8s/1000). A `multiprocessing.Pool` path was removed — it crashed on this
  platform (Py3.14 + PIL). Don't reintroduce process pools for generation.
- **Don't hardcode 36 classes.** The model output size, synthetic generator, and
  decoder all key off the data-driven `LabelCodec`/alphabet. Treating it as the
  full `0-9a-z` re-introduces the `0/o`, `1/l/i` confusions and wastes capacity.
- **Full-string is the hard, data-limited metric.** With 75 images and the font
  matched, held-out per-char is ~76–88% but full-string is only ~45–65% (all 5
  chars must be right). The model still memorises the train set (≈100%);
  **mixed fine-tuning** (`--mix-synth`, default on) stops held-out full-string
  collapsing after ~ep10. Pushing full-string toward >90% genuinely needs more
  real data (this style's strong solvers use hundreds–1000 images).
- **Adding data to improve accuracy** is the intended path: drop more
  `<solution>.jpg` files into `examples/` (or another folder and pass
  `--data examples more_dir`) and retrain. More data → raise `--epochs`. The
  cached pretraining is reused, so retrains are fast.

## Session handoff — state & next steps (as of 2026-06-16, shipped)

**Status: built, trained, and SHIPPED.** The user accepted the accuracy and
asked to ship. The deployed model is `models/captcha_net.pt` (data-driven
29-class alphabet, trained on the 75 examples with a seed-1337 60/15 split).
`models/pretrained.pt` is the cached matched-font synthetic pretrain (reused by
retrains). Still **no git commits** — everything untracked; commit not requested.

**Achieved accuracy (honest):**
- Shipped model on its held-out split: **67% full-string / 88% per-char** (10/15).
- 6-fold cross-validation (non-mixed recipe): **~44% full-string / ~76% per-char**
  (high variance 27–60%; the 67% split was lucky). Mixed fine-tuning holds
  held-out full-string ~**55–60%** instead of collapsing — more robust, and now
  the default (`--mix-synth 2000`).
- `eval` over all 75 reports 93%/98% but is **optimistic** (includes the 60
  train images) — quote the cross-val numbers as the real generalization.

**What made it work (this session's wins):**
- **Font match = Trebuchet MS Bold** (the dominant lever; ~35%→~76–88% per-char).
- **GPU**: installed `torch==2.12.0+cu130`; `--device auto`; ~10× faster.
- **On-the-fly synth warp** during pretrain; **mixed fine-tune** to curb overfit.
- Added 30 examples → `examples1/` (75 total). More data is the durable lever.

**Dead-ends (don't repeat):** independent 5×softmax heads (memorise instantly);
CTC all-blank without oversampling + negative blank-bias; over-hard synthetic
geometry; `multiprocessing.Pool` / DataLoader workers on Py3.14+PIL (crash).

**To push accuracy further (if asked again):**
1. **More real data** is the #1 lever — a few hundred labelled `<solution>.jpg`
   in another folder, then `train --data examples examples1 moredir`. Full-string
   should climb well past today's ~55–60%.
2. Ship a `--full` model (train on all 75, no val split) for deployment once a
   val-split run confirms the recipe; keep a held-out folder for honest `eval`.
3. Minor: CTC sometimes collapses doubled letters (e.g. `cahh`→`cah`); more data
   / longer mixed fine-tune helps.

**Crash caveat:** this machine hard-crashes (kernel MCE / hardware) ~hourly and
wipes `/tmp` — keep logs/checkpoints in the repo, never train into `/tmp`.
Pretraining is crash-resumable (`models/pretrained.pt` stores `pretrain_epochs_done`).
