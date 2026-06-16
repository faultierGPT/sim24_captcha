# Captcha Solver

Solves a specific style of **5-character, case-insensitive alphanumeric** image
captcha — 200×50 px, a single solid background colour, warped/rotated coloured
glyphs, and squiggly strike-through noise lines:

![example](examples/2sbv3.jpg) → `2sbv3`

It is a small, **retrainable** PyTorch model. The example images are the
training data; drop in more labelled examples and retrain to make it more
accurate.

## How it works

1. **Colour-invariant preprocessing.** Every image is reduced to a single-channel
   *foreground map*: the colour distance of each pixel from the estimated
   background colour. Text becomes bright, the background ~0 — regardless of the
   actual colours used. This lets a tiny model learn from few examples without
   having to also learn every colour scheme.
2. **Synthetic pretraining in the matching font.** A few dozen real images are
   too few to learn general glyph shapes (the model just memorises them). So we
   first **pretrain on thousands of synthetic captchas** rendered in the same
   style. The biggest lever for transfer is the **font**: the real captchas use
   **Trebuchet MS Bold** (identified by matching distinctive glyphs — the curly
   `f`, curved-bottom `t`, looped `k`, straight-descender `y`), so `synth.py`
   renders predominantly in that exact font (plus a few similar sans fonts for
   robustness), at the measured real glyph size, with realistic warp applied on
   the fly. Because the preprocessing is colour-invariant, matching *shape* is
   what matters — and matching it is what lets a model trained mostly on
   synthetic data read the real captchas. (Disable with `--synthetic 0`.)
3. **Heavy on-the-fly augmentation.** During fine-tuning each real image is
   warped (affine + sinusoidal wave + elastic) and gets random noise lines,
   blur, noise, brightness jitter and occlusion every epoch — turning a few
   dozen originals into an effectively unlimited stream of variants. An
   augmentation *curriculum* ramps this up from zero so training starts on clean
   glyphs and gets progressively harder.
4. **CRNN + CTC.** A convolutional backbone reads the image as a left-to-right
   sequence; a BiLSTM + **single shared classifier** (one class per alphabet
   character plus a CTC blank) labels each step. Sharing one recogniser across
   all character positions means every glyph instance in the dataset trains the
   same classifier — roughly 5× the effective data of independent per-position
   heads, which is what makes learning from only a few dozen images feasible.
6. **Mixed fine-tuning.** With only a few dozen real images, fine-tuning on them
   alone memorises them within ~10 epochs and held-out accuracy gets noisy. So
   fine-tuning interleaves synthetic Trebuchet batches with the real ones
   (`--mix-synth`, on by default), keeping the recogniser anchored to the
   general task while the real images adapt it to the exact rendering. This
   markedly stabilises held-out accuracy.
5. **Data-driven alphabet.** This captcha never uses the ambiguous characters
   `0 1 g i l o q`, so the model's output classes are derived from the
   characters actually present in your labels (29 for the bundled examples),
   not a fixed 36. This removes `0/o`, `1/l/i` confusions, and the alphabet
   grows automatically if you add examples containing new characters.

See `captcha_solver/` for the implementation:
`preprocess.py`, `augment.py`, `dataset.py`, `model.py`, `synth.py`,
`train.py`, `infer.py`.

## Setup

```bash
./setup.sh                      # creates .venv and installs torch (CPU), numpy, pillow
source .venv/bin/activate
```

**GPU (optional, much faster training).** Set `TORCH_CUDA` to the PyTorch CUDA
channel matching your NVIDIA driver (check `nvidia-smi`):

```bash
TORCH_CUDA=cu130 ./setup.sh     # CUDA 13.x driver (e.g. RTX 40-series); cu126 for CUDA 12.6, etc.
```

Training/inference then use the GPU automatically (`--device auto`, the default);
pass `--device cpu` to force CPU. Or install manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or .../whl/cu130 for GPU
pip install -r requirements.txt
```

## Usage

All commands go through `captcha.py`.

### Solve a captcha

```bash
python captcha.py solve path/to/image.jpg
# image.jpg: 2sbv3  (conf 0.97)

python captcha.py solve folder_of_images/          # batch
python captcha.py solve a.jpg b.jpg --quiet         # just the text, one per line
python captcha.py solve a.jpg --json                # machine-readable
```

The confidence is the product of the per-character softmax probabilities
(1.0 = fully certain).

### Train / retrain

```bash
python captcha.py train                       # synthetic pretrain + fine-tune on ./examples
python captcha.py train --epochs 250           # fine-tune longer
python captcha.py train --synthetic 10000 --pretrain-epochs 40   # more pretraining
python captcha.py train --synthetic 0          # skip pretraining (real images only)
python captcha.py train --full                 # use ALL real data, no validation split
```

Training runs in two stages — synthetic **pretrain** then real **fine-tune** —
and reports `train` / `val` accuracy as `full-string% / per-char%`. The best
model (by real validation accuracy) is saved incrementally to
`models/captcha_net.pt`, so progress survives an interrupted run.

The synthetic-pretraining weights are **cached** to `models/pretrained.pt`. The
first `train` is slow (tens of minutes on CPU), but later retrains — e.g. after
you add examples — reuse the cache and jump straight to fine-tuning. Use
`--repretrain` to force pretraining again (it's also redone automatically if the
alphabet changes).

### Evaluate

```bash
python captcha.py eval                    # accuracy of the saved model over ./examples
python captcha.py eval --data other_dir   # on a different labelled folder
```

## Adding more examples to improve accuracy

The model gets more accurate with more labelled data. Two ways to add images:

**1. Just drop files into `examples/`** named by their solution
(`<solution>.jpg`), then retrain:

```bash
cp newcaptcha_x7k2p.jpg examples/x7k2p.jpg
python captcha.py train
```

**2. Use the `add` helper** (validates the label and copies it in):

```bash
python captcha.py add x7k2p.jpg                    # filename is the solution
python captcha.py add screenshot.png --label x7k2p # explicit label
python captcha.py train
```

Tips for best results:
- Filenames are the labels and are **case-insensitive** (`9Jnew.jpg` == `9jnew`).
- Keep a separate held-out folder of labelled images and run
  `python captcha.py eval --data heldout/` to measure true accuracy.
- More data → raise `--epochs` and consider `--full` for the shipped model.
- You can train from multiple folders: `python captcha.py train --data examples more_examples`.

## Accuracy & limitations

- **Current accuracy (75 bundled examples).** On held-out captchas the model
  reads **~76–88% of characters** correctly and **solves ~45–65% of full
  strings**, depending on the split (single-split best: 10/15 = 67%
  full-string, 88% per-char; 6-fold cross-validation: ~76% per-char, and
  mixed fine-tuning holds full-string around 55–60% rather than letting it
  collapse). This is a large jump over the earlier ~35%-per-char baseline, and
  it came from **matching the real font (Trebuchet MS Bold)** so the synthetic
  pretraining transfers, plus mixed fine-tuning to curb overfitting.
- **Accuracy scales with data.** Full-string accuracy is the hard part (all 5
  characters must be right) and is still data-limited: this 200×50 wavy-captcha
  style is one where well-known solvers use **hundreds to a thousand** labelled
  images to reach >90%. The whole tool is built so accuracy climbs as you add
  real examples (see above) — that is the durable path to higher full-string
  rates.
- Tuned for this exact captcha style (200×50, solid light background, single
  text colour, 5 chars). Other styles need their own examples and a retrain.
- CPU training is fine — the dataset and model are small.
- The model files in `models/` are git-ignored by default.
