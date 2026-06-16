#!/usr/bin/env bash
# Create a virtual environment and install dependencies.
# Usage:  ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "Creating virtual environment in .venv ..."
"$PY" -m venv .venv

echo "Installing dependencies ..."
.venv/bin/pip install --upgrade pip -q

# Torch build: GPU greatly speeds up training. Set TORCH_CUDA to a PyTorch CUDA
# channel matching your NVIDIA driver to install a CUDA build, e.g.:
#   TORCH_CUDA=cu130 ./setup.sh      # CUDA 13.x driver (e.g. RTX 40-series)
#   TORCH_CUDA=cu126 ./setup.sh      # CUDA 12.6
# Leave unset for the small CPU-only build. Check your driver with `nvidia-smi`
# (CUDA Version) and pick the nearest channel at https://pytorch.org/get-started.
TORCH_CUDA="${TORCH_CUDA:-cpu}"
echo "  torch build: ${TORCH_CUDA}"
.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
.venv/bin/pip install -r requirements.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then train:           python captcha.py train   # uses the GPU automatically if present"
