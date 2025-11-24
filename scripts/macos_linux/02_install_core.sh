#!/usr/bin/env bash
set -euo pipefail

echo "[Step 2/4] Install core Python packages into .venv"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY. Run scripts/macos_linux/01_create_venv.sh first." >&2
  exit 1
fi

# Upgrade packaging tools
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

# Install PyTorch + torchvision (MPS supported wheels on Apple Silicon)
echo "Installing torch & torchvision ..."
"$VENV_PY" -m pip install --upgrade torch torchvision

echo "Installing nnUNet and supporting libraries ..."
"$VENV_PY" -m pip install --upgrade \
  nnunetv2 monai nibabel numpy scipy SimpleITK scikit-image matplotlib tqdm blosc2

echo "Done."
