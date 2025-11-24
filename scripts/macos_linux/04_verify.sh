#!/usr/bin/env bash
set -euo pipefail

echo "[Step 4/4] Verify installation"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY" >&2
  exit 1
fi

"$VENV_PY" - <<'PY'
import importlib, sys
mods = [
    ("torch", "__version__"),
    ("monai", "__version__"),
    ("nibabel", "__version__"),
    ("nnunetv2", None),
]
for m, attr in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, attr) if attr else "imported"
        print(f"{m}: {ver}")
    except Exception as e:
        print(f"{m}: FAILED - {e}")

try:
    import torch
    print("torch version:", torch.__version__)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS available: True (Apple Silicon)")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch backend check failed:", e)
PY

echo "CLI: $(command -v nnUNetv2_plan_and_preprocess || echo 'nnUNetv2_plan_and_preprocess not found in PATH (will still be usable via .venv/bin if installed as script)')"

echo "Done."
