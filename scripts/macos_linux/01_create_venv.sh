#!/usr/bin/env bash
set -euo pipefail

echo "[Step 1/4] Create project virtual environment (.venv)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Please install Python 3.8+ (e.g., brew install python@3.11)" >&2
  exit 1
fi

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  echo "Created venv at $VENV_DIR"
else
  echo "venv already exists at $VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY" >&2
  exit 1
fi

"$VENV_PY" --version

echo "Done."
