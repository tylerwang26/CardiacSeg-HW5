#!/bin/bash
# CardiacSeg Environment Setup Script for macOS/Linux
# This script installs Python packages and sets up nnU-Net environment variables

set -e  # Exit on error

echo "========================================"
echo "CardiacSeg Environment Setup (macOS)"
echo "========================================"
echo ""

# 1. Check Python installation
echo "[1/5] Checking Python installation..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+ from:"
    echo "  https://www.python.org/downloads/"
    echo "  or use: brew install python@3.11"
    exit 1
fi

PYTHON_CMD=$(command -v python3)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "✓ Using Python: $PYTHON_CMD ($PYTHON_VERSION)"

# 1b. Create and switch to project venv (.venv)
echo ""
echo "[1b/5] Creating/Using virtual environment (.venv)..."
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: venv python not found at $VENV_PYTHON" >&2
    exit 1
fi
PYTHON_CMD="$VENV_PYTHON"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "✓ Using venv Python: $PYTHON_CMD ($PYTHON_VERSION)"

# 2. Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# 3. Install required Python packages
echo ""
echo "[3/5] Installing Python packages (this may take a few minutes)..."

# Detect if running on Apple Silicon (M1/M2/M3/M4)
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (M1/M2/M3/M4). Installing MPS-enabled PyTorch."
    echo "  Installing torch and torchvision..."
    $PYTHON_CMD -m pip install --upgrade torch torchvision
    echo "  ✓ torch/torchvision installed (Apple Silicon MPS support)"
else
    echo "Installing CPU-only PyTorch."
    $PYTHON_CMD -m pip install --upgrade torch torchvision
    echo "  ✓ torch/torchvision (CPU) installed"
fi

# Install other required packages
PACKAGES=(
    "nnunetv2"
    "monai"
    "nibabel"
    "numpy"
    "scipy"
    "SimpleITK"
    "scikit-image"
    "matplotlib"
    "tqdm"
    "blosc2"
)

for package in "${PACKAGES[@]}"; do
    echo "  Installing $package..."
    if $PYTHON_CMD -m pip install "$package"; then
        echo "    ✓ $package installed successfully"
    else
        echo "    ✗ ERROR: $package installation failed"
        exit 1
    fi
done

# 4. Set nnU-Net environment variables
echo ""
echo "[4/5] Setting nnU-Net environment variables..."

# Use current directory as base
BASE_DIR="$PWD"
export nnUNet_raw="$BASE_DIR/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/nnUNet_results"

# Also set uppercase versions (some versions of nnU-Net use these)
export NNUNET_RAW="$nnUNet_raw"
export NNUNET_PREPROCESSED="$nnUNet_preprocessed"
export NNUNET_RESULTS="$nnUNet_results"

echo "✓ Environment variables set:"
echo "  nnUNet_raw = $nnUNet_raw"
echo "  nnUNet_preprocessed = $nnUNet_preprocessed"
echo "  nnUNet_results = $nnUNet_results"

# Create directories if they don't exist
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"
echo "✓ Created/verified nnU-Net directories"

# 5. Verify installation
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"

TEST_PACKAGES=("torch" "nnunetv2" "monai" "nibabel")
for pkg in "${TEST_PACKAGES[@]}"; do
    echo -n "Checking $pkg... "
    if $PYTHON_CMD -c "import importlib; m=importlib.import_module('$pkg'); print('✓ Version:', getattr(m,'__version__','OK'))" 2>/dev/null; then
        :  # Success message already printed
    else
        echo "✗ ERROR: Not installed"
        exit 1
    fi
done

# Report backend status
echo ""
echo "Backend status:"
$PYTHON_CMD << 'EOF'
import torch
print('torch version:', torch.__version__)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon GPU) available: True')
    print('Using Apple Silicon GPU acceleration')
elif torch.cuda.is_available():
    print('CUDA available:', torch.cuda.is_available())
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Using CPU backend')
EOF

# 6. Complete
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Notes:"
echo "1. Environment variables are only valid in current shell session"
echo "2. A Python virtual environment has been created at .venv"
echo "   Activate it with: source .venv/bin/activate  (bash/zsh)"
echo "3. To make nnU-Net variables permanent, add these lines to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "   export nnUNet_raw=\"$nnUNet_raw\""
echo "   export nnUNet_preprocessed=\"$nnUNet_preprocessed\""
echo "   export nnUNet_results=\"$nnUNet_results\""
echo ""
echo "You can now run:"
echo "  .venv/bin/python rename_dataset.py    (Rename dataset files)"
echo "  .venv/bin/python nnunet_train.py      (Train model)"
echo "  .venv/bin/python nnunet_infer.py      (Inference)"
echo "  .venv/bin/python nnunet_evaluate.py   (Evaluation)"
echo "  # or first run: source .venv/bin/activate, then use python"
echo ""
