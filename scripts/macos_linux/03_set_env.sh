#!/usr/bin/env bash
set -euo pipefail

echo "[Step 3/4] Set nnU-Net environment variables"

BASE_DIR="$(pwd)"
export nnUNet_raw="$BASE_DIR/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

# Write a helper file to source later
cat > .env.nnunet.sh <<EOF
# Source this file to export nnU-Net environment variables
export nnUNet_raw="$nnUNet_raw"
export nnUNet_preprocessed="$nnUNet_preprocessed"
export nnUNet_results="$nnUNet_results"
EOF

echo "Saved variables to .env.nnunet.sh (use: source .env.nnunet.sh)"
echo "Done."
