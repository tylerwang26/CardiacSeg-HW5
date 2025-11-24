#!/bin/bash
export PATH="$(pwd)/.venv/Scripts:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

echo "Environment activated!"
which python
echo "nnUNet_results: $nnUNet_results"