# Set Environment Variables
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

# Define Paths
$InputImages = "C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg\imagesTs"
$OutputFolder = "C:\CardiacSeg\inference_3d_lowres_ensemble_test_prob"

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

Write-Host "=== Running Inference on Test Set (Folds 0-4) with Probabilities ===" -ForegroundColor Cyan
# -tr nnUNetTrainerCustomEpochs: Use the custom trainer name (folder name matches)
# -f 0 1 2 3 4: Use all 5 folds
# --save_probabilities: REQUIRED for V8 ensemble
& .venv\Scripts\python.exe -m nnunetv2.inference.predict_from_raw_data -i $InputImages -o $OutputFolder -d Dataset001_CardiacSeg -c 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainerCustomEpochs --disable_tta --save_probabilities

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nInference Complete! Results saved to: $OutputFolder" -ForegroundColor Green
} else {
    Write-Error "Inference Failed"
}
