# Set Environment Variables
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

# Define Paths
$StandardTrainerPath = "C:\CardiacSeg\nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_lowres"
$CustomTrainerPath = "C:\CardiacSeg\nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainerCustomEpochs__nnUNetPlans__3d_lowres"
$InputImages = "C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg\imagesTs"
$OutputFolder = "C:\CardiacSeg\inference_3d_lowres_ensemble_test"

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

Write-Host "=== Preparing for Ensemble Inference ===" -ForegroundColor Cyan

# 1. Copy Fold 0 from Standard to Custom
Write-Host "Copying Fold 0..." -ForegroundColor Yellow
Copy-Item -Path "$StandardTrainerPath\fold_0" -Destination "$CustomTrainerPath" -Recurse -Force

# 2. Copy Fold 1 from Standard to Custom
Write-Host "Copying Fold 1..." -ForegroundColor Yellow
Copy-Item -Path "$StandardTrainerPath\fold_1" -Destination "$CustomTrainerPath" -Recurse -Force

# 3. Ensure Fold 1 has checkpoint_final.pth (using best as final since it didn't finish 1000 epochs)
Write-Host "Fixing Fold 1 Checkpoint..." -ForegroundColor Yellow
$Fold1Best = "$CustomTrainerPath\fold_1\checkpoint_best.pth"
$Fold1Final = "$CustomTrainerPath\fold_1\checkpoint_final.pth"
if (Test-Path $Fold1Best) {
    Copy-Item -Path $Fold1Best -Destination $Fold1Final -Force
    Write-Host "Created checkpoint_final.pth from checkpoint_best.pth for Fold 1" -ForegroundColor Green
} else {
    Write-Error "Fold 1 checkpoint_best.pth not found!"
}

# 4. Run Inference
Write-Host "=== Running Inference on Test Set (Folds 0-4) ===" -ForegroundColor Cyan
# -tr nnUNetTrainerCustomEpochs: Use the custom trainer name (folder name matches)
# -f 0 1 2 3 4: Use all 5 folds
# --save_probabilities: Optional, useful if we want to do fancy ensembling later, but takes space. Omitted for now to save space/time unless requested.
& .venv\Scripts\python.exe -m nnunetv2.inference.predict_from_raw_data -i $InputImages -o $OutputFolder -d Dataset001_CardiacSeg -c 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainerCustomEpochs --disable_tta

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nInference Complete! Results saved to: $OutputFolder" -ForegroundColor Green
} else {
    Write-Error "Inference Failed"
}
