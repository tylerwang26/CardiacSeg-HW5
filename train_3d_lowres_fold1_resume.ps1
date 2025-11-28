
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

Write-Host "Resuming Training for 3D Lowres Fold 1..." -ForegroundColor Cyan

# Use --c to continue training from the latest checkpoint
# Use the CustomEpochs trainer as before
nnUNetv2_train Dataset001_CardiacSeg 3d_lowres 1 -tr nnUNetTrainerCustomEpochs --c --npz

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training 3D Lowres Fold 1 failed."
    exit 1
}

Write-Host "Training 3D Lowres Fold 1 completed." -ForegroundColor Green
