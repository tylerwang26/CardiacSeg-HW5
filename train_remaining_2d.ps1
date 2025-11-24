# Set Environment Variables
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

Write-Host "=== Starting Training 2D Folds 1-4 ===" -ForegroundColor Cyan

# Train Fold 1
Write-Host "`n[1/4] Training Fold 1..." -ForegroundColor Green
nnUNetv2_train Dataset001_CardiacSeg 2d 1 --npz
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 1 failed"; exit 1 }

# Train Fold 2
Write-Host "`n[2/4] Training Fold 2..." -ForegroundColor Green
nnUNetv2_train Dataset001_CardiacSeg 2d 2 --npz
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 2 failed"; exit 1 }

# Train Fold 3
Write-Host "`n[3/4] Training Fold 3..." -ForegroundColor Green
nnUNetv2_train Dataset001_CardiacSeg 2d 3 --npz
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 3 failed"; exit 1 }

# Train Fold 4
Write-Host "`n[4/4] Training Fold 4..." -ForegroundColor Green
nnUNetv2_train Dataset001_CardiacSeg 2d 4 --npz
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 4 failed"; exit 1 }

Write-Host "`nAll 2D folds trained successfully!" -ForegroundColor Green
