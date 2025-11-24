# Set Environment Variables
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

# Set Custom Epochs to finish by 11/26 evening
# Calculation: ~68 hours available / 4 folds / ~2.25 min per epoch = ~450 epochs
$env:nnUNet_n_epochs = "450"

# Reduce number of data augmentation workers to prevent "paging file too small" error
$env:nnUNet_n_proc_DA = "2"

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

Write-Host "=== Starting Training 3D Lowres Folds 1-4 (Epochs: $env:nnUNet_n_epochs) ===" -ForegroundColor Cyan

# Train Fold 1
Write-Host "`n[1/4] Skipping Fold 1 (Already at Epoch 851)..." -ForegroundColor Yellow
# & .venv\Scripts\python.exe -m nnunetv2.run.run_training Dataset001_CardiacSeg 3d_lowres 1 -tr nnUNetTrainerCustomEpochs --c
# if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 1 failed"; exit 1 }

# Train Fold 2
Write-Host "`n[2/4] Training Fold 2..." -ForegroundColor Green
& .venv\Scripts\python.exe -m nnunetv2.run.run_training Dataset001_CardiacSeg 3d_lowres 2 -tr nnUNetTrainerCustomEpochs --c
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 2 failed"; exit 1 }

# Train Fold 3
Write-Host "`n[3/4] Training Fold 3..." -ForegroundColor Green
& .venv\Scripts\python.exe -m nnunetv2.run.run_training Dataset001_CardiacSeg 3d_lowres 3 -tr nnUNetTrainerCustomEpochs --c
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 3 failed"; exit 1 }

# Train Fold 4
Write-Host "`n[4/4] Training Fold 4..." -ForegroundColor Green
& .venv\Scripts\python.exe -m nnunetv2.run.run_training Dataset001_CardiacSeg 3d_lowres 4 -tr nnUNetTrainerCustomEpochs --c
if ($LASTEXITCODE -ne 0) { Write-Error "Training Fold 4 failed"; exit 1 }

Write-Host "`nAll 3D Lowres folds trained successfully!" -ForegroundColor Green
