
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

$folds = 1..4
foreach ($fold in $folds) {
    $checkpoint = "C:\CardiacSeg\nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__2d\fold_$fold\checkpoint_best.pth"
    if (Test-Path $checkpoint) {
        Write-Host "Fold $fold checkpoint already exists. Skipping."
    } else {
        Write-Host "Training 2D Fold $fold..."
        # Use --npz to save softmax predictions for validation (useful for ensemble)
        nnUNetv2_train Dataset001_CardiacSeg 2d $fold --npz
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training Fold $fold failed."
            exit 1
        }
    }
}

Write-Host "All 2D folds (1-4) training process completed."
