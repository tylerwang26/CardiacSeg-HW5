#Requires -Version 5.1
Write-Host "[Step 3/4] Set nnU-Net environment variables" -ForegroundColor Cyan

$baseDir = $PWD.Path
$env:nnUNet_raw = Join-Path $baseDir 'nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $baseDir 'nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $baseDir 'nnUNet_results'

$env:NNUNET_RAW = $env:nnUNet_raw
$env:NNUNET_PREPROCESSED = $env:nnUNet_preprocessed
$env:NNUNET_RESULTS = $env:nnUNet_results

New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null

# Write helper file to set variables in future sessions
$envFile = Join-Path $baseDir '.env.nnunet.ps1'
@(
    "# Dot-source this file to set nnU-Net variables",
    "$env:nnUNet_raw = '$($env:nnUNet_raw)'",
    "$env:nnUNet_preprocessed = '$($env:nnUNet_preprocessed)'",
    "$env:nnUNet_results = '$($env:nnUNet_results)'",
    "$env:NNUNET_RAW = '$($env:nnUNet_raw)'",
    "$env:NNUNET_PREPROCESSED = '$($env:nnUNet_preprocessed)'",
    "$env:NNUNET_RESULTS = '$($env:nnUNet_results)'"
) | Out-File -Encoding utf8 -FilePath $envFile

Write-Host "nnUNet_raw=$env:nnUNet_raw" -ForegroundColor Green
Write-Host "nnUNet_preprocessed=$env:nnUNet_preprocessed" -ForegroundColor Green
Write-Host "nnUNet_results=$env:nnUNet_results" -ForegroundColor Green
Write-Host "Saved variables to .env.nnunet.ps1 (use: . .\\.env.nnunet.ps1)" -ForegroundColor Cyan
Write-Host "Done." -ForegroundColor Green
