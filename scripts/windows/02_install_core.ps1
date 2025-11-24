#Requires -Version 5.1
Write-Host "[Step 2/4] Install core Python packages into .venv" -ForegroundColor Cyan

$VenvPython = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at $VenvPython. Run scripts\windows\01_create_venv.ps1 first." -ForegroundColor Red
    exit 1
}

# Upgrade packaging tools
& $VenvPython -m pip install --upgrade pip setuptools wheel

# Detect NVIDIA GPU for CUDA wheels
$gpuVendor = 'None'
try {
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    if ($gpus) {
        if ($gpus | Where-Object { $_.AdapterCompatibility -match 'NVIDIA' -or $_.Name -match 'NVIDIA' }) { $gpuVendor = 'NVIDIA' }
        elseif ($gpus | Where-Object { $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon' }) { $gpuVendor = 'AMD' }
    }
} catch {}

if ($gpuVendor -eq 'NVIDIA') {
    Write-Host "Installing CUDA-enabled PyTorch (cu121) ..." -ForegroundColor Green
    & $VenvPython -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Installing CPU PyTorch ..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade torch torchvision
}

Write-Host "Installing nnUNet and supporting libraries ..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade nnunetv2 monai nibabel numpy scipy SimpleITK scikit-image matplotlib tqdm blosc2

Write-Host "Done." -ForegroundColor Green
