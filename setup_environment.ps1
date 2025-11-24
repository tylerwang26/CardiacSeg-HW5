# CardiacSeg Environment Setup Script (Cross-Platform)
# This script installs Python packages and sets up nnU-Net environment variables
# Supports: Windows (PowerShell), macOS, Linux

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CardiacSeg Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Detect operating system (do not overwrite built-in readonly flags like $IsMacOS/$IsWindows)
$OnWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
$OnMacOS   = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)
$OnLinux   = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)

if ($OnMacOS) {
    Write-Host "Detected: macOS" -ForegroundColor Green
} elseif ($OnLinux) {
    Write-Host "Detected: Linux" -ForegroundColor Green
} elseif ($OnWindows) {
    Write-Host "Detected: Windows" -ForegroundColor Green
} else {
    Write-Host "WARNING: Unknown OS, assuming Windows" -ForegroundColor Yellow
    $OnWindows = $true
}
Write-Host ""

# 1. Ensure Python is installed and resolve command
Write-Host "[1/5] Checking/Installing Python..." -ForegroundColor Yellow

function Get-PythonExe {
    param([switch]$AllowWindowsApps)
    $paths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python'),
        'C:\Program Files',
        'C:\Program Files (x86)'
    )
    $candidates = @()
    foreach ($p in $paths) {
        $candidates += Get-ChildItem -Path $p -Recurse -File -Filter python.exe -ErrorAction SilentlyContinue
    }
    # Filter out venv and WindowsApps stubs
    $filtered = $candidates | Where-Object { $_.FullName -notmatch '\\Lib\\venv\\' -and $_.FullName -notmatch 'WindowsApps' }
    if ($filtered.Count -gt 0) {
        return ($filtered | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    }
    if ($AllowWindowsApps) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -and ($cmd.Source -like '*WindowsApps*')) { return $cmd.Source }
    }
    return $null
}

function Test-ValidPython($exe) {
    if (-not $exe) { return $false }
    try {
        $out = & $exe --version 2>&1
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($out -match 'Python was not found') { return $false }
        return $true
    } catch { return $false }
}

if ($OnWindows) {
    # Windows-specific Python detection
    $PythonExe = Get-PythonExe
    if (-not (Test-ValidPython $PythonExe)) {
        Write-Host "Valid Python not found. Attempting install via winget (Python 3.11)..." -ForegroundColor Yellow
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            try {
                winget install -e --id Python.Python.3.11 --source winget --accept-package-agreements --accept-source-agreements
            } catch {
                Write-Host "winget install failed: $($_.Exception.Message)" -ForegroundColor Red
            }
            $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
            $PythonExe = Get-PythonExe
        }
    }

    # As last resort allow WindowsApps but still validate
    if (-not (Test-ValidPython $PythonExe)) {
        $PythonExe = Get-PythonExe -AllowWindowsApps
    }

    if (-not (Test-ValidPython $PythonExe)) {
        Write-Host "ERROR: Real Python interpreter not available. Please install from python.org and re-run." -ForegroundColor Red
        Write-Host "Download: https://www.python.org/downloads/ (ensure 'Add Python to PATH' is checked)" -ForegroundColor Yellow
        exit 1
    }
} else {
    # macOS/Linux: Use python3
    $PythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
    if (-not $PythonExe) {
        Write-Host "ERROR: python3 not found. Please install Python 3.8+ from:" -ForegroundColor Red
        Write-Host "  https://www.python.org/downloads/" -ForegroundColor Yellow
        if ($OnMacOS) {
            Write-Host "  or use: brew install python@3.11" -ForegroundColor Yellow
        }
        exit 1
    }
}

$pythonVersion = & $PythonExe --version 2>&1
Write-Host "OK Using Python: $PythonExe ($pythonVersion)" -ForegroundColor Green

# 1b. Create and switch to a project-specific virtual environment (.venv)
Write-Host "[1b/5] Creating/Using virtual environment (.venv)..." -ForegroundColor Yellow
$venvDir = Join-Path $PWD.Path ".venv"
if (-not (Test-Path $venvDir)) {
    try {
        & $PythonExe -m venv $venvDir
    } catch {
        Write-Host "ERROR: Failed to create virtual environment at $venvDir: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

if ($OnWindows) {
    $VenvPython = Join-Path $venvDir "Scripts\python.exe"
} else {
    $VenvPython = Join-Path $venvDir "bin/python"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: Virtual environment python not found at $VenvPython" -ForegroundColor Red
    exit 1
}

# Use venv python going forward
$PythonExe = $VenvPython
$pythonVersion = & $PythonExe --version 2>&1
Write-Host "OK Using venv Python: $PythonExe ($pythonVersion)" -ForegroundColor Green

# Windows-only: Ensure Python directory is on PATH
if ($OnWindows) {
    function Add-ToPathIfMissing {
        param(
            [string]$Dir,
            [ValidateSet('User','Machine')]$Scope='User'
        )
        if (-not (Test-Path $Dir)) { return }
        $current = [System.Environment]::GetEnvironmentVariable('Path',$Scope)
        $normalized = ($current -split ';') | Where-Object { $_ -and $_.Trim() -ne '' } | ForEach-Object { $_.TrimEnd('\') }
        $dirNorm = $Dir.TrimEnd('\\')
        if ($normalized -notcontains $dirNorm) {
            $new = ($normalized + $dirNorm) -join ';'
            [System.Environment]::SetEnvironmentVariable('Path',$new,$Scope)
        }
    }

    $pythonDir = Split-Path $PythonExe -Parent
    $scriptsDir = Join-Path $pythonDir "Scripts"
    if ($pythonDir -like '*WindowsApps*') {
        Write-Host "WARNING: Using WindowsApps alias. Real installation recommended for package management." -ForegroundColor Yellow
    }
    Add-ToPathIfMissing -Dir $pythonDir -Scope User
    Add-ToPathIfMissing -Dir $scriptsDir -Scope User
    # Update current session PATH immediately
    if ($env:Path -notlike "*$pythonDir*") { $env:Path = "$env:Path;$pythonDir" }
    if ($env:Path -notlike "*$scriptsDir*") { $env:Path = "$env:Path;$scriptsDir" }
    Write-Host "PATH updated with Python directory: $pythonDir" -ForegroundColor Green
    Write-Host "PATH updated with Scripts directory: $scriptsDir" -ForegroundColor Green
}

# 2. Upgrade pip
Write-Host ""
Write-Host "[2/5] Upgrading pip..." -ForegroundColor Yellow
& $PythonExe -m pip install --upgrade pip

# 3. Install required Python packages (GPU-aware)
Write-Host ""
Write-Host "[3/5] Installing Python packages (this may take a few minutes)..." -ForegroundColor Yellow

# Detect GPU/acceleration based on OS
if ($OnMacOS) {
    # Check for Apple Silicon via .NET runtime info
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($arch -eq 'Arm64') {
        Write-Host "Detected Apple Silicon (M1/M2/M3/M4). Installing MPS-enabled PyTorch." -ForegroundColor Green
    } else {
        Write-Host "Detected Intel Mac. Installing CPU PyTorch." -ForegroundColor Yellow
    }
    try {
        & $PythonExe -m pip install --upgrade torch torchvision
        if ($LASTEXITCODE -eq 0) { Write-Host "    OK torch/torchvision installed" -ForegroundColor Green } else { Write-Host "    ERROR torch/torchvision install failed (exit $LASTEXITCODE)" -ForegroundColor Red }
    } catch { Write-Host "    ERROR torch/torchvision install failed: $($_.Exception.Message)" -ForegroundColor Red }
} elseif ($OnLinux) {
    Write-Host "Detected Linux. Installing PyTorch (will auto-detect CUDA if available)." -ForegroundColor Yellow
    try {
        & $PythonExe -m pip install --upgrade torch torchvision
        if ($LASTEXITCODE -eq 0) { Write-Host "    OK torch/torchvision installed" -ForegroundColor Green } else { Write-Host "    ERROR torch/torchvision install failed (exit $LASTEXITCODE)" -ForegroundColor Red }
    } catch { Write-Host "    ERROR torch/torchvision install failed: $($_.Exception.Message)" -ForegroundColor Red }
} else {
    # Windows: Detect GPU vendor
    $gpuVendor = 'None'
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        if ($gpus) {
            if ($gpus | Where-Object { $_.AdapterCompatibility -match 'NVIDIA' -or $_.Name -match 'NVIDIA' }) { $gpuVendor = 'NVIDIA' }
            elseif ($gpus | Where-Object { $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon' }) { $gpuVendor = 'AMD' }
        }
    } catch {}

    if ($gpuVendor -eq 'NVIDIA') {
        Write-Host "Detected NVIDIA GPU. Installing CUDA-enabled PyTorch (cu121)." -ForegroundColor Green
        try {
            & $PythonExe -m pip uninstall -y torch torchvision 2>$null | Out-Null
            & $PythonExe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
            if ($LASTEXITCODE -eq 0) { Write-Host "    OK torch/torchvision (CUDA) installed" -ForegroundColor Green } else { Write-Host "    ERROR torch/torchvision (CUDA) install failed (exit $LASTEXITCODE)" -ForegroundColor Red }
        } catch { Write-Host "    ERROR torch/torchvision (CUDA) install failed: $($_.Exception.Message)" -ForegroundColor Red }
    } elseif ($gpuVendor -eq 'AMD') {
        Write-Host "Detected AMD GPU on Windows. ROCm is not supported on Windows for PyTorch/nnU-Net. Using CPU backend." -ForegroundColor Yellow
        $env:CUDA_VISIBLE_DEVICES = "-1"
        Write-Host "    Forced CPU mode (CUDA_VISIBLE_DEVICES=-1)" -ForegroundColor Yellow
        try {
            & $PythonExe -m pip install --upgrade torch torchvision
            if ($LASTEXITCODE -eq 0) { Write-Host "    OK torch/torchvision (CPU) installed" -ForegroundColor Green } else { Write-Host "    ERROR torch/torchvision (CPU) install failed (exit $LASTEXITCODE)" -ForegroundColor Red }
        } catch { Write-Host "    ERROR torch/torchvision (CPU) install failed: $($_.Exception.Message)" -ForegroundColor Red }
    } else {
        Write-Host "No supported GPU detected. Installing CPU PyTorch." -ForegroundColor Yellow
        $env:CUDA_VISIBLE_DEVICES = "-1"
        Write-Host "    Forced CPU mode (CUDA_VISIBLE_DEVICES=-1)" -ForegroundColor Yellow
        try {
            & $PythonExe -m pip install --upgrade torch torchvision
            if ($LASTEXITCODE -eq 0) { Write-Host "    OK torch/torchvision (CPU) installed" -ForegroundColor Green } else { Write-Host "    ERROR torch/torchvision (CPU) install failed (exit $LASTEXITCODE)" -ForegroundColor Red }
        } catch { Write-Host "    ERROR torch/torchvision (CPU) install failed: $($_.Exception.Message)" -ForegroundColor Red }
    }
}

$packages = @(
    "nnunetv2",
    "monai",
    "nibabel",
    "numpy",
    "scipy",
    "SimpleITK",
    "scikit-image",
    "matplotlib",
    "tqdm",
    "blosc2"
)

foreach ($package in $packages) {
    Write-Host "  Installing $package..." -ForegroundColor Cyan
    try {
        & $PythonExe -m pip install $package
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    OK $package installed successfully" -ForegroundColor Green
        } else {
            Write-Host "    ERROR $package installation failed (exit $LASTEXITCODE)" -ForegroundColor Red
        }
    } catch {
        Write-Host "    ERROR $package installation failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# 4. Set nnU-Net environment variables and ensure directories
Write-Host ""
Write-Host "[4/5] Setting nnU-Net environment variables..." -ForegroundColor Yellow

# Use current directory as base (cross-platform)
if ($OnWindows) {
    $baseDir = $PWD.Path
    $env:nnUNet_raw = "$baseDir\nnUNet_raw"
    $env:nnUNet_preprocessed = "$baseDir\nnUNet_preprocessed"
    $env:nnUNet_results = "$baseDir\nnUNet_results"
} else {
    $baseDir = $PWD.Path
    $env:nnUNet_raw = "$baseDir/nnUNet_raw"
    $env:nnUNet_preprocessed = "$baseDir/nnUNet_preprocessed"
    $env:nnUNet_results = "$baseDir/nnUNet_results"
}

$env:NNUNET_RAW = $env:nnUNet_raw
$env:NNUNET_PREPROCESSED = $env:nnUNet_preprocessed
$env:NNUNET_RESULTS = $env:nnUNet_results

Write-Host "OK Environment variables set:" -ForegroundColor Green
Write-Host "  nnUNet_raw = $env:nnUNet_raw" -ForegroundColor Gray
Write-Host "  nnUNet_preprocessed = $env:nnUNet_preprocessed" -ForegroundColor Gray
Write-Host "  nnUNet_results = $env:nnUNet_results" -ForegroundColor Gray

# Ensure directories exist
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null
Write-Host "OK Created/verified nnU-Net directories." -ForegroundColor Green

# 5. Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$testPackages = @("torch", "nnunetv2", "monai", "nibabel")
foreach ($pkg in $testPackages) {
    Write-Host "Checking $pkg..." -ForegroundColor Yellow -NoNewline
    $code = "import importlib; m=importlib.import_module('$pkg'); print(getattr(m,'__version__','OK'))"
    $result = & $PythonExe -c $code 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK Version: $result" -ForegroundColor Green
    } else {
        Write-Host " ERROR Not installed" -ForegroundColor Red
    }
}

# Report backend status
Write-Host "" 
Write-Host "Backend status:" -ForegroundColor Cyan
$backendCode = @"
import torch
print('torch version:', torch.__version__)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon GPU) available: True')
    print('Using Apple Silicon GPU acceleration')
elif torch.cuda.is_available():
    print('CUDA available:', torch.cuda.is_available())
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Using CPU backend')
"@
$backendOut = & $PythonExe -c $backendCode 2>&1
Write-Host $backendOut

# 6. Complete
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "1. A Python virtual environment has been created at .venv" -ForegroundColor White
if ($OnWindows) {
    Write-Host "   Activate with: .\\.venv\\Scripts\\Activate.ps1 (PowerShell)" -ForegroundColor Gray
} else {
    Write-Host "   Activate with: . .venv/bin/Activate.ps1 (PowerShell) or source .venv/bin/activate (bash/zsh)" -ForegroundColor Gray
}
Write-Host "2. Environment variables are only valid in current shell session" -ForegroundColor White
Write-Host "3. Run this script again in new terminal windows" -ForegroundColor White

if ($OnWindows) {
    Write-Host "3. For permanent setup, add variables to System Environment Variables" -ForegroundColor White
} else {
    Write-Host "4. For permanent setup, add these lines to your ~/.bashrc or ~/.zshrc:" -ForegroundColor White
    Write-Host "   export nnUNet_raw=`"$env:nnUNet_raw`"" -ForegroundColor Gray
    Write-Host "   export nnUNet_preprocessed=`"$env:nnUNet_preprocessed`"" -ForegroundColor Gray
    Write-Host "   export nnUNet_results=`"$env:nnUNet_results`"" -ForegroundColor Gray
}

Write-Host ""
Write-Host "You can now run:" -ForegroundColor Cyan
if ($OnWindows) {
    Write-Host "  python rename_dataset.py    (Rename dataset files)" -ForegroundColor White
    Write-Host "  python nnunet_train.py      (Train model)" -ForegroundColor White
    Write-Host "  python nnunet_infer.py      (Inference)" -ForegroundColor White
    Write-Host "  python nnunet_evaluate.py   (Evaluation)" -ForegroundColor White
    Write-Host "If a new terminal doesn't recognize 'python', sign out and back in (PATH refresh)." -ForegroundColor Yellow
} else {
    Write-Host "  python3 rename_dataset.py    (Rename dataset files)" -ForegroundColor White
    Write-Host "  python3 nnunet_train.py      (Train model)" -ForegroundColor White
    Write-Host "  python3 nnunet_infer.py      (Inference)" -ForegroundColor White
    Write-Host "  python3 nnunet_evaluate.py   (Evaluation)" -ForegroundColor White
}
Write-Host ""
