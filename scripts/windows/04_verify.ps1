#Requires -Version 5.1
Write-Host "[Step 4/4] Verify installation" -ForegroundColor Cyan

$VenvPython = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at $VenvPython" -ForegroundColor Red
    exit 1
}

$code = @"
import importlib
mods = [
    ("torch", "__version__"),
    ("monai", "__version__"),
    ("nibabel", "__version__"),
    ("nnunetv2", None),
]
for m, attr in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, attr) if attr else "imported"
        print(f"{m}: {ver}")
    except Exception as e:
        print(f"{m}: FAILED - {e}")

try:
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch backend check failed:", e)
"@

& $VenvPython -c $code

$plan = Get-Command nnUNetv2_plan_and_preprocess -ErrorAction SilentlyContinue
if ($plan) {
    Write-Host "CLI found: $($plan.Source)" -ForegroundColor Green
} else {
    Write-Host "CLI not found in PATH (you can still run via python module calls)." -ForegroundColor Yellow
}

Write-Host "Done." -ForegroundColor Green
