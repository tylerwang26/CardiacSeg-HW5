#Requires -Version 5.1
Write-Host "[Step 1/4] Create project virtual environment (.venv)" -ForegroundColor Cyan

function Get-Python {
    $candidates = @('py','python3','python')
    foreach ($c in $candidates) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

$py = Get-Python
if (-not $py) {
    Write-Host "ERROR: Python 3 not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

$venvDir = Join-Path $PWD ".venv"
if (-not (Test-Path $venvDir)) {
    & $py -m venv $venvDir
    Write-Host "Created venv at $venvDir" -ForegroundColor Green
} else {
    Write-Host "venv already exists at $venvDir" -ForegroundColor Yellow
}

$VenvPython = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at $VenvPython" -ForegroundColor Red
    exit 1
}

& $VenvPython --version
Write-Host "Done." -ForegroundColor Green
