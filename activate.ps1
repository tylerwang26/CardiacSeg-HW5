$env:PATH = "$PSScriptRoot\.venv\Scripts;$env:PATH"
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
$env:nnUNet_raw = "$PSScriptRoot\nnUNet_raw"
$env:nnUNet_preprocessed = "$PSScriptRoot\nnUNet_preprocessed"
$env:nnUNet_results = "$PSScriptRoot\nnUNet_results"

Write-Host "Environment activated!" -ForegroundColor Green
Write-Host "Python path: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Yellow
Write-Host "nnUNet_results: $env:nnUNet_results" -ForegroundColor Yellow
