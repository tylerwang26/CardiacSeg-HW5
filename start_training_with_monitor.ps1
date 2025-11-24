# 啟動 3d_lowres 訓練並監控
# 使用方法: .\start_training_with_monitor.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  啟動 3d_lowres 訓練" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 設定環境變數
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

Write-Host "`n環境變數已設定：" -ForegroundColor Green
Write-Host "  nnUNet_raw: $env:nnUNet_raw"
Write-Host "  nnUNet_preprocessed: $env:nnUNet_preprocessed"
Write-Host "  nnUNet_results: $env:nnUNet_results"

# 啟動訓練（背景執行）
Write-Host "`n正在啟動訓練..." -ForegroundColor Yellow

$logFile = "training_3d_lowres_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# 使用 Start-Process 在新視窗啟動訓練
$trainProcess = Start-Process -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "continue_training.py --config 3d_lowres --fold 0 --device cuda" `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "${logFile}.err" `
    -PassThru `
    -NoNewWindow

Write-Host "✓ 訓練已啟動 (PID: $($trainProcess.Id))" -ForegroundColor Green
Write-Host "  日誌: $logFile" -ForegroundColor Cyan
Write-Host "  錯誤日誌: ${logFile}.err" -ForegroundColor Cyan

# 等待一下讓訓練初始化
Write-Host "`n等待 15 秒讓訓練初始化..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# 檢查進程是否還在運行
if (Get-Process -Id $trainProcess.Id -ErrorAction SilentlyContinue) {
    Write-Host "✓ 訓練進程正在運行" -ForegroundColor Green
} else {
    Write-Host "✗ 訓練進程已停止，請檢查日誌" -ForegroundColor Red
    if (Test-Path $logFile) {
        Write-Host "`n=== 日誌內容 ===" -ForegroundColor Yellow
        Get-Content $logFile
    }
    exit 1
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "  訓練已在背景執行" -ForegroundColor Cyan
Write-Host "  使用以下命令監控進度：" -ForegroundColor Cyan
Write-Host "  .\monitor_training.ps1" -ForegroundColor White
Write-Host "======================================" -ForegroundColor Cyan
