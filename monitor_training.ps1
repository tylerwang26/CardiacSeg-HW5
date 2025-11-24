# 監控訓練進度腳本
# 每 30 秒更新一次 GPU 和訓練狀態

param(
    [int]$IntervalSeconds = 30
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  CardiacSeg 訓練監控" -ForegroundColor Cyan
Write-Host "  更新間隔: $IntervalSeconds 秒" -ForegroundColor Cyan
Write-Host "  按 Ctrl+C 停止監控" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    Clear-Host
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Host "=== 訓練狀態監控 [$timestamp] ===" -ForegroundColor Green
    Write-Host ""
    
    # 1. GPU 狀態
    Write-Host "1. GPU 狀態：" -ForegroundColor Yellow
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
    Write-Host ""
    
    # 2. Python 訓練進程
    Write-Host "2. Python 訓練進程：" -ForegroundColor Yellow
    $pythonProcs = Get-Process python* -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 50MB}
    if ($pythonProcs) {
        $pythonProcs | Select-Object Id, @{Name="CPU(min)";Expression={[math]::Round($_.CPU/60,1)}}, @{Name="Mem(MB)";Expression={[math]::Round($_.WorkingSet/1MB,0)}}, StartTime | Format-Table -AutoSize
    } else {
        Write-Host "  沒有大型 Python 進程運行" -ForegroundColor Gray
    }
    Write-Host ""
    
    # 3. 3D lowres 訓練日誌
    Write-Host "3. 3D lowres 訓練進度：" -ForegroundColor Yellow
    $log3d = Get-ChildItem "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_lowres\fold_0\training_log*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log3d) {
        Write-Host "  日誌: $($log3d.Name)" -ForegroundColor Cyan
        Write-Host "  更新: $($log3d.LastWriteTime)" -ForegroundColor Cyan
        $lastLines = Get-Content $log3d.FullName -Tail 3 -ErrorAction SilentlyContinue
        if ($lastLines) {
            Write-Host "  最新內容:" -ForegroundColor Cyan
            $lastLines | ForEach-Object { Write-Host "    $_" -ForegroundColor White }
        }
    } else {
        Write-Host "  找不到訓練日誌" -ForegroundColor Red
    }
    Write-Host ""
    
    # 4. 2D Inference 進度
    Write-Host "4. 2D Inference 進度：" -ForegroundColor Yellow
    if (Test-Path "inference_2d_validation") {
        $count = (Get-ChildItem "inference_2d_validation" -Filter "*.nii.gz" -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "  已完成: $count / 50 cases" -ForegroundColor Cyan
        if ($count -gt 0) {
            $latest = Get-ChildItem "inference_2d_validation" -Filter "*.nii.gz" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            Write-Host "  最新: $($latest.Name) [$($latest.LastWriteTime)]" -ForegroundColor Cyan
        }
    } else {
        Write-Host "  資料夾不存在" -ForegroundColor Gray
    }
    Write-Host ""
    
    Write-Host "=== 下次更新: $IntervalSeconds 秒後 ===" -ForegroundColor Gray
    Start-Sleep -Seconds $IntervalSeconds
}
