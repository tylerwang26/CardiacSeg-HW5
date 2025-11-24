# 同時監控 3d_lowres 訓練和 2D inference
# 使用方法: .\monitor_both.ps1

$ErrorActionPreference = "SilentlyContinue"

function Show-Status {
    Clear-Host
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "CardiacSeg 即時監控 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    # GPU 狀態
    Write-Host "`n[GPU 狀態]" -ForegroundColor Yellow
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    if ($gpu) {
        $parts = $gpu -split ','
        $util = [int]$parts[0].Trim()
        $memUsed = [int]$parts[1].Trim()
        $memTotal = [int]$parts[2].Trim()
        $temp = [int]$parts[3].Trim()
        $power = [math]::Round([decimal]$parts[4].Trim(), 1)
        
        $utilColor = if ($util -gt 80) { "Green" } elseif ($util -gt 30) { "Yellow" } else { "Red" }
        
        Write-Host "  使用率: " -NoNewline
        Write-Host "$util%" -ForegroundColor $utilColor -NoNewline
        Write-Host " | 記憶體: $memUsed / $memTotal MiB | 溫度: ${temp}°C | 功耗: ${power}W"
    }
    
    # Python 進程
    Write-Host "`n[Python 進程]" -ForegroundColor Yellow
    $processes = Get-Process python* | Where-Object {$_.WorkingSet -gt 50MB} | Select-Object Id, ProcessName, @{Name="CPU(min)";Expression={[math]::Round($_.CPU/60,1)}}, @{Name="Mem(GB)";Expression={[math]::Round($_.WorkingSet/1GB,2)}}, StartTime
    if ($processes) {
        $processes | Format-Table -AutoSize
    } else {
        Write-Host "  沒有大型 Python 進程在執行" -ForegroundColor Red
    }
    
    # 3D lowres 訓練狀態
    Write-Host "`n[3D Lowres 訓練]" -ForegroundColor Yellow
    $log3d = "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_lowres\fold_0\training_log*.txt"
    $latestLog3d = Get-ChildItem $log3d -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($latestLog3d) {
        Write-Host "  日誌: $($latestLog3d.Name) (更新: $($latestLog3d.LastWriteTime.ToString('HH:mm:ss')))"
        
        # 提取最新 epoch 資訊
        $content = Get-Content $latestLog3d.FullName -Tail 50
        $epochLines = $content | Select-String "epoch:" | Select-Object -Last 1
        if ($epochLines) {
            Write-Host "  最新: " -NoNewline
            Write-Host $epochLines.Line.Trim() -ForegroundColor Green
        }
        
        # 檢查最佳 Dice
        $diceLines = $content | Select-String "Yayy! New best EMA pseudo Dice" | Select-Object -Last 1
        if ($diceLines) {
            Write-Host "  " -NoNewline
            Write-Host $diceLines.Line.Trim() -ForegroundColor Cyan
        }
    } else {
        Write-Host "  找不到訓練日誌" -ForegroundColor Red
    }
    
    # 檢查即時輸出日誌
    if (Test-Path "training_3d_output.log") {
        $output3d = Get-Content "training_3d_output.log" -Tail 3 -ErrorAction SilentlyContinue
        if ($output3d) {
            Write-Host "  即時輸出: $($output3d -join ' | ')" -ForegroundColor Gray
        }
    }
    
    # 2D Inference 狀態
    Write-Host "`n[2D Inference]" -ForegroundColor Yellow
    if (Test-Path "inference_2d_validation") {
        $count = (Get-ChildItem "inference_2d_validation" -Filter "*.nii.gz" -ErrorAction SilentlyContinue | Measure-Object).Count
        $total = 10
        $percent = [math]::Round(($count / $total) * 100, 1)
        
        $color = if ($count -eq $total) { "Green" } elseif ($count -gt 0) { "Yellow" } else { "Red" }
        Write-Host "  進度: " -NoNewline
        Write-Host "$count / $total cases ($percent%)" -ForegroundColor $color
        
        # 檢查即時輸出
        if (Test-Path "inference_2d_output.log") {
            $output2d = Get-Content "inference_2d_output.log" -Tail 3 -ErrorAction SilentlyContinue
            if ($output2d) {
                Write-Host "  即時輸出: $($output2d[-1])" -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "  輸出資料夾不存在" -ForegroundColor Red
    }
    
    # 錯誤檢查
    Write-Host "`n[錯誤檢查]" -ForegroundColor Yellow
    $hasError = $false
    
    if (Test-Path "training_3d_error.log") {
        $errors3d = Get-Content "training_3d_error.log" -Tail 5 -ErrorAction SilentlyContinue | Where-Object {$_ -match "error|exception|traceback" -and $_ -notmatch "UserWarning"}
        if ($errors3d) {
            Write-Host "  3D 訓練錯誤: $($errors3d[-1])" -ForegroundColor Red
            $hasError = $true
        }
    }
    
    if (Test-Path "inference_2d_error.log") {
        $errors2d = Get-Content "inference_2d_error.log" -Tail 5 -ErrorAction SilentlyContinue | Where-Object {$_ -match "error|exception|traceback" -and $_ -notmatch "UserWarning"}
        if ($errors2d) {
            Write-Host "  2D Inference 錯誤: $($errors2d[-1])" -ForegroundColor Red
            $hasError = $true
        }
    }
    
    if (-not $hasError) {
        Write-Host "  沒有錯誤" -ForegroundColor Green
    }
    
    Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
    Write-Host "按 Ctrl+C 停止監控 | 每 15 秒自動更新" -ForegroundColor Gray
    Write-Host "=" * 80 -ForegroundColor Cyan
}

# 主循環
Write-Host "開始監控..." -ForegroundColor Green
while ($true) {
    Show-Status
    Start-Sleep -Seconds 15
}
