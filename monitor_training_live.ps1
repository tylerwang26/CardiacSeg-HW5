# Training Monitor for CardiacSeg (All Models)
$TotalEpochs = 450 # Set total epochs for estimation

Write-Host "=== CardiacSeg Training Monitor ===" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

function Get-FoldStatus {
    param (
        [string]$BasePath,
        [int]$Fold,
        [string]$Label
    )
    
    $foldPath = Join-Path $BasePath "fold_$Fold"
    if (Test-Path $foldPath) {
        $logFiles = Get-ChildItem "$foldPath\training_log_*.txt" -ErrorAction SilentlyContinue
        if ($logFiles) {
            # 合併所有 log 檔案內容
            $allLines = @()
            $lastUpdate = $null
            foreach ($logFile in $logFiles) {
                $lines = Get-Content $logFile.FullName -ErrorAction SilentlyContinue
                $allLines += $lines
                if (-not $lastUpdate -or $logFile.LastWriteTime -gt $lastUpdate) {
                    $lastUpdate = $logFile.LastWriteTime
                }
            }
            $timeDiff = (Get-Date) - $lastUpdate

            # 解析所有 epoch 行，找最大 epoch
            $epochNumbers = @()
            foreach ($line in $allLines) {
                if ($line -match "Epoch (\d+)") {
                    $epochNumbers += [int]$matches[1]
                } elseif ($line -match "Current epoch: (\d+)") {
                    $epochNumbers += [int]$matches[1]
                }
            }
            $currentEpoch = 0
            if ($epochNumbers.Count -gt 0) {
                $currentEpoch = ($epochNumbers | Measure-Object -Maximum).Maximum
            }

            $statusColor = "Yellow"
            $statusText = "STOPPED"
            if ($timeDiff.TotalMinutes -lt 10) {
                $statusColor = "Green"
                $statusText = "RUNNING"
            }

            Write-Host "  $Label Fold $Fold" -NoNewline -ForegroundColor $statusColor
            Write-Host ": $statusText (Epoch $currentEpoch/$TotalEpochs) - Updated $([math]::Round($timeDiff.TotalMinutes, 1)) min ago"
        } else {
            Write-Host "  $Label Fold ${Fold}: Preparing (no log)" -ForegroundColor Gray
        }
    } else {
        Write-Host "  $Label Fold ${Fold}: Not started" -ForegroundColor DarkGray
    }
}

function Get-AvgDuration {
    param (
        [string]$BasePath
    )
    $totalHours = 0
    $count = 0
    
    for ($i=0; $i -le 4; $i++) {
        $foldPath = Join-Path $BasePath "fold_$i"
        if (Test-Path $foldPath) {
            $logFiles = Get-ChildItem "$foldPath\training_log_*.txt" -ErrorAction SilentlyContinue
            if ($logFiles) {
                $sortedLogs = $logFiles | Sort-Object LastWriteTime
                $firstLog = $sortedLogs | Select-Object -First 1
                $lastLog = $sortedLogs | Select-Object -Last 1
                
                if ($firstLog -and $lastLog) {
                    $duration = $lastLog.LastWriteTime - $firstLog.CreationTime
                    if ($duration.TotalHours -gt 0.1) {
                        $totalHours += $duration.TotalHours
                        $count++
                    }
                }
            }
        }
    }
    
    if ($count -gt 0) {
        $avg = [math]::Round($totalHours / $count, 1)
        return " (Avg: $avg hrs/fold)"
    }
    return ""
}

while ($true) {
    Clear-Host
    $currentTime = Get-Date -Format 'HH:mm:ss'
    Write-Host "=== Training Status [$currentTime] ===" -ForegroundColor Cyan
    
    # GPU Status
    Write-Host "`n[GPU Usage]" -ForegroundColor Yellow
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    

    # Python Processes & Training Status
    Write-Host "`n[Python Processes]" -ForegroundColor Yellow
    $pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 50MB}
    $processCount = $pythonProcesses.Count
    Write-Host "  Running: $processCount processes"

    $trainingFound = $false
    foreach ($proc in $pythonProcesses) {
        $cmdline = $null
        try {
            $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)").CommandLine
        } catch {}
        if ($cmdline -and ($cmdline -match "nnunet" -or $cmdline -match "train_3d_fullres")) {
            Write-Host "    [Active] $($proc.ProcessName) (PID $($proc.Id)) : $cmdline" -ForegroundColor Green
            $trainingFound = $true
        }
    }
    if (-not $trainingFound) {
        Write-Host "    No nnU-Net training process detected." -ForegroundColor Red
    }
    

    # Fold Progress
    $avg2d = Get-AvgDuration -BasePath "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__2d"
    Write-Host "`n[2D Model$avg2d]" -ForegroundColor Yellow
    for ($i=0; $i -le 4; $i++) {
        Get-FoldStatus -BasePath "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__2d" -Fold $i -Label "2D"
    }

    # 3D Lowres
    $pathCustom = "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainerCustomEpochs__nnUNetPlans__3d_lowres"
    $pathStandard = "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_lowres"
    $targetPath = $pathStandard
    if (Test-Path $pathCustom) { $targetPath = $pathCustom }
    $avg3dLow = Get-AvgDuration -BasePath $targetPath
    Write-Host "`n[3D Lowres Model$avg3dLow]" -ForegroundColor Yellow
    for ($i=0; $i -le 4; $i++) {
        Get-FoldStatus -BasePath $targetPath -Fold $i -Label "3D Low"
    }

    # 3D Fullres
    $avg3dFull = Get-AvgDuration -BasePath "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_fullres"
    Write-Host "`n[3D Fullres Model$avg3dFull]" -ForegroundColor Yellow
    for ($i=0; $i -le 4; $i++) {
        Get-FoldStatus -BasePath "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_fullres" -Fold $i -Label "3D Full"
    }

    # Inference & Post-processing Status
    Write-Host "`n[Inference & Post-processing (Target: 50 cases)]" -ForegroundColor Yellow
    $InferenceFolders = @{
        "2D Probabilities"       = "inference_2d_test_prob"
        "3D Lowres (5-Fold)"     = "inference_3d_lowres_5fold_prob"
        "3D Fullres (Fold 0)"    = "inference_3d_fullres_f0_prob"
        "3D Fullres (Fold 1)"    = "inference_3d_fullres_f1_prob"
        "Submission V16"         = "submission_final_v16"
        "Submission V16 Updated" = "submission_final_v16_updated"
    }
    foreach ($name in $InferenceFolders.Keys) {
        $path = $InferenceFolders[$name]
        if (Test-Path $path) {
            # Count .npz for probabilities, .nii.gz for submissions
            if ($path -like "*prob*") {
                $count = (Get-ChildItem $path -Filter "*.npz").Count
            } else {
                $count = (Get-ChildItem $path -Filter "*.nii.gz").Count
            }
            $color = "Yellow"
            if ($count -eq 50) { $color = "Green" }
            Write-Host "  $name : $count / 50" -ForegroundColor $color
        } else {
            Write-Host "  $name : Not Started" -ForegroundColor DarkGray
        }
    }

    Write-Host "`nNext update in 30 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
}
