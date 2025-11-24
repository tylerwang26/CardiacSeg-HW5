# Training Monitor for 3D Lowres
$TotalEpochs = 450 # Set total epochs for estimation

Write-Host "=== 3D Lowres Training Monitor ===" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

while ($true) {
    Clear-Host
    $currentTime = Get-Date -Format 'HH:mm:ss'
    Write-Host "=== Training Status [$currentTime] ===" -ForegroundColor Cyan
    
    # GPU Status
    Write-Host "`n[GPU Usage]" -ForegroundColor Yellow
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    
    # Python Processes
    Write-Host "`n[Python Processes]" -ForegroundColor Yellow
    $pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 50MB}
    $processCount = $pythonProcesses.Count
    Write-Host "  Running: $processCount processes"
    if ($processCount -gt 0) {
        $mainProcess = $pythonProcesses | Sort-Object WorkingSet -Descending | Select-Object -First 1
        $memMB = [math]::Round($mainProcess.WorkingSet/1MB,0)
        Write-Host "  Main process memory: $memMB MB"
    }
    
    # Fold Training Progress
    Write-Host "`n[Fold Progress]" -ForegroundColor Yellow
    for ($i=0; $i -le 4; $i++) {
        # Updated to point to the Custom Trainer directory
        $foldPath = "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainerCustomEpochs__nnUNetPlans__3d_lowres\fold_$i"
        if (Test-Path $foldPath) {
            $logFiles = Get-ChildItem "$foldPath\training_log_*.txt" -ErrorAction SilentlyContinue
            $logFile = $logFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($logFile) {
                $lastUpdate = $logFile.LastWriteTime
                $timeDiff = (Get-Date) - $lastUpdate
                
                # Read recent lines for parsing
                $recentLines = Get-Content $logFile.FullName -Tail 100 -ErrorAction SilentlyContinue
                
                # Parse Epoch and Timestamp
                $epochLines = $recentLines | Where-Object {$_ -match "Epoch \d+"}
                $currentEpoch = 0
                $epochTimestamp = ""
                if ($epochLines) {
                    $lastEpochLine = $epochLines | Select-Object -Last 1
                    if ($lastEpochLine -match "Epoch (\d+)") {
                        $currentEpoch = [int]$matches[1]
                    }
                    if ($lastEpochLine -match "^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+):") {
                        $epochTimestamp = $matches[1]
                    }
                }

                # Parse Epoch Time
                $timeLines = $recentLines | Where-Object {$_ -match "Epoch time: ([\d\.]+) s"}
                $epochDurationMin = 0
                if ($timeLines) {
                    $lastTimeLine = $timeLines | Select-Object -Last 1
                    if ($lastTimeLine -match "Epoch time: ([\d\.]+) s") {
                        $epochDurationSec = [double]$matches[1]
                        $epochDurationMin = [math]::Round($epochDurationSec / 60, 2)
                    }
                }

                # Calculate Estimates
                $estimatedRemainingMin = 0
                $endTimeStr = "Unknown"
                if ($epochDurationMin -gt 0) {
                    $remainingEpochs = $TotalEpochs - $currentEpoch
                    $estimatedRemainingMin = [math]::Round($remainingEpochs * $epochDurationMin, 0)
                    $endTime = (Get-Date).AddMinutes($estimatedRemainingMin)
                    $endTimeStr = $endTime.ToString("yyyy-MM-dd HH:mm:ss")
                }
                
                if ($timeDiff.TotalMinutes -lt 5) {
                    Write-Host "  Fold $i" -NoNewline -ForegroundColor Green
                    Write-Host ": RUNNING " -NoNewline
                } else {
                    Write-Host "  Fold $i" -NoNewline -ForegroundColor Yellow
                    Write-Host ": STOPPED " -NoNewline
                }
                
                if ($epochTimestamp) {
                    Write-Host "- ${epochTimestamp}: ${epochDurationMin} min/ estimate last ${estimatedRemainingMin} min/ End at $endTimeStr"
                } else {
                    Write-Host " (updated $([math]::Round($timeDiff.TotalMinutes, 1)) min ago)"
                }
            } else {
                Write-Host "  Fold ${i}: Preparing (no log yet)" -ForegroundColor Gray
            }
        } else {
            Write-Host "  Fold ${i}: Not started" -ForegroundColor DarkGray
        }
    }
    
    # Latest Training Log
    Write-Host "`n[Latest Log]" -ForegroundColor Yellow
    # Updated to point to the Custom Trainer directory
    $allLogs = Get-ChildItem "nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainerCustomEpochs__nnUNetPlans__3d_lowres\fold_*\training_log_*.txt" -ErrorAction SilentlyContinue
    $latestLog = $allLogs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        $recentLines = Get-Content $latestLog.FullName -Tail 100 -ErrorAction SilentlyContinue
        
        # Parse Epoch
        $epochLines = $recentLines | Where-Object {$_ -match "Epoch \d+"}
        $currentEpoch = 0
        if ($epochLines) {
            $lastEpochLine = $epochLines | Select-Object -Last 1
            if ($lastEpochLine -match "Epoch (\d+)") {
                $currentEpoch = [int]$matches[1]
            }
        }

        # Parse Learning Rates
        $lrLines = $recentLines | Where-Object {$_ -match "Current learning rate: ([\d\.]+)"}
        $currentLr = "N/A"
        $prevLr = "N/A"
        if ($lrLines) {
            $lastLrLine = $lrLines | Select-Object -Last 1
            if ($lastLrLine -match "Current learning rate: ([\d\.]+)") {
                $currentLr = $matches[1]
            }
            if ($lrLines.Count -ge 2) {
                $prevLrLine = $lrLines | Select-Object -Last 2 | Select-Object -First 1
                if ($prevLrLine -match "Current learning rate: ([\d\.]+)") {
                    $prevLr = $matches[1]
                }
            }
        }
        
        Write-Host "  Epoch $currentEpoch/$TotalEpochs, Last/Current leaning rate: $prevLr/$currentLr" -ForegroundColor Gray
    }
    
    Write-Host "`nNext update in 30 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
}
