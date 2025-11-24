# 簡化版 2D Inference - 只處理 validation cases (10 個)
# 避免記憶體問題

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  2D Inference (Validation Set)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 設定環境變數
$env:nnUNet_raw = "C:\CardiacSeg\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\CardiacSeg\nnUNet_preprocessed"
$env:nnUNet_results = "C:\CardiacSeg\nnUNet_results"

# Validation cases for fold_0
$valCases = @(
    "patient0009",
    "patient0013", 
    "patient0022",
    "patient0023",
    "patient0031",
    "patient0033",
    "patient0034",
    "patient0039",
    "patient0043",
    "patient0046"
)

Write-Host "`n將處理 $($valCases.Count) 個 validation cases" -ForegroundColor Green
Write-Host ""

# 創建輸出資料夾
$outputDir = "inference_2d_validation"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# 創建臨時輸入資料夾（只包含 validation cases）
$tempInput = "temp_val_input"
if (Test-Path $tempInput) {
    Remove-Item $tempInput -Recurse -Force
}
New-Item -ItemType Directory -Path $tempInput | Out-Null

Write-Host "複製 validation cases 到臨時資料夾..." -ForegroundColor Yellow
$copied = 0
foreach ($case in $valCases) {
    $srcFile = "nnUNet_raw\Dataset001_CardiacSeg\imagesTr\${case}_0000.nii.gz"
    if (Test-Path $srcFile) {
        Copy-Item $srcFile -Destination $tempInput
        $copied++
        Write-Host "  ✓ $case" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $case (找不到)" -ForegroundColor Red
    }
}

Write-Host "`n已複製 $copied 個檔案" -ForegroundColor Cyan

if ($copied -eq 0) {
    Write-Host "`n錯誤: 沒有找到任何輸入檔案" -ForegroundColor Red
    exit 1
}

# 執行 inference (減少 worker 避免記憶體問題)
Write-Host "`n開始 2D inference..." -ForegroundColor Yellow
Write-Host "  模型: 2D (checkpoint_best.pth)"
Write-Host "  輸入: $copied cases"
Write-Host "  輸出: $outputDir"
Write-Host ""

.venv\Scripts\nnUNetv2_predict.exe `
    -i $tempInput `
    -o $outputDir `
    -d Dataset001_CardiacSeg `
    -c 2d `
    -f 0 `
    -chk checkpoint_best.pth `
    --disable_tta `
    --num_processes_preprocessing 1 `
    --num_processes_segmentation_export 1

$exitCode = $LASTEXITCODE

# 清理臨時資料夾
Write-Host "`n清理臨時資料夾..." -ForegroundColor Yellow
Remove-Item $tempInput -Recurse -Force

# 檢查結果
if ($exitCode -eq 0) {
    Write-Host "`n======================================" -ForegroundColor Green
    Write-Host "  ✓ 2D Inference 完成！" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    
    $resultCount = (Get-ChildItem $outputDir -Filter "*.nii.gz" | Measure-Object).Count
    Write-Host "`n產生 $resultCount 個預測檔案" -ForegroundColor Cyan
    
    if ($resultCount -gt 0) {
        Write-Host "`n預測檔案列表:" -ForegroundColor Yellow
        Get-ChildItem $outputDir -Filter "*.nii.gz" | ForEach-Object {
            Write-Host "  $($_.Name)" -ForegroundColor White
        }
    }
} else {
    Write-Host "`n======================================" -ForegroundColor Red
    Write-Host "  ✗ 2D Inference 失敗 (Exit Code: $exitCode)" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
}
