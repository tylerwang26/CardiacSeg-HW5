"""
2D Model Validation Inference Script
使用 checkpoint_best.pth 對所有驗證案例進行預測
"""
import os
import subprocess
import sys
from pathlib import Path

# 設定基礎目錄
base_dir = Path(__file__).parent.absolute()
os.chdir(base_dir)

# nnU-Net 環境變數
os.environ["nnUNet_raw"] = str(base_dir / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(base_dir / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(base_dir / "nnUNet_results")

# 設定路徑
dataset_id = "001"
config = "2d"
fold = "0"

# 模型路徑
model_folder = base_dir / "nnUNet_results" / f"Dataset{dataset_id}_CardiacSeg" / f"nnUNetTrainer__nnUNetPlans__{config}" / f"fold_{fold}"
checkpoint = "checkpoint_best.pth"

# 輸入：使用 labelsTr 作為驗證集（因為我們有 ground truth）
input_folder = base_dir / "nnUNet_raw" / f"Dataset{dataset_id}_CardiacSeg" / "labelsTr"
output_folder = base_dir / "inference_2d_validation"
output_folder.mkdir(exist_ok=True)

print("="*70)
print("2D Model Validation Inference")
print("="*70)
print(f"模型路徑: {model_folder}")
print(f"Checkpoint: {checkpoint}")
print(f"輸入資料夾: {input_folder}")
print(f"輸出資料夾: {output_folder}")
print(f"配置: {config}")
print(f"Fold: {fold}")
print("="*70)

# 檢查模型是否存在
if not (model_folder / checkpoint).exists():
    print(f"錯誤：找不到 checkpoint: {model_folder / checkpoint}")
    sys.exit(1)

# 讀取 splits_final.json 以獲取驗證案例
import json
splits_file = base_dir / "nnUNet_preprocessed" / f"Dataset{dataset_id}_CardiacSeg" / "splits_final.json"
with open(splits_file, 'r') as f:
    splits = json.load(f)

val_cases = splits[int(fold)]['val']
print(f"\nFold {fold} 驗證案例 ({len(val_cases)} 個):")
for i, case in enumerate(val_cases, 1):
    print(f"  {i}. {case}")

# 創建臨時輸入資料夾，只包含驗證案例
temp_input = base_dir / "temp_val_input"
temp_input.mkdir(exist_ok=True)

print(f"\n準備驗證案例到臨時資料夾...")
raw_images = base_dir / "nnUNet_raw" / f"Dataset{dataset_id}_CardiacSeg" / "imagesTr"
for case in val_cases:
    src = raw_images / f"{case}_0000.nii.gz"
    dst = temp_input / f"{case}_0000.nii.gz"
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"  複製: {case}_0000.nii.gz")
    else:
        print(f"  警告: 找不到 {src}")

print(f"\n執行 nnUNetv2_predict...")

# 構建 nnUNetv2_predict 命令（使用 -m 參數直接指定模型資料夾）
predict_cmd = [
    sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data",
    "-i", str(temp_input),
    "-o", str(output_folder),
    "-m", str(model_folder),  # 直接指定模型資料夾
    "-chk", checkpoint,
    "--disable_tta",  # 禁用測試時增強以加快速度
    "-device", "cuda"
]

print(f"\n命令: {' '.join(predict_cmd)}")
print("\n開始預測...")

try:
    subprocess.run(predict_cmd, check=True)
    print("\n✓ 預測完成!")
    print(f"結果保存在: {output_folder}")
    
    # 列出輸出檔案
    pred_files = sorted(output_folder.glob("*.nii.gz"))
    print(f"\n產生的預測檔案 ({len(pred_files)} 個):")
    for pf in pred_files:
        print(f"  - {pf.name}")
        
except subprocess.CalledProcessError as e:
    print(f"\n✗ 預測失敗: {e}")
    sys.exit(1)
