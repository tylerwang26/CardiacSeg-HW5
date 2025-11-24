"""
2D Model Validation Inference - 使用 Python API
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import json

# 設定基礎目錄
base_dir = Path(__file__).parent.absolute()
os.chdir(base_dir)

# 設定環境變數（在 import nnUNet 之前）
os.environ["nnUNet_raw"] = str(base_dir / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(base_dir / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(base_dir / "nnUNet_results")

# 讀取模型（使用上一層目錄，因為 predictor 會自動添加 fold_X）
model_folder = base_dir / "nnUNet_results" / "Dataset001_CardiacSeg" / "nnUNetTrainer__nnUNetPlans__2d"
checkpoint_path = model_folder / "fold_0" / "checkpoint_best.pth"

print("="*70)
print("2D Model Validation Inference (Python API)")
print("="*70)
print(f"模型資料夾: {model_folder}")
print(f"Checkpoint: {checkpoint_path.name}")

# 載入 checkpoint 檢查
checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
print(f"✓ 載入 checkpoint: epoch {checkpoint.get('current_epoch', 'unknown')}")

# 讀取驗證案例列表
splits_file = base_dir / "nnUNet_preprocessed" / "Dataset001_CardiacSeg" / "splits_final.json"
with open(splits_file, 'r') as f:
    splits = json.load(f)

fold = 0
val_cases = splits[fold]['val']
print(f"\nFold {fold} 驗證案例 ({len(val_cases)} 個):")
for case in val_cases:
    print(f"  - {case}")

# 使用 nnUNetv2 的 nnUNetPredictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# 初始化 predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,  # 禁用 TTA
    perform_everything_on_device=True,
    device=torch.device('cuda'),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

print("\n初始化 predictor...")
predictor.initialize_from_trained_model_folder(
    str(model_folder),
    use_folds=(fold,),
    checkpoint_name='checkpoint_best.pth'
)

# 準備輸入輸出路徑
input_folder = base_dir / "nnUNet_raw" / "Dataset001_CardiacSeg" / "imagesTr"
output_folder = base_dir / "inference_2d_validation"
output_folder.mkdir(exist_ok=True)

# 準備輸入檔案列表
input_files = []
for case in val_cases:
    img_path = input_folder / f"{case}_0000.nii.gz"
    if img_path.exists():
        input_files.append(str(img_path))
    else:
        print(f"警告: 找不到 {img_path}")

if __name__ == '__main__':
    print(f"\n開始預測 {len(input_files)} 個案例...")

    # 執行預測（禁用多進程避免 Windows 問題）
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[f] for f in input_files],
        output_folder_or_list_of_truncated_output_files=str(output_folder),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,  # Windows multiprocessing 安全
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    print("\n✓ 預測完成!")
    print(f"結果保存在: {output_folder}")

    # 列出輸出檔案
    pred_files = sorted(output_folder.glob("*.nii.gz"))
    print(f"\n產生的預測檔案 ({len(pred_files)} 個):")
    for pf in pred_files:
        print(f"  - {pf.name}")

    print("\n" + "="*70)
