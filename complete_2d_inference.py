"""
簡單的 2D Inference 腳本 - 完成剩餘的 validation cases
直接使用 Python API 避免環境變數問題
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

# 剩餘需要處理的 cases
remaining_cases = ["patient0031", "patient0033"]

base_dir = Path(r"C:\CardiacSeg")
input_folder = base_dir / "nnUNet_raw" / "Dataset001_CardiacSeg" / "imagesTr"
output_folder = base_dir / "inference_2d_validation"
# 不要包含 fold_0，API 會自動加上
model_folder = base_dir / "nnUNet_results" / "Dataset001_CardiacSeg" / "nnUNetTrainer__nnUNetPlans__2d"

print("=" * 70)
print("2D Inference - 剩餘 Validation Cases")
print("=" * 70)
print(f"模型: {model_folder}")
print(f"輸出: {output_folder}")
print(f"Cases: {remaining_cases}")
print("=" * 70)

# 準備輸入檔案
input_files = []
for case in remaining_cases:
    input_file = input_folder / f"{case}_0000.nii.gz"
    if input_file.exists():
        input_files.append(str(input_file))
        print(f">> {case}_0000.nii.gz")
    else:
        print(f"XX 找不到: {case}_0000.nii.gz")

if not input_files:
    print("\n錯誤: 找不到任何輸入檔案")
    exit(1)

# 設定 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用 device: {device}")

# 初始化 predictor
print("\n初始化 predictor...")
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,  # 關閉 TTA 加速
    perform_everything_on_device=True,
    device=device,
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

# 載入模型
print("載入模型...")
predictor.initialize_from_trained_model_folder(
    str(model_folder),
    use_folds=(0,),
    checkpoint_name='checkpoint_best.pth'
)

print(">> 模型載入完成")

# 執行預測
print("\n開始預測...")
print("=" * 70)

try:
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_files,
        output_folder_or_list_of_truncated_output_files=str(output_folder),
        save_probabilities=False,
        overwrite=False,  # 不覆蓋已存在的
        num_processes_preprocessing=1,  # 減少 workers
        num_processes_segmentation_export=0,  # 0 = 在主進程中執行，避免 Windows multiprocessing 問題
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    print("=" * 70)
    print(">> 預測完成！")
    
    # 檢查輸出
    output_files = list(output_folder.glob("*.nii.gz"))
    print(f"\n總共產生 {len(output_files)} 個預測檔案")
    
except Exception as e:
    print(f"\n錯誤: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
