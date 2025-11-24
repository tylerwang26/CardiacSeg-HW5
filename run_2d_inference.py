"""
執行 2D Model Inference on Validation Set
使用 nnU-Net API 避免 multiprocessing 問題
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

def main():
    # Validation cases for fold_0
    val_cases = [
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
    ]
    
    base_dir = Path(__file__).parent
    
    # 輸入/輸出路徑
    input_folder = base_dir / "nnUNet_raw" / "Dataset001_CardiacSeg" / "imagesTr"
    output_folder = base_dir / "inference_2d_validation"
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # 模型路徑（不要包含 fold_X，API 會自動加上）
    model_folder = base_dir / "nnUNet_results" / "Dataset001_CardiacSeg" / "nnUNetTrainer__nnUNetPlans__2d"
    
    print("=" * 70)
    print("2D Model Inference on Validation Set")
    print("=" * 70)
    print(f"Model: {model_folder}")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Cases: {len(val_cases)} validation cases")
    print("=" * 70)
    
    # 檢查 checkpoint
    checkpoint = model_folder / "checkpoint_best.pth"
    if not checkpoint.exists():
        print(f"錯誤: Checkpoint 不存在: {checkpoint}")
        return
    
    print(f"\n>> Using checkpoint: {checkpoint}")
    print(f"  Size: {checkpoint.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 檢查輸入檔案
    print(f"\n檢查輸入檔案...")
    input_files = []
    for case in val_cases:
        # nnU-Net 格式: patient0009_0000.nii.gz
        input_file = input_folder / f"{case}_0000.nii.gz"
        if input_file.exists():
            input_files.append(str(input_file))
            print(f"  >> {case}_0000.nii.gz")
        else:
            print(f"  XX {case}_0000.nii.gz 不存在")
    
    if not input_files:
        print("\n錯誤: 找不到任何輸入檔案")
        return
    
    print(f"\n找到 {len(input_files)} 個輸入檔案")
    
    # 設定 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用 device: {device}")
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 初始化 predictor
    print("\n初始化 predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # 載入模型參數
    print("載入模型參數...")
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )
    
    print(">> 模型載入完成")
    
    # 執行預測
    print("\n開始預測...")
    print("=" * 70)
    
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_files,
        output_folder_or_list_of_truncated_output_files=str(output_folder),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    print("=" * 70)
    print(">> 預測完成！")
    
    # 檢查輸出
    output_files = list(output_folder.glob("*.nii.gz"))
    print(f"\n產生 {len(output_files)} 個預測檔案:")
    for f in sorted(output_files):
        print(f"  {f.name}")
    
    print("\n" + "=" * 70)
    print("2D Inference 完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
