"""快速 inference 並產生視覺化範例與指標"""
import os
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 設定環境變數
os.environ['nnUNet_raw'] = r'c:\CardiacSeg\nnUNet_raw'
os.environ['nnUNet_preprocessed'] = r'c:\CardiacSeg\nnUNet_preprocessed'
os.environ['nnUNet_results'] = r'c:\CardiacSeg\nnUNet_results'

def run_inference(
    model_folder: str,
    input_folder: str,
    output_folder: str,
    folds: tuple = (0,),
    save_npz: bool = False
):
    """執行 nnUNet inference"""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 初始化預測器
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth'
    )
    
    # 執行預測
    predictor.predict_from_files(
        [[f] for f in input_folder],  # 每個病人一個 list
        output_folder,
        save_probabilities=save_npz,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

def compute_dice(pred, gt, label):
    """計算單一標籤的 Dice 係數"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    if pred_mask.sum() + gt_mask.sum() == 0:
        return float('nan')
    
    dice = 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())
    return dice

def create_overlay_visualization(
    image_path: Path,
    pred_path: Path,
    gt_path: Path,
    output_path: Path,
    slice_idx: int = None
):
    """建立疊加視覺化"""
    # 讀取影像
    img = sitk.ReadImage(str(image_path))
    pred = sitk.ReadImage(str(pred_path))
    gt = sitk.ReadImage(str(gt_path))
    
    # 轉成 numpy
    img_arr = sitk.GetArrayFromImage(img)
    pred_arr = sitk.GetArrayFromImage(pred)
    gt_arr = sitk.GetArrayFromImage(gt)
    
    # 選擇中間 slice（若未指定）
    if slice_idx is None:
        slice_idx = img_arr.shape[0] // 2
    
    img_slice = img_arr[slice_idx]
    pred_slice = pred_arr[slice_idx]
    gt_slice = gt_arr[slice_idx]
    
    # 建立 colormap（背景透明，label 1-3 不同顏色）
    colors_pred = ['none', 'red', 'green', 'blue', 'yellow']
    colors_gt = ['none', 'cyan', 'magenta', 'orange', 'pink']
    cmap_pred = ListedColormap(colors_pred[:4])
    cmap_gt = ListedColormap(colors_gt[:4])
    
    # 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始影像
    axes[0].imshow(img_slice, cmap='gray', vmin=-100, vmax=300)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground Truth 疊加
    axes[1].imshow(img_slice, cmap='gray', vmin=-100, vmax=300)
    gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[1].imshow(gt_masked, cmap=cmap_gt, alpha=0.5, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction 疊加
    axes[2].imshow(img_slice, cmap='gray', vmin=-100, vmax=300)
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[2].imshow(pred_masked, cmap=cmap_pred, alpha=0.5, vmin=0, vmax=3)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 計算 Dice
    dice_scores = []
    for label in [1, 2, 3]:
        dice = compute_dice(pred_arr, gt_arr, label)
        if not np.isnan(dice):
            dice_scores.append(f"Label {label}: {dice:.3f}")
    
    fig.suptitle(f'{output_path.stem}\nSlice {slice_idx}\n' + ' | '.join(dice_scores), 
                 fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 設定路徑（model_folder 應該是 trainer 資料夾，不是 fold）
    model_folder = Path(r'c:\CardiacSeg\nnUNet_results\Dataset001_CardiacSeg\nnUNetTrainer__nnUNetPlans__3d_lowres')
    raw_folder = Path(r'c:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg')
    output_folder = Path(r'c:\CardiacSeg\inference_output')
    output_folder.mkdir(exist_ok=True)
    
    # 選擇驗證集中的幾個病人（從 splits_final.json 的 fold_0 validation）
    # 這裡手動選擇前 5 個驗證病人作為範例
    val_cases = ['patient0003', 'patient0006', 'patient0009', 'patient0013', 'patient0017']
    
    print("準備 inference 輸入檔案...")
    input_files = []
    gt_files = []
    for case in val_cases:
        img_path = raw_folder / 'imagesTr' / f'{case}_0000.nii.gz'
        gt_path = raw_folder / 'labelsTr' / f'{case}.nii.gz'
        if img_path.exists() and gt_path.exists():
            input_files.append(img_path)
            gt_files.append(gt_path)
        else:
            print(f"警告: {case} 檔案不存在，跳過")
    
    if not input_files:
        print("錯誤: 沒有找到可用的驗證檔案")
        return
    
    print(f"找到 {len(input_files)} 個驗證案例")
    print(f"開始 inference...")
    
    # 執行 inference
    try:
        run_inference(
            model_folder=str(model_folder),
            input_folder=[str(f) for f in input_files],  # 轉換為字串
            output_folder=str(output_folder),
            folds=(0,),
            save_npz=False
        )
    except Exception as e:
        print(f"Inference 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n產生視覺化...")
    viz_folder = output_folder / 'visualizations'
    viz_folder.mkdir(exist_ok=True)
    
    # 為每個案例產生視覺化
    all_dice_scores = {1: [], 2: [], 3: []}
    
    for img_path, gt_path in zip(input_files, gt_files):
        # 正確處理檔名：移除 _0000.nii.gz 得到 patient0003
        case_name = Path(img_path).name.replace('_0000.nii.gz', '')
        pred_path = output_folder / f'{case_name}.nii.gz'
        
        if not pred_path.exists():
            print(f"警告: 預測檔案不存在: {pred_path}")
            continue
        
        # 讀取並計算整體 Dice
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        
        print(f"\n{case_name}:")
        for label in [1, 2, 3]:
            dice = compute_dice(pred, gt, label)
            if not np.isnan(dice):
                all_dice_scores[label].append(dice)
                print(f"  Label {label} Dice: {dice:.4f}")
        
        # 產生視覺化（多個 slice）
        for slice_offset in [0.3, 0.5, 0.7]:  # 不同位置的 slice
            slice_idx = int(pred.shape[0] * slice_offset)
            viz_path = viz_folder / f'{case_name}_slice{slice_idx:03d}.png'
            try:
                create_overlay_visualization(
                    img_path, pred_path, gt_path, viz_path, slice_idx
                )
                print(f"  已儲存視覺化: {viz_path.name}")
            except Exception as e:
                print(f"  視覺化錯誤: {e}")
    
    # 總結統計
    print("\n" + "="*60)
    print("整體統計 (平均 Dice):")
    for label in [1, 2, 3]:
        if all_dice_scores[label]:
            mean_dice = np.mean(all_dice_scores[label])
            std_dice = np.std(all_dice_scores[label])
            print(f"  Label {label}: {mean_dice:.4f} ± {std_dice:.4f}")
        else:
            print(f"  Label {label}: N/A")
    
    print(f"\n所有結果已儲存至: {output_folder}")
    print(f"視覺化圖片位於: {viz_folder}")
    print("="*60)

if __name__ == '__main__':
    main()
