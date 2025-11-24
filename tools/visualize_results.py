"""只產生視覺化（inference 結果已經存在）"""
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    
    # 建立 colormap
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
    
    return dice_scores

def main():
    # 設定路徑
    raw_folder = Path(r'c:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg')
    output_folder = Path(r'c:\CardiacSeg\inference_output')
    viz_folder = output_folder / 'visualizations'
    viz_folder.mkdir(exist_ok=True)
    
    # 找出所有已產生的預測
    pred_files = list(output_folder.glob('patient*.nii.gz'))
    
    if not pred_files:
        print("錯誤: 沒有找到預測檔案")
        return
    
    print(f"找到 {len(pred_files)} 個預測檔案")
    
    # 為每個案例產生視覺化
    all_dice_scores = {1: [], 2: [], 3: []}
    
    for pred_path in pred_files:
        # 正確移除 .nii.gz 得到 patient0003
        case_name = pred_path.name.replace('.nii.gz', '')
        
        # 找對應的原始影像和 GT
        img_path = raw_folder / 'imagesTr' / f'{case_name}_0000.nii.gz'
        gt_path = raw_folder / 'labelsTr' / f'{case_name}.nii.gz'
        
        if not img_path.exists() or not gt_path.exists():
            print(f"警告: {case_name} 的原始檔案不存在，跳過")
            continue
        
        # 讀取並計算整體 Dice
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        
        print(f"\n{case_name}:")
        case_dice = {}
        for label in [1, 2, 3]:
            dice = compute_dice(pred, gt, label)
            if not np.isnan(dice):
                all_dice_scores[label].append(dice)
                case_dice[label] = dice
                print(f"  Label {label} Dice: {dice:.4f}")
        
        # 產生視覺化（多個 slice）
        for slice_offset in [0.3, 0.5, 0.7]:
            slice_idx = int(pred.shape[0] * slice_offset)
            viz_path = viz_folder / f'{case_name}_slice{slice_idx:03d}.png'
            try:
                create_overlay_visualization(
                    img_path, pred_path, gt_path, viz_path, slice_idx
                )
                print(f"  ✓ {viz_path.name}")
            except Exception as e:
                print(f"  ✗ 視覺化錯誤: {e}")
    
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
    
    # 計算平均
    if any(all_dice_scores[i] for i in [1, 2, 3]):
        all_scores = []
        for label in [1, 2, 3]:
            all_scores.extend(all_dice_scores[label])
        if all_scores:
            print(f"\n  Overall Mean Dice: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    
    print(f"\n所有視覺化已儲存至: {viz_folder}")
    print("="*60)

if __name__ == '__main__':
    main()
