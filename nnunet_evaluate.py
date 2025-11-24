import os
import subprocess
import numpy as np
import nibabel as nib
import sys

try:
    from tqdm import tqdm
except ImportError:
    print("安裝 tqdm 以顯示進度條...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 跨平台：使用當前腳本所在目錄作為 base_dir
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"腳本 base_dir: {base_dir}")  # 診斷打印

# nnU-Net v2 環境變數設定
os.environ["NNUNET_RAW"] = os.path.join(base_dir, "nnUNet_raw")
os.environ["NNUNET_PREPROCESSED"] = os.path.join(base_dir, "nnUNet_preprocessed")
os.environ["NNUNET_RESULTS"] = os.path.join(base_dir, "nnUNet_results")

dataset_id = "001"
dataset_name = "CardiacSeg"
dataset_dir = os.path.join(os.environ["NNUNET_RAW"], f"Dataset{dataset_id}_{dataset_name}")
val_output_dir = os.path.join(base_dir, "val_predictions")
os.makedirs(val_output_dir, exist_ok=True)

# 預設使用 imagesTr 作為驗證來源（有標註）
images_val_dir = os.path.join(dataset_dir, "imagesTr")
labels_dir = os.path.join(dataset_dir, "labelsTr")

def dice_iou_per_label(pred, gt, label_val):
    pred_mask = (pred == label_val)
    gt_mask = (gt == label_val)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    union = pred_sum + gt_sum - inter
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    return dice, iou

def evaluate_folder(pred_dir, labels_dir):
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    all_case_dice = []
    all_case_iou = []
    missing = []
    
    print(f"\n📊 評估 {len(label_files)} 個案例...")
    
    for lf in tqdm(label_files, desc="計算指標", unit="案例"):
        case = lf.replace('.nii.gz', '')
        pred_file = f"{case}.nii.gz"  # nnUNet 預設輸出分割檔名為 case.nii.gz
        pred_path = os.path.join(pred_dir, pred_file)
        label_path = os.path.join(labels_dir, lf)
        if not os.path.exists(pred_path):
            missing.append(pred_file)
            tqdm.write(f"  ⚠ 缺少預測檔案: {pred_file}")
            continue
        
        gt = nib.load(label_path).get_fdata().astype(np.int32)
        pred = nib.load(pred_path).get_fdata().astype(np.int32)
        labels = sorted(list(set(np.unique(gt)).union(set(np.unique(pred)))))
        labels = [l for l in labels if l != 0]
        if len(labels) == 0:
            continue
        dices, ious = [], []
        for l in labels:
            d, j = dice_iou_per_label(pred, gt, l)
            dices.append(d)
            ious.append(j)
        case_dice = np.mean(dices)
        case_iou = np.mean(ious)
        all_case_dice.append(case_dice)
        all_case_iou.append(case_iou)
        tqdm.write(f"  ✓ {case}: Dice={case_dice:.4f}, IoU={case_iou:.4f}")
    
    mean_dice = float(np.mean(all_case_dice)) if all_case_dice else 0.0
    mean_iou = float(np.mean(all_case_iou)) if all_case_iou else 0.0
    return mean_dice, mean_iou, missing

if __name__ == '__main__':
    # 計算驗證檔案數量
    val_files = [f for f in os.listdir(images_val_dir) if f.endswith('.nii.gz')]
    
    print("\n" + "="*60)
    print("🔍 開始驗證評估")
    print("="*60)
    print(f"輸入目錄: {images_val_dir}")
    print(f"標籤目錄: {labels_dir}")
    print(f"輸出目錄: {val_output_dir}")
    print(f"資料集 ID: {dataset_id}")
    print(f"配置: 3d_fullres")
    print(f"驗證檔案數量: {len(val_files)}")
    print("\n開始執行 nnUNetv2_predict...\n")
    
    subprocess.run([
        "nnUNetv2_predict",
        "-i", images_val_dir,
        "-o", val_output_dir,
        "-d", dataset_id,
        "-c", "3d_fullres",
        "-f", "all"
    ])

    mean_dice, mean_iou, missing = evaluate_folder(val_output_dir, labels_dir)
    
    print("\n" + "="*60)
    print("✓ 評估結果")
    print("="*60)
    print(f"  平均 Dice 係數: {mean_dice:.4f}")
    print(f"  平均 IoU 分數:  {mean_iou:.4f}")
    print(f"  成功評估案例: {len([f for f in os.listdir(val_output_dir) if f.endswith('.nii.gz')])}")
    if missing:
        print(f"  缺少預測檔案: {len(missing)}")
        for m in missing[:5]:
            print(f"    - {m}")
        if len(missing) > 5:
            print(f"    ... 還有 {len(missing) - 5} 個")
    print("="*60)