# cite: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

import os
import subprocess
import sys
from pathlib import Path
import multiprocessing
import json
import argparse
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

# nnU-Net v2 環境變數設定（同時設定大小寫，避免不同版本引用不同名稱）
_raw_dir = os.path.join(base_dir, "nnUNet_raw")
_prep_dir = os.path.join(base_dir, "nnUNet_preprocessed")
_res_dir = os.path.join(base_dir, "nnUNet_results")
os.environ["NNUNET_RAW"] = _raw_dir
os.environ["NNUNET_PREPROCESSED"] = _prep_dir
os.environ["NNUNET_RESULTS"] = _res_dir
os.environ["nnUNet_raw"] = _raw_dir
os.environ["nnUNet_preprocessed"] = _prep_dir
os.environ["nnUNet_results"] = _res_dir

# 若偵測到使用系統 Python 而非虛擬環境，給出提醒
if ".venv" not in sys.executable:
    print("WARNING: 目前使用的 Python 並非專案虛擬環境 (.venv)。建議先執行: source .venv/bin/activate 或使用 ./.venv/bin/python。")

# 數據集 ID 和名稱
dataset_id = "001"
dataset_name = "CardiacSeg"
dataset_dir = os.path.join(os.environ["NNUNET_RAW"], f"Dataset{dataset_id}_{dataset_name}")

def install_dependencies():
    required = ["nnunetv2", "blosc2"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"安裝缺失套件: {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def generate_dataset_json(force_regenerate=False):
    json_path = os.path.join(dataset_dir, "dataset.json")
    if os.path.exists(json_path) and not force_regenerate:
        print(f"dataset.json 已存在於 {json_path}，跳過生成。若需強制重新生成，請設定 force_regenerate=True。")
        return

    # 動態掃描 imagesTr 和 labelsTr 以生成 training 清單（匹配用戶檔名模式：patientXXXX.nii.gz 和 patientXXXX_gt.nii.gz）
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    labels_tr_dir = os.path.join(dataset_dir, "labelsTr")
    images_ts_dir = os.path.join(dataset_dir, "imagesTs")

    training = []
    for img_file in sorted(os.listdir(images_tr_dir)):
        if img_file.endswith(".nii.gz"):
            # 假設影像檔名如 patient0001.nii.gz，轉換為 nnU-Net 標準格式（添加 _0000 若缺少）
            if '_' not in img_file or not img_file.split('_')[-1].startswith('0000'):
                # 警告：nnU-Net 期望影像檔名如 patient0001_0000.nii.gz（單模態）
                print(f"警告：影像檔 {img_file} 缺少 _0000 後綴。建議重命名為 {img_file.replace('.nii.gz', '_0000.nii.gz')} 以符合 nnU-Net 標準。")
                std_img_file = img_file.replace('.nii.gz', '_0000.nii.gz')
            else:
                std_img_file = img_file

            # 標籤檔名：patientXXXX_gt.nii.gz，nnU-Net 期望無 _gt（如 patient0001.nii.gz）
            case_id = img_file.split('.')[0]  # e.g., patient0001
            label_file = f"{case_id}_gt.nii.gz"
            std_label_file = f"{case_id}.nii.gz"  # 建議移除 _gt
            if os.path.exists(os.path.join(labels_tr_dir, label_file)):
                print(f"警告：標籤檔 {label_file} 有 _gt 後綴。建議重命名為 {std_label_file} 以符合 nnU-Net 標準。")
                training.append({"image": f"./imagesTr/{std_img_file}", "label": f"./labelsTr/{std_label_file}"})

    test = [f"./imagesTs/{f.replace('.nii.gz', '_0000.nii.gz') if '_' not in f or not f.split('_')[-1].startswith('0000') else f}" 
            for f in sorted(os.listdir(images_ts_dir)) if f.endswith(".nii.gz")]

    dataset_json = {
        "channel_names": {"0": "CT"},  # 調整為您的模態，例如 "MRI"
        "labels": {
            "background": 0,
            "label1": 1,  # 調整為實際標籤，例如 "heart": 1, "aorta": 2 等
            # 添加更多標籤，如 "myocardium": 2 等
        },
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Cardiac segmentation dataset",
        "reference": "Your reference",
        "licence": "Your licence",
        "release": "1.0",
        "tensorImageSize": "3D",  # 或 "4D" 若為時序資料
        "training": training,
        "test": test
    }

    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"已生成/更新 dataset.json 於 {json_path}。請手動驗證內容，並重命名檔案以匹配 nnU-Net 標準（影像: case_0000.nii.gz，標籤: case.nii.gz）！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nnU-Net 訓練腳本（支援自訂參數）')
    parser.add_argument('--epochs', type=int, default=250, help='訓練的 epoch 數量（預設: 250，nnUNet 原始預設 1000）')
    parser.add_argument('--fold', type=str, default='0', help='訓練的 fold（0-4 或 all，預設: 0）')
    parser.add_argument('--config', type=str, default='3d_fullres', help='訓練配置（2d, 3d_fullres, 3d_lowres，預設: 3d_fullres）')
    parser.add_argument('--device', type=str, default='mps', help='使用的裝置（cuda, mps, cpu，預設: mps）')
    parser.add_argument('--continue-training', action='store_true', help='從最新 checkpoint 繼續訓練')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳過預處理步驟（假設已完成）')
    args = parser.parse_args()

    multiprocessing.freeze_support()
    install_dependencies()

    # 生成 dataset.json 若缺少（保留現有 json 不覆蓋）
    generate_dataset_json(force_regenerate=False)

    # 注意：用戶已手動組織數據，因此跳過 organize_data()

    if not args.skip_preprocess:
        print("\n" + "="*60)
        print("🔧 步驟 1: 規劃和預處理數據")
        print("="*60)
        print(f"資料集 ID: {dataset_id}")
        print(f"資料集名稱: {dataset_name}")
        print(f"原始資料: {os.environ['NNUNET_RAW']}")
        print(f"預處理資料: {os.environ['NNUNET_PREPROCESSED']}")
        print("\n開始執行 nnUNetv2_plan_and_preprocess...\n")
        
        # 步驟 1: 規劃和預處理數據（只需運行一次）
        subprocess.run([
            "nnUNetv2_plan_and_preprocess",
            "-d", dataset_id,
            "--verify_dataset_integrity"
        ], check=True)
    else:
        print("\n>> 跳過預處理步驟（使用 --skip-preprocess）")

    print("\n" + "="*60)
    print(">> 步驟 2: 開始訓練模型")
    print("="*60)
    print(f"資料集 ID: {dataset_id}")
    print(f"配置: {args.config}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"裝置: {args.device}")
    print(f"繼續訓練: {'是' if args.continue_training else '否'}")
    print(f"結果儲存: {os.environ['NNUNET_RESULTS']}")
    print("\n開始執行 nnUNetv2_train...\n")
    
    # 步驟 2: 訓練
    # 使用當前 Python 直譯器以避免 PATH 對應到系統安裝的舊版/錯誤環境
    train_cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        dataset_id, args.config, args.fold,
        "-p", "nnUNetPlans",
        "-device", args.device,
        "--npz"
    ]
    
    # 設定自訂 epochs（透過環境變數，nnUNet 會讀取）
    train_env = os.environ.copy()
    train_env["nnUNet_n_epochs"] = str(args.epochs)
    # nnUNet (and dependencies) may set environment variables expecting string values.
    # Ensure numeric environment variables are strings to avoid TypeError: str expected, not int
    # Example: running nnunetv2.run.run_training may attempt `os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1`
    # which raises when the right-hand side is an int. Provide a safe string value here.
    train_env["TORCHINDUCTOR_COMPILE_THREADS"] = str(train_env.get("TORCHINDUCTOR_COMPILE_THREADS", "1"))
    
    if args.continue_training:
        train_cmd.append("--c")
        print("📂 將從最新的 checkpoint 繼續訓練...")
    
    subprocess.run(train_cmd, env=train_env, check=True)

    print("\n" + "="*60)
    print("✓ 訓練完成！")
    print(f"  模型已儲存於: {os.environ['NNUNET_RESULTS']}")
    print("="*60)