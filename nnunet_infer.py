import os
import subprocess
import warnings
import sys
warnings.filterwarnings("ignore")

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

output_dir = os.path.join(base_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    # 注意：用戶已手動組織測試數據，因此跳過 organize_test_data()
    
    input_dir = os.path.join(dataset_dir, "imagesTs")
    
    # 計算測試檔案數量
    test_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    print("\n" + "="*60)
    print("🔮 開始推論測試集")
    print("="*60)
    print(f"輸入目錄: {input_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"資料集 ID: {dataset_id}")
    print(f"配置: 3d_fullres")
    print(f"測試檔案數量: {len(test_files)}")
    print("\n測試檔案清單:")
    for i, f in enumerate(test_files[:10], 1):  # 顯示前10個檔案
        print(f"  {i}. {f}")
    if len(test_files) > 10:
        print(f"  ... 還有 {len(test_files) - 10} 個檔案")
    print("\n開始執行 nnUNetv2_predict...\n")

    # 運行 nnUNetv2_predict 對測試集
    subprocess.run(["nnUNetv2_predict", "-i", input_dir, "-o", output_dir,
                    "-d", dataset_id, "-c", "3d_fullres", "-f", "all", "--save_probabilities"])

    print("\n" + "="*60)
    print("✓ 推論完成！")
    print(f"  所有預測檔案已保存至: {output_dir}")
    
    # 顯示輸出檔案
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
    print(f"  生成的預測檔案數量: {len(output_files)}")
    print("="*60)