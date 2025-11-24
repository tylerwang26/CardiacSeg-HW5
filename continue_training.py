"""直接繼續 nnUNet 訓練的簡化腳本"""
import os
import sys

# 設定 nnU-Net 環境變數
base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["nnUNet_raw"] = os.path.join(base_dir, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(base_dir, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(base_dir, "nnUNet_results")
os.environ["nnUNet_n_epochs"] = "250"

print("="*60)
print("繼續訓練 3d_lowres - Dataset001_CardiacSeg")
print("="*60)
print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results: {os.environ['nnUNet_results']}")
print(f"Epochs: {os.environ['nnUNet_n_epochs']}")
print("="*60)

# 直接調用 nnUNet 訓練模組
from nnunetv2.run.run_training import run_training_entry

# 設定訓練參數
sys.argv = [
    'run_training',
    '001',           # dataset_id
    '3d_lowres',     # configuration
    '0',             # fold
    '-p', 'nnUNetPlans',
    '-device', 'cuda',
    '--npz',
    '--c'            # continue from checkpoint
]

print("\n開始訓練...")
print(f"指令: {' '.join(sys.argv)}\n")

# 執行訓練
run_training_entry()
