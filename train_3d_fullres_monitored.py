import os
import sys
import subprocess
import time
import re
import threading
import argparse
from pathlib import Path

# Configuration
DATASET_ID = "001"
DATASET_NAME = "CardiacSeg"
CONFIG = "3d_fullres"
FOLD = "0"
DEVICE = "cuda"
TARGET_DICE = 0.90  # 目標 Dice，達到此分數自動停止
CHECK_INTERVAL = 30 # 檢查間隔 (秒)
MAX_EPOCHS = 1000   # 最大 Epochs

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "nnUNet_raw")
PREP_DIR = os.path.join(BASE_DIR, "nnUNet_preprocessed")
RES_DIR = os.path.join(BASE_DIR, "nnUNet_results")

# Set Environment Variables
os.environ["NNUNET_RAW"] = RAW_DIR
os.environ["NNUNET_PREPROCESSED"] = PREP_DIR
os.environ["NNUNET_RESULTS"] = RES_DIR
os.environ["nnUNet_raw"] = RAW_DIR
os.environ["nnUNet_preprocessed"] = PREP_DIR
os.environ["nnUNet_results"] = RES_DIR
os.environ["nnUNet_n_epochs"] = str(MAX_EPOCHS)
# Fix for potential torch inductor issue
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

def get_latest_log_file(trainer_output_dir):
    if not os.path.exists(trainer_output_dir):
        return None
    files = [f for f in os.listdir(trainer_output_dir) if f.startswith("training_log") and f.endswith(".txt")]
    if not files:
        return None
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(trainer_output_dir, x)))
    return os.path.join(trainer_output_dir, files[-1])

def monitor_training(process, trainer_output_dir):
    print(f"[*] 監控目錄: {trainer_output_dir}")
    
    best_dice = 0.0
    last_epoch = -1
    # Wait for directory to be created
    while not os.path.exists(trainer_output_dir) and process.poll() is None:
        time.sleep(5)
    while process.poll() is None:
        log_file = get_latest_log_file(trainer_output_dir)
        if log_file:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Extract Epoch
                    epoch_matches = re.findall(r"Current epoch: (\d+)", content)
                    if epoch_matches:
                        current_epoch = int(epoch_matches[-1])
                        if current_epoch != last_epoch:
                            # print(f"[Monitor] 目前 Epoch: {current_epoch}")
                            last_epoch = current_epoch

                    # Extract Dice
                    matches = re.findall(r"Mean Validation Dice: ([0-9\.]+)", content)
                    if matches:
                        current_dice = float(matches[-1])
                        
                        if current_dice > best_dice:
                            best_dice = current_dice
                            print(f"[Monitor] Epoch {last_epoch} - 新的最佳 Dice: {best_dice:.4f}")
                        
                        if current_dice >= TARGET_DICE:
                            print(f"\n[Monitor] 達到目標 Dice ({current_dice} >= {TARGET_DICE})! 正在停止訓練...")
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            print("[Monitor] 訓練已停止。")
                            return
            except Exception as e:
                # print(f"[Monitor] 讀取日誌錯誤: {e}")
                pass
        
        time.sleep(CHECK_INTERVAL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=1000)
    args = parser.parse_args()

    FOLD = str(args.fold)
    MAX_EPOCHS = args.max_epochs
    os.environ["nnUNet_n_epochs"] = str(MAX_EPOCHS)

    print(f"=== 啟動 3D Fullres 訓練 (Dataset {DATASET_ID}) ===")
    print(f"配置: {CONFIG}, Fold: {FOLD}, Device: {DEVICE}")
    print(f"最大 Epochs: {MAX_EPOCHS}")
    print(f"早停目標 Dice: {TARGET_DICE}")
    
    # Determine Python Executable (Prefer venv)
    venv_python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_exe = venv_python
        print(f"Using venv python: {python_exe}")
    else:
        python_exe = sys.executable
        print(f"Using system python: {python_exe}")

    # Command
    # Use patched runner to fix TypeError: str expected, not int
    cmd = [
        python_exe, "run_training_patched.py",
        DATASET_ID, CONFIG, FOLD,
        "-p", "nnUNetPlans",
        "-device", DEVICE,
        "--npz", # Save softmax predictions for validation
        "--c" # Continue training
    ]
    
    print(f"執行指令: {' '.join(cmd)}")
    
    # Determine output directory to monitor
    trainer_name = "nnUNetTrainer" 
    plans_name = "nnUNetPlans" 
    output_dir = os.path.join(RES_DIR, f"Dataset{DATASET_ID}_{DATASET_NAME}", f"{trainer_name}__{plans_name}__{CONFIG}", f"fold_{FOLD}")
    
    # Start process
    process = subprocess.Popen(cmd, env=os.environ.copy())
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_training, args=(process, output_dir))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[Main] 收到中斷訊號，正在停止訓練...")
        process.terminate()
        process.wait()
    
    print("\n=== 訓練程序結束 ===")

if __name__ == "__main__":
    main()
