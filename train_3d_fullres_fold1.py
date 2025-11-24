import os
import sys
import subprocess
import time
import re
import threading
from pathlib import Path

# Configuration
DATASET_ID = "001"
DATASET_NAME = "CardiacSeg"
CONFIG = "3d_fullres"
FOLD = "1"  # Changed to Fold 1
DEVICE = "cuda"
TARGET_DICE = 0.92  # Slightly higher target for Fold 1
CHECK_INTERVAL = 30
MAX_EPOCHS = 1000

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
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

def get_latest_log_file(trainer_output_dir):
    if not os.path.exists(trainer_output_dir):
        return None
    files = [f for f in os.listdir(trainer_output_dir) if f.startswith("training_log") and f.endswith(".txt")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(trainer_output_dir, x)))
    return os.path.join(trainer_output_dir, files[-1])

def monitor_training(process, trainer_output_dir):
    print(f"[*] 啟動監控線程 (Fold {FOLD})... 目標 Dice: {TARGET_DICE}")
    print(f"[*] 監控目錄: {trainer_output_dir}")
    
    best_dice = 0.0
    
    while process.poll() is None:
        log_file = get_latest_log_file(trainer_output_dir)
        if log_file:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Read last few lines
                    content = "".join(lines[-20:])
                    
                    # Extract Dice
                    # Format: "Mean Validation Dice: 0.8543"
                    dice_match = re.search(r"Mean Validation Dice: ([0-9.]+)", content)
                    if dice_match:
                        current_dice = float(dice_match.group(1))
                        if current_dice > best_dice:
                            best_dice = current_dice
                            print(f"[+] Fold {FOLD} New Best Dice: {best_dice:.4f}")
                            
                        if best_dice >= TARGET_DICE:
                            print(f"\n[!] 達到目標 Dice {TARGET_DICE}！停止訓練...")
                            process.terminate()
                            break
            except Exception as e:
                pass
        
        time.sleep(CHECK_INTERVAL)

def main():
    cmd = [
        "nnUNetv2_train",
        DATASET_ID,
        CONFIG,
        FOLD,
        "--npz"  # Save softmax predictions for ensembling
    ]
    
    print(f"[*] 開始訓練 Fold {FOLD}...")
    print(f"[*] Command: {' '.join(cmd)}")
    
    # Determine trainer output directory for monitoring
    # Default: nnUNetTrainer__nnUNetPlans__3d_fullres
    trainer_name = "nnUNetTrainer"
    plans_name = "nnUNetPlans"
    trainer_output_dir = os.path.join(
        RES_DIR, 
        f"Dataset{DATASET_ID}_{DATASET_NAME}",
        f"{trainer_name}__{plans_name}__{CONFIG}",
        f"fold_{FOLD}"
    )
    
    # Start training process
    if sys.platform == "win32":
        process = subprocess.Popen(cmd, shell=True)
    else:
        process = subprocess.Popen(cmd)
        
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_training, args=(process, trainer_output_dir))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[*] 用戶中斷訓練")
        process.terminate()

if __name__ == "__main__":
    main()
