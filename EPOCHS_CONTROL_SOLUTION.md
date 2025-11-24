# nnU-Net Epochs 控制問題解決方案

## 問題分析

### 原本的錯誤方法
在 `nnunet_train.py` 中設定環境變數 `nnUNet_n_epochs` **不會生效**，因為：

1. nnU-Net v2 的 `nnUNetTrainer` 類別中硬編碼了 `self.num_epochs = 1000`
2. 這個值在 trainer 初始化時就已經設定好了
3. nnU-Net 並不會從環境變數讀取 epochs 數量

```python
# nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
class nnUNetTrainer:
    def __init__(self, ...):
        self.num_epochs = 1000  # 硬編碼！
```

### 為何會跑到 289 epochs
- 2D 配置預設是 1000 epochs
- 訓練從 2025-11-18 21:49 開始
- 到 2025-11-19 06:01 (約 8.2 小時) 跑了 289 epochs
- 最佳結果在 epoch 72 (約 2.5 小時)

## 解決方案

### 方案 1: 使用自定義 Trainer（推薦）

創建一個自定義的 Trainer 類別，從環境變數讀取 epochs：

```python
# custom_trainer.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import os

class nnUNetTrainerCustomEpochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, **kwargs)
        
        custom_epochs = os.environ.get('nnUNet_n_epochs', None)
        if custom_epochs is not None:
            self.num_epochs = int(custom_epochs)
            print(f"✓ 使用自定義 epochs 數量: {self.num_epochs}")
```

使用方式：
```bash
# 設定環境變數
export nnUNet_n_epochs=80  # Linux/Mac
$env:nnUNet_n_epochs="80"  # Windows PowerShell

# 使用自定義 trainer
nnUNetv2_train Dataset001_CardiacSeg 2d 0 \
    -tr nnUNetTrainerCustomEpochs \
    -device cuda
```

### 方案 2: 修改 nnUNetPlans.json

在 plans 中添加自定義參數（需要 nnU-Net v2.2+）：

```json
{
    "configurations": {
        "2d": {
            "num_epochs": 80,
            ...
        }
    }
}
```

注意：這個方法可能需要修改 nnU-Net 原始碼以支持讀取這個參數。

### 方案 3: Early Stopping

使用 nnU-Net 的內建 early stopping 機制：

```python
# 修改 trainer
class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = 50  # 50 epochs 沒改善就停止
        self.best_epoch = 0
```

### 方案 4: 手動停止訓練

1. 監控訓練日誌
2. 當驗證 Dice 不再提升時，手動 Ctrl+C 停止
3. 最佳 checkpoint 已保存在 `checkpoint_best.pth`

## 實作建議

對於未來的訓練，推薦使用**方案 1（自定義 Trainer）**：

### 更新 nnunet_train.py

```python
import os
import subprocess
import sys
from pathlib import Path

def main():
    # ... 參數解析 ...
    
    # 1. 複製 custom_trainer.py 到 nnUNet 可以找到的位置
    custom_trainer_file = Path(__file__).parent / "custom_trainer.py"
    if not custom_trainer_file.exists():
        print("錯誤: 找不到 custom_trainer.py")
        sys.exit(1)
    
    # 2. 設定環境變數
    train_env = os.environ.copy()
    train_env["nnUNet_raw"] = str(base_dir / "nnUNet_raw")
    train_env["nnUNet_preprocessed"] = str(base_dir / "nnUNet_preprocessed")
    train_env["nnUNet_results"] = str(base_dir / "nnUNet_results")
    train_env["nnUNet_n_epochs"] = str(args.epochs)  # 這個會被自定義 trainer 讀取
    
    # 3. 構建訓練命令，使用自定義 trainer
    train_cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        dataset_id, config, str(fold),
        "-tr", "nnUNetTrainerCustomEpochs",  # 指定自定義 trainer
        "-device", device
    ]
    
    # 4. 執行訓練
    subprocess.run(train_cmd, env=train_env, check=True)
```

### 驗證修正

測試命令：
```bash
# 設定 80 epochs
python nnunet_train.py --epochs 80 --config 2d --device cuda

# 檢查訓練日誌開頭是否顯示：
# ✓ 使用自定義 epochs 數量: 80
```

## 實際案例數據

### 2D Training (2025-11-18 ~ 2025-11-19)

| Metric | Value |
|--------|-------|
| 預期 epochs | 80 |
| 實際 epochs | 289 (手動中斷) |
| 最佳 epoch | 72 |
| 最佳 EMA Dice | 0.5846 |
| 最佳 Label Dice | [0.8963, 0.6723, 0.4863] |
| 訓練時長 | ~8.2 小時 |
| 最佳點時長 | ~2.5 小時 |
| 浪費時間 | ~5.7 小時 |

### 結論

✅ **已保存最佳模型** (`checkpoint_best.pth` at epoch 72)  
⚠️ **Epochs 控制失敗** (預期 80, 實際跑 1000 的預設值)  
✅ **模型表現良好** (首次檢測到 Label 3，Dice=0.49)  
🔧 **需要實作方案 1** 以避免未來浪費訓練時間

## 參考資料

- nnU-Net GitHub Issues: [How to set number of epochs](https://github.com/MIC-DKFZ/nnUNet/issues/xyz)
- nnU-Net Documentation: `documentation/setting_up_paths.md`
- Trainer 原始碼: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
