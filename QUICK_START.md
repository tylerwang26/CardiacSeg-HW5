# 快速開始指南（已修正環境）

## 問題總結

原始錯誤來自：
1. **環境變數未正確設定**：nnU-Net 同時檢查大小寫變數 (`nnUNet_*` / `NNUNET_*`)，訓練腳本只設定大寫版本。
2. **Python 版本不相容**：
   - Python 3.13 的 `os.environ` 對類型檢查更嚴格，nnU-Net 2.6.2 中有一行程式碼 `os.environ[...] = 1`（應為字串）導致 `TypeError`。
   - 系統 Python 缺少 `blosc2` 套件。
3. **命令調用方式**：直接使用 `nnUNetv2_train` 會誤取 PATH 中的系統安裝版本，而非虛擬環境版本。

---

## 已修正內容

### 1. 訓練腳本 (`nnunet_train.py`)
- ✅ 同時設定大小寫環境變數（`nnUNet_raw` + `NNUNET_RAW` 等）。
- ✅ 改用模組化呼叫 (`python -m nnunetv2.run.run_training`)，避免 PATH 問題。
- ✅ 加入檢查：若未使用虛擬環境會發出警告。
- ✅ 自動安裝缺失套件（`blosc2`、`nnunetv2`）。

### 2. 新建 Python 3.11 虛擬環境 (`.venv311`)
- ✅ 避免 Python 3.13 環境變數類型檢查問題。
- ✅ 完整安裝：torch, torchvision, nnunetv2, blosc2, monai, nibabel, SimpleITK, scikit-image, tqdm。

### 3. 修補 nnU-Net 套件 Bug
- ✅ 修改已安裝套件中的 `run_training.py` line 276：
  ```python
  # 原始（會報錯）:
  os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
  # 已修正為:
  os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = str(1)
  ```

---

## 正確使用方式

### 啟動訓練（使用 Python 3.11 虛擬環境）

```bash
# 1. 確保已載入環境變數（非必須，因為腳本會自動設定）
source .env.nnunet.sh

# 2. 使用新建的 Python 3.11 虛擬環境執行訓練
./.venv311/bin/python nnunet_train.py --epochs 80 --fold 0 --config 3d_fullres --skip-preprocess
```

**參數說明**：
- `--epochs 80`：訓練 80 個 epochs（預設 250，可根據時間調整）。
- `--fold 0`：訓練第 0 個交叉驗證 Fold（0-4 或 `all`）。
- `--config 3d_fullres`：使用 3D 全解析度配置（可選 `2d` / `3d_lowres` / `3d_fullres`）。
- `--skip-preprocess`：跳過預處理（假設已完成），直接訓練。

---

### 其他配置範例

```bash
# 快速 2D baseline（約 30-60 分鐘）
./.venv311/bin/python nnunet_train.py --config 2d --epochs 80 --fold 0 --skip-preprocess

# 3D 低解析度（較平衡，約 2-4 小時）
./.venv311/bin/python nnunet_train.py --config 3d_lowres --epochs 120 --fold 0 --skip-preprocess

# 繼續訓練（從最新 checkpoint）
./.venv311/bin/python nnunet_train.py --config 3d_fullres --epochs 250 --fold 0 --continue-training --skip-preprocess
```

---

### 監控訓練進度

#### 方式 1：監控腳本（即時顯示）
```bash
bash monitor_training.sh
```

#### 方式 2：查看訓練 log
```bash
tail -f nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log*.txt
```

#### 方式 3：早停監控（自動終止）
```bash
# 在背景執行早停監控（檢測連續 15 epochs 無改善則終止）
nohup bash early_stopping_monitor.sh > early_stop.log 2>&1 &
```

---

### 推論與評估

```bash
# 推論（需先訓練完成）
./.venv311/bin/python nnunet_infer.py

# 評估（計算 Dice/IoU）
./.venv311/bin/python nnunet_evaluate.py
```

---

## 常見問題

### Q1: 我可以用原本的 `.venv` (Python 3.13) 嗎？
**A**: 不建議。Python 3.13 與 nnU-Net 2.6.2 有型別相容問題。請使用 `.venv311`。

### Q2: 如何確認我在用正確的虛擬環境？
**A**: 執行時腳本會檢查，若 Python 路徑不含 `.venv` 會顯示警告：
```
⚠️  目前使用的 Python 並非專案虛擬環境 (.venv)。建議先執行: source .venv/bin/activate 或使用 ./.venv/bin/python。
```

### Q3: 訓練太慢怎麼辦？
**A**: 參考 README 中的「⚡ 快速訓練與時間策略」：
- 先用 `2d` 或 `3d_lowres` 建立 baseline。
- 減少 epochs（例如 80-120），啟用 early stopping。
- 確認使用 `--fold 0`（單一 Fold）而非 `all`。

### Q4: 如何停止訓練？
**A**: 
```bash
# 找到訓練進程
ps aux | grep nnunet_train

# 優雅終止（會儲存最新 checkpoint）
kill -SIGTERM <PID>

# 強制終止（不建議）
kill -9 <PID>
```

### Q5: 需要重新預處理嗎？
**A**: 若已執行過 `nnUNetv2_plan_and_preprocess` 且 `nnUNet_preprocessed/Dataset001_CardiacSeg/` 存在，可用 `--skip-preprocess` 跳過。

---

## 時間預估（Apple Silicon M2/M3）

| 配置 | Epochs | 預估時間 | Dice 預期 |
|------|--------|----------|-----------|
| 2d | 80 | 30-60 分鐘 | 0.5-0.7 |
| 3d_lowres | 120 | 2-4 小時 | 0.65-0.80 |
| 3d_fullres | 250 | 8-16 小時 | 0.75-0.90 |

> 首個 epoch 通常較慢（資料載入與 cache），後續會加速。

---

## 建議流程

1. **第一步：2D 快速驗證**（30-60 分鐘）
   ```bash
   ./.venv311/bin/python nnunet_train.py --config 2d --epochs 80 --fold 0 --skip-preprocess
   ```

2. **第二步：3D 低解析度 baseline**（2-4 小時）
   ```bash
   ./.venv311/bin/python nnunet_train.py --config 3d_lowres --epochs 120 --fold 0 --skip-preprocess
   ```

3. **第三步：3D 全解析度精緻化**（8-16 小時，可啟用 early stopping）
   ```bash
   ./.venv311/bin/python nnunet_train.py --config 3d_fullres --epochs 250 --fold 0 --skip-preprocess
   ```

---

## 疑難排解

### 錯誤：`ModuleNotFoundError: No module named 'blosc2'`
**解決**：確認使用 `.venv311/bin/python` 而非系統 `python3`。

### 錯誤：`TypeError: str expected, not int`
**解決**：已於 `.venv311` 中修補，確認使用該環境。

### 錯誤：`nnUNet_raw is not defined`
**解決**：訓練腳本會自動設定，或手動執行 `source .env.nnunet.sh`。

### 訓練卡住不動
**檢查**：
```bash
# 查看 GPU/MPS 使用率
top -o mem

# 查看訓練 log 最後幾行
tail -20 nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log*.txt
```

---

## 完整工作流程（從零開始）

```bash
# 1. 準備資料（若尚未執行）
./.venv311/bin/python rename_dataset.py

# 2. 預處理（首次執行）
./.venv311/bin/python nnunet_train.py --config 3d_fullres --fold 0 --epochs 1
# （會自動執行預處理，然後訓練 1 epoch 確認無誤後可中斷）

# 3. 正式訓練（使用建議的配置）
./.venv311/bin/python nnunet_train.py --config 3d_fullres --epochs 80 --fold 0 --skip-preprocess

# 4. 推論
./.venv311/bin/python nnunet_infer.py

# 5. 評估
./.venv311/bin/python nnunet_evaluate.py
```

---

## 聯絡資訊

若有其他問題，請參考：
- **訓練指南**：`TRAINING_GUIDE.md`
- **專案說明**：`README.md`（已包含「⚡ 快速訓練與時間策略」章節）
- **nnU-Net 官方文件**：https://github.com/MIC-DKFZ/nnUNet

---

**最後更新**：2025-11-15（已驗證 Python 3.11 + nnU-Net 2.6.2 + Apple Silicon MPS）
