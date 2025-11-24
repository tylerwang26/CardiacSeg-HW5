# nnU-Net 訓練使用指南（優化版）

## 🎯 已完成的修改

### 1. 停止了當前訓練
- ✅ 已溫和終止原本的長時間訓練（1000 epochs）
- ✅ Checkpoint 已保存在 `nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`

### 2. 修改了訓練腳本
- ✅ `nnunet_train.py` 現在支援自訂參數
- ✅ 預設 epochs 從 1000 降為 **250**（更合理的訓練時長）
- ✅ 可指定 fold、config、device 等參數

### 3. 新增提前停止監控
- ✅ `early_stopping_monitor.sh` 可自動偵測收斂並停止訓練
- ✅ 當驗證 Dice 分數 20 個 epochs 內無改善時自動停止

---

## 🚀 快速開始

### 方案 A：從頭開始訓練（推薦 - 使用 250 epochs）

```bash
# 使用預設設定（250 epochs, fold 0, 3d_fullres, mps）
python3 nnunet_train.py --skip-preprocess

# 或自訂參數
python3 nnunet_train.py --epochs 200 --fold 0 --config 3d_fullres --device mps --skip-preprocess
```

**預估時間**：250 epochs × 1.44 小時 = **約 15 天**（比原本 60 天快 4 倍）

### 方案 B：從之前的 checkpoint 繼續

```bash
# 繼續之前的訓練（會讀取 checkpoint_latest.pth）
python3 nnunet_train.py --continue-training --skip-preprocess --epochs 250
```

### 方案 C：訓練單一 fold（最快）

```bash
# 只訓練 fold 0（不做完整 5-fold 交叉驗證）
python3 nnunet_train.py --epochs 200 --fold 0 --skip-preprocess
```

**預估時間**：200 epochs × 1.44 小時 = **約 12 天**

---

## 🛡️ 使用提前停止監控（強烈推薦）

在另一個終端機視窗執行監控腳本，它會自動偵測收斂：

```bash
# 基本用法（預設：20 epochs 無改善就停止）
./early_stopping_monitor.sh

# 自訂參數
./early_stopping_monitor.sh Dataset001_CardiacSeg 3d_fullres fold_0 30 0.002 600
# 參數說明：資料集 配置 fold 容忍epochs 最小改善閾值 檢查間隔秒數
```

**建議配合使用**：
```bash
# 終端機 1：啟動訓練
python3 nnunet_train.py --epochs 300 --skip-preprocess

# 終端機 2：啟動提前停止監控
./early_stopping_monitor.sh
```

這樣即使設定 300 epochs，如果 Dice 分數不再提升，監控腳本會自動停止訓練。

---

## 📊 監控訓練進度

### 即時查看訓練日誌
```bash
tail -f nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt
```

### 使用監控腳本
```bash
./monitor_training.sh Dataset001_CardiacSeg 3d_fullres
```

### 查看當前 epoch 和最佳 Dice
```bash
grep -E "(Epoch [0-9]+|best EMA pseudo Dice)" nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt | tail -n 10
```

---

## 🔧 進階選項

### 所有可用參數

```bash
python3 nnunet_train.py --help
```

主要參數：
- `--epochs`: 訓練的 epoch 數（預設 250）
- `--fold`: 訓練哪個 fold（0-4 或 all，預設 0）
- `--config`: 配置類型（2d, 3d_fullres, 3d_lowres，預設 3d_fullres）
- `--device`: 使用的裝置（cuda, mps, cpu，預設 mps）
- `--continue-training`: 從 checkpoint 繼續
- `--skip-preprocess`: 跳過預處理（已完成時使用）

### 訓練所有 5 folds（完整交叉驗證）

```bash
python3 nnunet_train.py --epochs 250 --fold all --skip-preprocess
```

**注意**：這會訓練 5 個模型，總時間 = 15 天 × 5 = **75 天**

### 背景執行（nohup）

```bash
nohup python3 nnunet_train.py --epochs 250 --skip-preprocess > training.log 2>&1 &

# 同時啟動提前停止監控
nohup ./early_stopping_monitor.sh > early_stop.log 2>&1 &
```

---

## ⏹️ 停止訓練

### 溫和停止（保存 checkpoint）
```bash
pkill -SIGTERM -f "nnUNetv2_train"
```

### 強制停止
```bash
pkill -SIGKILL -f "nnUNetv2_train"
```

---

## 📈 訓練完成後

### 1. 查看訓練結果
```bash
ls -lh nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
```

關鍵檔案：
- `checkpoint_best.pth` - 最佳模型
- `checkpoint_latest.pth` - 最新模型
- `checkpoint_final.pth` - 最終模型

### 2. 執行推論
```bash
python3 nnunet_infer.py
```

### 3. 評估模型
```bash
python3 nnunet_evaluate.py
```

---

## 💡 建議策略

### 小規模測試（快速驗證）
```bash
python3 nnunet_train.py --epochs 50 --fold 0 --skip-preprocess
```
時間：約 3 天，用於確認設定正確

### 標準訓練（推薦）
```bash
# 終端機 1
python3 nnunet_train.py --epochs 250 --fold 0 --skip-preprocess

# 終端機 2
./early_stopping_monitor.sh
```
時間：實際可能 7-15 天（視收斂速度）

### 完整訓練（追求最佳效果）
```bash
python3 nnunet_train.py --epochs 300 --fold all --skip-preprocess
```
時間：約 60-90 天（但可用提前停止縮短）

---

## 🔍 故障排除

### Q: 訓練被中斷了怎麼辦？
A: 使用 `--continue-training` 從 checkpoint 繼續：
```bash
python3 nnunet_train.py --continue-training --skip-preprocess --epochs 250
```

### Q: 如何更改 epochs 數？
A: 直接修改 `--epochs` 參數，環境變數會自動設定

### Q: 提前停止腳本沒有作用？
A: 確認：
1. 訓練進程正在運行：`pgrep -f nnUNetv2_train`
2. 日誌檔存在且更新中
3. 檢查監控腳本的日誌輸出

### Q: 想要更激進的提前停止？
A: 調整參數（例如 10 epochs 無改善就停止）：
```bash
./early_stopping_monitor.sh Dataset001_CardiacSeg 3d_fullres fold_0 10
```

---

## 📝 範例：從現在開始的推薦流程

```bash
# 1. 啟動新的訓練（250 epochs）
python3 nnunet_train.py --epochs 250 --fold 0 --skip-preprocess &

# 2. 在另一個終端啟動提前停止監控
./early_stopping_monitor.sh &

# 3. 監控進度
./monitor_training.sh Dataset001_CardiacSeg 3d_fullres

# 4. 等待訓練完成或自動停止...

# 5. 訓練完成後執行推論和評估
python3 nnunet_infer.py
python3 nnunet_evaluate.py
```

預估總時間：**7-15 天**（視提前停止時機而定）

---

**提示**：所有腳本都支援在背景執行，不會因為關閉終端機而中斷！
