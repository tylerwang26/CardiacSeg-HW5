# 🚀 RTX 4090 CUDA 遷移完整指南

## 📦 Step 1: 檔案準備（macOS - 已完成 ✅）

### Checkpoint 打包狀態
```bash
檔案：nnunet_checkpoint_epoch5.tar.gz
大小：1.0 GB
位置：/Users/tyler/Documents/GitHub/CardiacSeg/
```

### 包含內容
- ✅ Preprocessed 資料（3d_lowres 所有 .npy 檔案）
- ✅ Dataset 配置（dataset.json, nnUNetPlans.json, splits_final.json 等）
- ✅ 最佳 Checkpoint（checkpoint_best.pth - Epoch 4 後，EMA Dice 0.3302）
- ✅ 訓練日誌與 debug 資訊

### 當前訓練進度
- **已完成**：Epoch 0-4（5個epochs）
- **剩餘**：Epoch 5-80（75個epochs）
- **最佳指標**：
  - Best EMA Dice: 0.3302
  - 類別1 Dice: 0.888
  - 類別2 Dice: 0.646
  - 類別3 Dice: 0.000（尚未學習）

---

## ⚡ 加速效益分析

### 硬體對比

| 規格 | Apple Silicon MPS | RTX 4090 CUDA |
|------|------------------|---------------|
| **運算核心** | ~40 GPU cores (M1/M2 Max) | 16,384 CUDA cores + Tensor cores |
| **記憶體頻寬** | ~400 GB/s | 1,008 GB/s |
| **VRAM** | 統一記憶體 32-96 GB | 24 GB GDDR6X |
| **nnU-Net 優化** | 基本支援 | 原生最佳化 |

### 速度對比與預估

| 項目 | MPS | RTX 4090 | 加速比 |
|------|-----|----------|--------|
| **每 Epoch 時間** | 1h 45m (6,300s) | **26 分鐘 (1,560s)** | **4.0x** |
| **剩餘 75 epochs** | 131 小時 (5.5 天) | **33 小時 (1.4 天)** | **4.0x** |
| **總訓練時間** (80 epochs) | 140 小時 (5.8 天) | **35 小時 (1.5 天)** | **4.0x** |
| **節省時間** | - | **105 小時 (4.4 天)** | - |

### 里程碑時間表對比

| 里程碑 | MPS 預估完成 | RTX 4090 預估完成 | 節省時間 |
|--------|-------------|-----------------|---------|
| Epoch 10 | 18 小時 | **4.5 小時** | 13.5 小時 |
| Epoch 20 | 35 小時 (1.5 天) | **9 小時** | 26 小時 |
| Epoch 40 | 70 小時 (2.9 天) | **17.5 小時** | 52.5 小時 |
| Epoch 60 | 105 小時 (4.4 天) | **26 小時** | 79 小時 |
| **Epoch 80** | **140 小時 (5.8 天)** | **35 小時 (1.5 天)** | **105 小時** |

---

## 🔧 Step 2: 停止 macOS 訓練

### 2.1 檢查當前訓練狀態
```bash
ps -Ao pid,pcpu,pmem,etime,command | grep -Ei 'nnunet' | grep -v grep
```

### 2.2 優雅停止訓練
```bash
# 找到訓練進程 PID（例如 19223）
PID=$(pgrep -f "nnunetv2.run.run_training.*3d_lowres")

# 發送 SIGINT（優雅停止，會保存 checkpoint）
kill -SIGINT $PID

# 等待 10 秒
sleep 10

# 如果還在運行，強制停止
if ps -p $PID > /dev/null 2>&1; then
    kill -SIGKILL $PID
fi

echo "✅ 訓練已停止"
```

### 2.3 確認最終狀態
```bash
# 確認進程已結束
ps -p $PID 2>/dev/null || echo "進程已結束"

# 檢查最後的 checkpoint 時間
ls -lh nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_best.pth
```

---

## 📤 Step 3: 傳輸檔案到 RTX 4090 機器

### 3.1 使用 scp 傳輸（推薦）
```bash
# 在 macOS 上執行
cd /Users/tyler/Documents/GitHub/CardiacSeg

# 傳輸打包檔案（替換 user@rtx-machine 為實際位址）
scp nnunet_checkpoint_epoch5.tar.gz user@rtx-machine:/path/to/destination/

# 傳輸訓練腳本與監控腳本
scp nnunet_train.py monitor_and_stop.sh user@rtx-machine:/path/to/destination/
```

### 3.2 或使用 rsync（續傳支援）
```bash
rsync -avz --progress nnunet_checkpoint_epoch5.tar.gz user@rtx-machine:/path/to/destination/
```

### 3.3 或使用雲端硬碟/USB
```bash
# 複製到外接硬碟
cp nnunet_checkpoint_epoch5.tar.gz /Volumes/USB_DRIVE/

# 在 RTX 4090 機器上掛載並複製
```

---

## 🖥️ Step 4: RTX 4090 機器環境設定

### 4.1 確認 CUDA 環境
```bash
# 檢查 NVIDIA 驅動與 CUDA
nvidia-smi

# 應顯示類似：
# GPU: NVIDIA GeForce RTX 4090
# CUDA Version: 12.x
```

### 4.2 建立 Python 虛擬環境
```bash
# 建議使用 Python 3.10 或 3.11
python3.11 -m venv ~/nnunet_venv
source ~/nnunet_venv/bin/activate

# 確認 Python 版本
python --version  # 應顯示 Python 3.11.x
```

### 4.3 安裝 CUDA 版本的 PyTorch
```bash
# 安裝 PyTorch with CUDA 12.1（根據你的 CUDA 版本調整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 驗證 CUDA 可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 應輸出：
# CUDA available: True
# CUDA version: 12.1
# GPU: NVIDIA GeForce RTX 4090
```

### 4.4 安裝 nnU-Net v2
```bash
# 安裝 nnU-Net
pip install nnunetv2

# 或從源碼安裝（取得最新版本）
# git clone https://github.com/MIC-DKFZ/nnUNet.git
# cd nnUNet
# pip install -e .

# 確認安裝
nnUNetv2_train --version
```

### 4.5 設定 nnU-Net 環境變數
```bash
# 建立目錄
mkdir -p ~/nnunet_data/{raw,preprocessed,results}

# 設定環境變數（加入到 ~/.bashrc 或 ~/.zshrc）
export nnUNet_raw="$HOME/nnunet_data/raw"
export nnUNet_preprocessed="$HOME/nnunet_data/preprocessed"
export nnUNet_results="$HOME/nnunet_data/results"

# 立即生效
source ~/.bashrc  # 或 source ~/.zshrc
```

---

## 📂 Step 5: 解壓與配置 Checkpoint

### 5.1 解壓檔案
```bash
cd ~/nnunet_data

# 解壓 checkpoint（會自動建立目錄結構）
tar -xzf nnunet_checkpoint_epoch5.tar.gz

# 確認解壓成功
ls -lh nnUNet_preprocessed/Dataset001_CardiacSeg/
ls -lh nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/
```

### 5.2 驗證必要檔案存在
```bash
# 檢查 preprocessed 資料
ls nnUNet_preprocessed/Dataset001_CardiacSeg/nnUNetPlans_3d_lowres/*.npy | wc -l
# 應顯示 100 個檔案（50 training + 50 seg）

# 檢查 checkpoint
ls -lh nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_best.pth
# 應顯示 ~235 MB

# 檢查配置檔
cat nnUNet_preprocessed/Dataset001_CardiacSeg/nnUNetPlans.json | jq '.configurations."3d_lowres".batch_size'
# 應顯示 2（預設）
```

### 5.3 （可選）調整 batch size
```bash
# 編輯 nnUNetPlans.json，將 batch_size 從 2 增加到 4
cd nnUNet_preprocessed/Dataset001_CardiacSeg/
cp nnUNetPlans.json nnUNetPlans.json.bak  # 備份

# 使用 sed 或手動編輯
# 將 "batch_size": 2 改為 "batch_size": 4
nano nnUNetPlans.json  # 或使用 vim/code

# 注意：batch_size=4 需要約 16-18 GB VRAM，RTX 4090 24GB 足夠
# 若要更激進，可以試 batch_size=6（需監控記憶體使用）
```

---

## 🚀 Step 6: 啟動 RTX 4090 訓練

### 6.1 使用 nnU-Net 官方命令（推薦）
```bash
# 啟動虛擬環境
source ~/nnunet_venv/bin/activate

# 切換到工作目錄
cd ~/nnunet_data

# 從 checkpoint 繼續訓練
nnUNetv2_train 001 3d_lowres 0 \
  -p nnUNetPlans \
  -device cuda \
  --c

# 參數說明：
# 001: Dataset ID
# 3d_lowres: 配置名稱
# 0: fold 編號
# -p nnUNetPlans: planner 名稱
# -device cuda: 使用 CUDA（自動偵測 GPU）
# --c: 繼續訓練（會自動載入 checkpoint_best.pth）
```

### 6.2 或使用背景執行（推薦，可登出不中斷）
```bash
# 使用 nohup 背景執行
nohup nnUNetv2_train 001 3d_lowres 0 -p nnUNetPlans -device cuda --c > training_rtx4090.log 2>&1 &

# 記下 PID
echo $! > training_pid.txt

# 查看即時日誌
tail -f training_rtx4090.log
```

### 6.3 或使用 tmux/screen（推薦，可隨時重新連接）
```bash
# 安裝 tmux（如果沒有）
sudo apt install tmux  # Ubuntu/Debian
# 或 sudo yum install tmux  # CentOS/RHEL

# 建立 tmux session
tmux new -s nnunet_training

# 在 tmux 內啟動訓練
source ~/nnunet_venv/bin/activate
cd ~/nnunet_data
nnUNetv2_train 001 3d_lowres 0 -p nnUNetPlans -device cuda --c

# 離開 tmux（訓練繼續）：按 Ctrl+B 然後按 D
# 重新連接：tmux attach -t nnunet_training
```

---

## 📊 Step 7: 監控訓練進度

### 7.1 複製並設定監控腳本
```bash
# 確保監控腳本可執行
chmod +x monitor_and_stop.sh

# 啟動自動監控（每 15 分鐘檢查，目標 Dice 0.70）
nohup ./monitor_and_stop.sh Dataset001_CardiacSeg 3d_lowres 0.70 900 > monitor_auto.log 2>&1 &

# 參數說明：
# Dataset001_CardiacSeg: dataset 名稱
# 3d_lowres: 配置
# 0.70: 目標 EMA Dice（達到後自動停止）
# 900: 檢查間隔（秒，900=15分鐘）
```

### 7.2 手動查看進度
```bash
# 查看最新訓練日誌
tail -n 50 nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/training_log_*.txt

# 查看進度圖（需要 X11 forwarding 或下載到本機）
# 在本機（macOS）：
scp user@rtx-machine:~/nnunet_data/nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/progress.png ~/Desktop/
open ~/Desktop/progress.png

# 查看 GPU 使用率
watch -n 2 nvidia-smi
```

### 7.3 檢查訓練狀態
```bash
# 查看訓練進程
ps aux | grep nnUNetv2_train

# 查看 GPU 記憶體使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 查看當前 epoch 與 EMA Dice
grep -i "epoch\|best EMA" nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/training_log_*.txt | tail -n 20
```

---

## 📅 RTX 4090 訓練時程預估

### 假設從 Epoch 5 繼續（75 個 epochs 剩餘）

| 檢查時間點 | Epoch | 預估經過時間 | 預期 EMA Dice | 建議動作 |
|-----------|-------|------------|--------------|---------|
| **首次檢查** | 10 | +2 小時 | 0.40-0.45 | 確認 CUDA 運行正常，無錯誤 |
| **早期評估** | 20 | +6.5 小時 | 0.48-0.53 | 檢查 loss 下降趨勢，GPU 溫度 |
| **中期檢查** | 40 | +15 小時 | 0.58-0.63 | 評估是否接近目標，可能提前達標 |
| **目標檢查** | 50-60 | +20-23 小時 | **0.68-0.73** | 可能達到 0.70，監控腳本會自動停止 |
| **完整訓練** | 80 | +33 小時 | 0.75-0.80 | 獲得最佳模型 |

### 建議檢查頻率
- **前 10 epochs**：每 2 小時檢查（確保順利啟動）
- **Epoch 10-40**：每 4-6 小時檢查
- **Epoch 40+**：每 6-8 小時，或依賴自動監控

---

## ⚙️ 優化建議與進階配置

### 1. Batch Size 調整
```bash
# 預設 batch_size=2（保守，MPS 限制）
# RTX 4090 24GB VRAM 可以增加：

# 選項 A：batch_size=4（推薦，穩定）
# 預估記憶體：16-18 GB
# 速度提升：~10-15%

# 選項 B：batch_size=6（激進）
# 預估記憶體：22-23 GB
# 速度提升：~20-25%
# 風險：可能 OOM（記憶體不足）

# 編輯配置（在啟動訓練前）
nano nnUNet_preprocessed/Dataset001_CardiacSeg/nnUNetPlans.json
# 修改 "configurations" -> "3d_lowres" -> "batch_size": 4
```

### 2. 混合精度訓練（Automatic Mixed Precision）
```bash
# nnU-Net v2 預設已啟用 AMP
# 如果需要停用（不推薦）：
# 編輯訓練腳本或使用環境變數
export NNUNET_AMP=0  # 停用 AMP
```

### 3. 監控 GPU 溫度與功耗
```bash
# 即時監控（每 2 秒更新）
watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader'

# RTX 4090 正常範圍：
# 溫度：70-85°C
# 功耗：350-450W（滿載）
# GPU 使用率：95-100%
```

### 4. 多 GPU 訓練（如果有多張卡）
```bash
# nnU-Net v2 原生支援 DataParallel
# 使用多 GPU
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 001 3d_lowres 0 -p nnUNetPlans -device cuda --c

# 注意：
# - batch_size 會自動分配到各 GPU
# - 需要調整 batch_size 以充分利用（例如 batch_size=8 for 2 GPUs）
```

### 5. 提早停止策略
```bash
# 如果監控腳本顯示 EMA Dice 達到 0.70
# 可以手動停止訓練：
PID=$(cat training_pid.txt)
kill -SIGINT $PID

# 或等待自動停止（monitor_and_stop.sh 會自動處理）
```

---

## 🔍 故障排查

### 問題 1：CUDA Out of Memory (OOM)
```bash
# 症狀：RuntimeError: CUDA out of memory
# 解決方案：
# 1. 降低 batch size（從 4 → 2）
# 2. 檢查是否有其他程序使用 GPU：nvidia-smi
# 3. 重啟 Python 進程清除快取
```

### 問題 2：Checkpoint 無法載入
```bash
# 症狀：找不到 checkpoint 或載入失敗
# 確認檔案存在：
ls -lh nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_best.pth

# 確認路徑正確（nnU-Net 會自動搜尋）
# 如果還是失敗，可能需要從 Epoch 0 重新開始（會快很多）
```

### 問題 3：訓練速度沒有明顯提升
```bash
# 可能原因：
# 1. 使用了 CPU 而非 GPU
python -c "import torch; print(torch.cuda.is_available())"  # 應為 True

# 2. I/O 瓶頸（磁碟讀取慢）
# 解決：將資料複製到 NVMe SSD

# 3. batch_size 太小
# 解決：增加 batch_size 到 4 或 6
```

### 問題 4：SSH 斷線導致訓練中斷
```bash
# 預防方案：
# 使用 tmux 或 nohup（見 Step 6）

# 如果已中斷，檢查是否有 checkpoint：
ls -lt nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_*.pth

# 重新啟動（會從最後的 checkpoint 繼續）
nnUNetv2_train 001 3d_lowres 0 -p nnUNetPlans -device cuda --c
```

---

## ✅ 完成檢查清單

遷移前（macOS）：
- [ ] 確認訓練到 Epoch 4-5
- [ ] 建立 checkpoint 打包檔（nnunet_checkpoint_epoch5.tar.gz）
- [ ] 驗證打包檔完整性（1.0 GB）
- [ ] 停止 macOS 訓練進程
- [ ] 傳輸檔案到 RTX 4090 機器

RTX 4090 機器設定：
- [ ] 確認 CUDA 可用（nvidia-smi）
- [ ] 建立 Python 虛擬環境（Python 3.11）
- [ ] 安裝 PyTorch with CUDA
- [ ] 安裝 nnU-Net v2
- [ ] 設定 nnU-Net 環境變數
- [ ] 解壓 checkpoint 並驗證
- [ ] （可選）調整 batch size

啟動訓練：
- [ ] 使用 tmux/screen 或 nohup 背景執行
- [ ] 確認訓練從 Epoch 5 繼續（--c 參數）
- [ ] 啟動自動監控腳本
- [ ] 檢查 GPU 使用率（應接近 100%）
- [ ] 確認首個 epoch 時間（應 ~26 分鐘）

監控與優化：
- [ ] 每 2-4 小時檢查進度
- [ ] 監控 GPU 溫度（<85°C）
- [ ] 查看 loss 下降趨勢
- [ ] 在 EMA Dice 0.70 時評估是否停止

---

## 📞 需要協助？

如果遇到問題，提供以下資訊：
1. 錯誤訊息（完整 traceback）
2. `nvidia-smi` 輸出
3. Python 與 PyTorch 版本
4. 最新訓練日誌的最後 50 行
5. nnU-Net 版本：`pip show nnunetv2`

---

## 🎯 總結

### 遷移優勢
- ⚡ **4倍速度提升**：從 5.5 天 → 1.4 天
- 💰 **節省 98 小時**：提早 4 天完成
- 🎯 **更快迭代**：可以更快測試不同配置
- 📊 **更好硬體**：原生 CUDA 優化，更穩定

### 預期結果
- Epoch 10：約 2 小時後完成
- Epoch 40：約 15 小時後完成
- **達到 Dice 0.70**：預估 20-24 小時
- 完整 80 epochs：約 33 小時（1.4 天）

### 下一步
1. 傳輸 checkpoint 到 RTX 4090 機器
2. 按照本指南設定環境
3. 啟動訓練並監控
4. 在達到目標 Dice 時停止或繼續到 80 epochs

**祝訓練順利！🚀**
