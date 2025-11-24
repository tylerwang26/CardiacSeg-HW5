#!/usr/bin/env bash
set -euo pipefail

# 提前停止監控腳本：自動偵測訓練收斂並溫和終止
# 當驗證 Dice 分數在指定 epochs 內無明顯提升時，自動停止訓練

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 載入環境變數
if [[ -f ".env.nnunet.sh" ]]; then
  source ".env.nnunet.sh"
  export NNUNET_RESULTS="$nnUNet_results"
else
  export NNUNET_RESULTS="$SCRIPT_DIR/nnUNet_results"
fi

# 配置參數
DATASET=${1:-Dataset001_CardiacSeg}
CONFIG=${2:-3d_fullres}
FOLD=${3:-fold_0}
PATIENCE=${4:-20}  # 容忍無進步的 epoch 數
MIN_IMPROVEMENT=${5:-0.001}  # 最小改善閾值
CHECK_INTERVAL=${6:-300}  # 檢查間隔（秒），預設 5 分鐘

RESULTS_DIR="$NNUNET_RESULTS/$DATASET"
LOG_FILE="early_stopping_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "======================================"
echo "  提前停止監控"
echo "======================================"
echo "資料集: $DATASET"
echo "配置: $CONFIG"
echo "Fold: $FOLD"
echo "容忍度: $PATIENCE epochs"
echo "最小改善: $MIN_IMPROVEMENT"
echo "檢查間隔: $CHECK_INTERVAL 秒"
echo "日誌: $LOG_FILE"
echo "======================================"
echo

# 找到對應的訓練目錄
TRAIN_DIR=$(find "$RESULTS_DIR" -type d -name "*${CONFIG}*" | head -n1)
if [[ -z "$TRAIN_DIR" ]]; then
  echo "❌ 找不到訓練目錄：$RESULTS_DIR/*${CONFIG}*"
  exit 1
fi

FOLD_DIR="$TRAIN_DIR/$FOLD"
if [[ ! -d "$FOLD_DIR" ]]; then
  echo "❌ 找不到 fold 目錄：$FOLD_DIR"
  exit 1
fi

echo "✓ 監控目錄：$FOLD_DIR"
echo

TRAIN_LOG=$(find "$FOLD_DIR" -name "training_log_*.txt" | head -n1)
if [[ -z "$TRAIN_LOG" || ! -f "$TRAIN_LOG" ]]; then
  echo "⚠️  尚未找到訓練日誌，等待訓練開始..."
  sleep 30
  TRAIN_LOG=$(find "$FOLD_DIR" -name "training_log_*.txt" | head -n1)
fi

best_dice=0
epochs_without_improvement=0
last_epoch=-1

while true; do
  # 檢查訓練進程是否還在運行
  if ! pgrep -f "nnUNetv2_train.*${CONFIG}" >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 訓練進程已結束" | tee -a "$LOG_FILE"
    break
  fi

  # 從日誌提取最新的 epoch 和 Dice 分數
  if [[ -f "$TRAIN_LOG" ]]; then
    # 提取最新的 epoch 號碼
    current_epoch=$(grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}.*Epoch [0-9]+" "$TRAIN_LOG" | tail -n1 | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+")
    
    # 提取最新的 EMA pseudo Dice（nnUNet 的主要驗證指標）
    current_dice=$(grep "best EMA pseudo Dice" "$TRAIN_LOG" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" || echo "0")
    
    if [[ -n "$current_epoch" && "$current_epoch" != "$last_epoch" ]]; then
      last_epoch=$current_epoch
      improvement=$(python3 -c "print(${current_dice} - ${best_dice})")
      
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Epoch $current_epoch: Dice=$current_dice (最佳=$best_dice, 改善=$improvement)" | tee -a "$LOG_FILE"
      
      # 檢查是否有改善
      is_better=$(python3 -c "print(1 if ${current_dice} > ${best_dice} + ${MIN_IMPROVEMENT} else 0)")
      
      if [[ "$is_better" == "1" ]]; then
        best_dice=$current_dice
        epochs_without_improvement=0
        echo "  ✓ 有改善！重置計數器" | tee -a "$LOG_FILE"
      else
        epochs_without_improvement=$((epochs_without_improvement + 1))
        echo "  ⏳ 無明顯改善（${epochs_without_improvement}/${PATIENCE}）" | tee -a "$LOG_FILE"
      fi
      
      # 檢查是否達到提前停止條件
      if [[ $epochs_without_improvement -ge $PATIENCE ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 🛑 達到提前停止條件（${PATIENCE} epochs 無改善）" | tee -a "$LOG_FILE"
        echo "最佳 Dice: $best_dice" | tee -a "$LOG_FILE"
        echo "發送終止信號..." | tee -a "$LOG_FILE"
        
        # 溫和終止訓練
        pkill -SIGTERM -f "nnUNetv2_train.*${CONFIG}"
        
        echo "✓ 已發送終止信號，訓練將在當前 epoch 完成後停止" | tee -a "$LOG_FILE"
        break
      fi
    fi
  fi
  
  sleep "$CHECK_INTERVAL"
done

echo
echo "======================================"
echo "監控結束"
echo "最終最佳 Dice: $best_dice"
echo "完整日誌: $LOG_FILE"
echo "======================================"
