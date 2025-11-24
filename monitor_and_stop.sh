#!/bin/bash

################################################################################
# monitor_and_stop.sh
# 自動監控 nnU-Net 訓練進度，達到目標 Dice 時自動停止訓練
# 
# 用法:
#   ./monitor_and_stop.sh [DATASET] [CONFIG] [TARGET_DICE] [CHECK_INTERVAL]
#
# 參數:
#   DATASET         - Dataset 目錄名稱，預設: Dataset001_CardiacSeg
#   CONFIG          - 配置名稱，預設: 3d_lowres
#   TARGET_DICE     - 目標 EMA Pseudo Dice 門檻（0-1），預設: 0.70
#   CHECK_INTERVAL  - 檢查間隔（秒），預設: 1800（30分鐘）
#
# 範例:
#   ./monitor_and_stop.sh Dataset001_CardiacSeg 3d_lowres 0.75 1800
#   ./monitor_and_stop.sh  # 使用所有預設值
################################################################################

set -euo pipefail

# 預設值
DATASET_DIR_NAME="${1:-Dataset001_CardiacSeg}"
CFG="${2:-3d_lowres}"
TARGET_DICE="${3:-0.70}"
CHECK_INTERVAL="${4:-1800}"  # 30 分鐘

# 結果目錄
RESULTS_BASE="nnUNet_results/${DATASET_DIR_NAME}"
MONITOR_LOG="monitor_auto_stop_$(date +%Y%m%d_%H%M%S).log"

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo -e "${msg}" | tee -a "${MONITOR_LOG}"
}

log_color() {
    local color="$1"
    shift
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo -e "${color}${msg}${NC}" | tee -a "${MONITOR_LOG}"
}

# 找到實驗資料夾
find_experiment_dir() {
    local exp_dirs=("${RESULTS_BASE}"/nnUNetTrainer__nnUNetPlans__"${CFG}")
    if [[ -d "${exp_dirs[0]}" ]]; then
        echo "${exp_dirs[0]}"
    else
        return 1
    fi
}

# 找到最新的 training_log
find_latest_log() {
    local exp_dir="$1"
    local fold_dirs=("${exp_dir}"/fold_*)
    
    local latest_log=""
    local latest_time=0
    
    for fold in "${fold_dirs[@]}"; do
        if [[ -d "$fold" ]]; then
            for log_file in "${fold}"/training_log_*.txt; do
                if [[ -f "$log_file" ]]; then
                    local mtime
                    mtime=$(stat -f %m "$log_file" 2>/dev/null || stat -c %Y "$log_file" 2>/dev/null)
                    if (( mtime > latest_time )); then
                        latest_time=$mtime
                        latest_log="$log_file"
                    fi
                fi
            done
        fi
    done
    
    if [[ -n "$latest_log" ]]; then
        echo "$latest_log"
    else
        return 1
    fi
}

# 解析最新的 EMA Pseudo Dice
parse_latest_ema_dice() {
    local log_file="$1"
    
    # 找到最後一個 "New best EMA pseudo Dice" 行
    local ema_line
    ema_line=$(grep -i "best EMA pseudo Dice" "$log_file" | tail -n1 || echo "")
    
    if [[ -n "$ema_line" ]]; then
        # 提取數字：例如 "Yayy! New best EMA pseudo Dice: 0.3100999891757965"
        local ema_value
        ema_value=$(echo "$ema_line" | grep -oE '[0-9]+\.[0-9]+' | tail -n1)
        echo "$ema_value"
    else
        echo ""
    fi
}

# 解析最新 epoch 的三類 Pseudo dice
parse_latest_pseudo_dice() {
    local log_file="$1"
    
    # 找到最後一個 "Pseudo dice" 行
    local dice_line
    dice_line=$(grep -i "Pseudo dice" "$log_file" | tail -n1 || echo "")
    
    if [[ -n "$dice_line" ]]; then
        # 提取陣列內容：例如 "Pseudo dice [np.float32(0.8721), np.float32(0.6091), np.float32(0.0)]"
        echo "$dice_line" | grep -oE '\[.*\]' | sed 's/np\.float32//g' | tr -d '()[],'
    else
        echo ""
    fi
}

# 解析最新 epoch 編號
parse_latest_epoch() {
    local log_file="$1"
    
    local epoch_line
    epoch_line=$(grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}.*Epoch [0-9]+" "$log_file" | tail -n1 || echo "")
    
    if [[ -n "$epoch_line" ]]; then
        echo "$epoch_line" | grep -oE 'Epoch [0-9]+' | grep -oE '[0-9]+'
    else
        echo ""
    fi
}

# 找到訓練進程 PID
find_training_pid() {
    # 找 nnUNetv2_train 或 run_training 相關進程
    pgrep -f "nnunetv2.run.run_training.*${CFG}" || pgrep -f "nnunet_train.py.*${CFG}" || echo ""
}

# 停止訓練
stop_training() {
    local pid="$1"
    log_color "${RED}" "🛑 正在停止訓練進程 PID: ${pid}..."
    
    # 優雅停止（SIGINT）
    kill -SIGINT "$pid" 2>/dev/null || true
    sleep 5
    
    # 檢查是否還在運行
    if ps -p "$pid" > /dev/null 2>&1; then
        log_color "${YELLOW}" "⚠️  進程未響應 SIGINT，發送 SIGTERM..."
        kill -SIGTERM "$pid" 2>/dev/null || true
        sleep 3
    fi
    
    # 最後強制停止
    if ps -p "$pid" > /dev/null 2>&1; then
        log_color "${RED}" "⚠️  強制停止進程 SIGKILL..."
        kill -SIGKILL "$pid" 2>/dev/null || true
    fi
    
    log_color "${GREEN}" "✅ 訓練已停止"
}

################################################################################
# 主監控迴圈
################################################################################

log_color "${BLUE}" "=========================================="
log_color "${BLUE}" "🚀 開始自動監控 nnU-Net 訓練"
log_color "${BLUE}" "=========================================="
log "Dataset: ${DATASET_DIR_NAME}"
log "Config: ${CFG}"
log "目標 EMA Dice: ${TARGET_DICE}"
log "檢查間隔: ${CHECK_INTERVAL} 秒 ($(( CHECK_INTERVAL / 60 )) 分鐘)"
log "監控日誌: ${MONITOR_LOG}"
log "=========================================="

# 找到實驗目錄
EXP_DIR=$(find_experiment_dir)
if [[ -z "$EXP_DIR" ]]; then
    log_color "${RED}" "❌ 找不到實驗目錄: ${RESULTS_BASE}/nnUNetTrainer__nnUNetPlans__${CFG}"
    exit 1
fi

log_color "${GREEN}" "✅ 找到實驗目錄: ${EXP_DIR}"

CHECK_COUNT=0

while true; do
    CHECK_COUNT=$((CHECK_COUNT + 1))
    log ""
    log_color "${BLUE}" "==================== 檢查 #${CHECK_COUNT} ===================="
    
    # 找到最新日誌
    LATEST_LOG=$(find_latest_log "$EXP_DIR" || echo "")
    if [[ -z "$LATEST_LOG" ]]; then
        log_color "${YELLOW}" "⚠️  找不到訓練日誌，等待下一次檢查..."
        sleep "$CHECK_INTERVAL"
        continue
    fi
    
    log "最新日誌: $(basename "$LATEST_LOG")"
    
    # 檢查訓練進程
    TRAINING_PID=$(find_training_pid)
    if [[ -z "$TRAINING_PID" ]]; then
        log_color "${YELLOW}" "⚠️  未偵測到訓練進程，訓練可能已完成或停止"
        log "監控結束"
        exit 0
    fi
    
    log "訓練進程 PID: ${TRAINING_PID}"
    
    # 解析最新指標
    LATEST_EPOCH=$(parse_latest_epoch "$LATEST_LOG")
    LATEST_EMA=$(parse_latest_ema_dice "$LATEST_LOG")
    LATEST_DICE=$(parse_latest_pseudo_dice "$LATEST_LOG")
    
    if [[ -n "$LATEST_EPOCH" ]]; then
        log "最新 Epoch: ${LATEST_EPOCH}"
    fi
    
    if [[ -n "$LATEST_DICE" ]]; then
        log "最新 Pseudo Dice (3 類): ${LATEST_DICE}"
    fi
    
    if [[ -n "$LATEST_EMA" ]]; then
        log_color "${GREEN}" "📊 當前 Best EMA Dice: ${LATEST_EMA}"
        
        # 比較目標門檻（使用 bc 進行浮點數比較）
        if command -v bc &>/dev/null; then
            REACHED=$(echo "${LATEST_EMA} >= ${TARGET_DICE}" | bc -l)
        else
            # 如果沒有 bc，使用 awk
            REACHED=$(awk -v ema="$LATEST_EMA" -v target="$TARGET_DICE" 'BEGIN { print (ema >= target) ? 1 : 0 }')
        fi
        
        if [[ "$REACHED" == "1" ]]; then
            log_color "${GREEN}" "🎉 達到目標 Dice 門檻！"
            log_color "${GREEN}" "   當前: ${LATEST_EMA} >= 目標: ${TARGET_DICE}"
            stop_training "$TRAINING_PID"
            log_color "${BLUE}" "=========================================="
            log_color "${BLUE}" "監控完成，已自動停止訓練"
            log_color "${BLUE}" "=========================================="
            exit 0
        else
            log "距離目標: $(awk -v target="$TARGET_DICE" -v ema="$LATEST_EMA" 'BEGIN { printf "%.4f", target - ema }')"
        fi
    else
        log_color "${YELLOW}" "⚠️  尚未找到 EMA Dice 記錄（可能訓練剛開始）"
    fi
    
    # 等待下一次檢查
    log "下次檢查時間: $(date -v +${CHECK_INTERVAL}S '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -d "+${CHECK_INTERVAL} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$(( CHECK_INTERVAL / 60 )) 分鐘後")"
    sleep "$CHECK_INTERVAL"
done
