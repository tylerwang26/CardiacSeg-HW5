#!/bin/bash
set -euo pipefail

RESULTS_BASE="nnUNet_results/Dataset001_CardiacSeg"
CFG="3d_lowres"

find_latest_log() {
    local exp_dir="${RESULTS_BASE}/nnUNetTrainer__nnUNetPlans__${CFG}"
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
    
    echo "$latest_log"
}

LATEST_LOG=$(find_latest_log)
echo "📁 最新日誌: $LATEST_LOG"
echo ""

LATEST_EPOCH=$(grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}.*Epoch [0-9]+" "$LATEST_LOG" | tail -n1 | grep -oE 'Epoch [0-9]+' | grep -oE '[0-9]+' || echo "N/A")
echo "📊 最新 Epoch: $LATEST_EPOCH"

LATEST_EMA=$(grep -i "best EMA pseudo Dice" "$LATEST_LOG" | tail -n1 | grep -oE '[0-9]+\.[0-9]+' | tail -n1 || echo "N/A")
echo "🎯 當前 Best EMA Dice: $LATEST_EMA"

LATEST_DICE=$(grep -i "Pseudo dice" "$LATEST_LOG" | tail -n1 | grep -oE '\[.*\]' | sed 's/np\.float32//g' | tr -d '()[],' || echo "N/A")
echo "📈 最新 Pseudo Dice (3 類): $LATEST_DICE"

TRAINING_PID=$(pgrep -f "nnunetv2.run.run_training.*${CFG}" || echo "")
if [[ -n "$TRAINING_PID" ]]; then
    echo "✅ 訓練進程運行中，PID: $TRAINING_PID"
else
    echo "⚠️  未偵測到訓練進程"
fi
