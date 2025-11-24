#!/usr/bin/env bash
set -euo pipefail

# 排程腳本：在訓練穩定後自動驗證預處理完整性與補完必要檔案
# 用途：避免與正在運行的訓練爭搶資源，擇期檢查並修復 preprocessing 中斷

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/scheduled_verify_$(date +%Y%m%d_%H%M%S).log"

echo "======================================"
echo "  排程驗證預處理完整性"
echo "======================================"
echo "開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "日誌檔案: $LOG_FILE"
echo

# 載入環境變數
if [[ -f ".env.nnunet.sh" ]]; then
  source ".env.nnunet.sh"
  export NNUNET_RAW="$nnUNet_raw"
  export NNUNET_PREPROCESSED="$nnUNet_preprocessed"
  export NNUNET_RESULTS="$nnUNet_results"
  echo "✓ 已載入環境變數"
else
  echo "⚠ 未找到 .env.nnunet.sh，使用預設路徑"
  export NNUNET_RAW="$SCRIPT_DIR/nnUNet_raw"
  export NNUNET_PREPROCESSED="$SCRIPT_DIR/nnUNet_preprocessed"
  export NNUNET_RESULTS="$SCRIPT_DIR/nnUNet_results"
fi

# 等待訓練進程穩定或完成（可選：檢查是否在跑，若在跑則稍候）
WAIT_IF_TRAINING=${WAIT_IF_TRAINING:-true}
if [[ "$WAIT_IF_TRAINING" == "true" ]]; then
  while pgrep -f "nnUNetv2_train" >/dev/null 2>&1; do
    echo "偵測到訓練進程運行中，等待 5 分鐘後重試..."
    sleep 300  # 5分鐘
  done
  echo "✓ 訓練進程已結束或不存在，開始驗證"
fi

echo
echo "執行驗證與補完..."
echo "指令：nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity"
echo

# 僅做驗證（不強制重新規劃）
./.venv/bin/nnUNetv2_plan_and_preprocess \
  -d 001 \
  --verify_dataset_integrity \
  2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo
echo "======================================"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓ 驗證完成（退出碼 0）"
  echo "  若看到 'verify_dataset_integrity Done' 且無錯誤，表示預處理完整。"
else
  echo "⚠ 驗證過程退出碼 $EXIT_CODE"
  echo "  請檢查日誌：$LOG_FILE"
  echo "  可能需要重新執行預處理或檢查資料集結構。"
fi
echo "結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

exit $EXIT_CODE
