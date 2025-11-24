#!/usr/bin/env bash
set -euo pipefail

# 簡介：
# - 自動偵測 nnUNet 結果目錄與最新的實驗/折數資料夾
# - 不強制需要外部 log；若有 *.log 會自動 tail，否則顯示檔案活動與 checkpoint 狀態
# - 顯示訓練進程狀態、每個 fold 的最近活動、以及關鍵檔案

REFRESH_INTERVAL=${REFRESH_INTERVAL:-15}
DATASET_DIR_NAME=${1:-Dataset001_CardiacSeg}
CFG=${2:-3d_fullres}  # 例如 3d_fullres / 3d_lowres / 2d

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 載入 .env.nnunet.sh（若存在）
if [[ -f "$SCRIPT_DIR/.env.nnunet.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env.nnunet.sh"
fi

# 解析環境變數（兼容大小寫）
RESULTS_ROOT=${NNUNET_RESULTS:-${nnUNet_results:-"$SCRIPT_DIR/nnUNet_results"}}

DATASET_RESULTS_DIR="$RESULTS_ROOT/$DATASET_DIR_NAME"
if [[ ! -d "$DATASET_RESULTS_DIR" ]]; then
  echo "找不到結果目錄：$DATASET_RESULTS_DIR"
  echo "請確認 .env.nnunet.sh 或執行過訓練。"
  exit 1
fi

echo "======================================"
echo "  nnU-Net 訓練即時監控"
echo "======================================"
echo "資料集: $DATASET_DIR_NAME"
echo "配置: $CFG"
echo "RESULTS: $DATASET_RESULTS_DIR"
echo "按 Ctrl+C 停止監控"
echo

while true; do
  clear
  echo "--------------------------------------"
  echo "更新時間: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "--------------------------------------"

  # 訓練進程狀態（尋找包含 config 的命令列）
  tpid=$(pgrep -f "nnUNetv2_train .* $CFG" || true)
  if [[ -n "${tpid:-}" ]]; then
    echo "✓ 訓練進程運行中"
    ps -o pid,pcpu,pmem,etime,command -p "$tpid" | sed -n '1p;$p'
  else
    echo "✗ 未偵測到訓練進程（可能已完成或尚未開始）"
  fi

  echo
  echo "📁 掃描實驗目錄（包含 $CFG）..."
  mapfile -t exps < <(find "$DATASET_RESULTS_DIR" -maxdepth 1 -type d -name "*__*" | sort)
  if [[ ${#exps[@]} -eq 0 ]]; then
    echo "尚未建立任何訓練實驗目錄。"
  fi

  for exp in "${exps[@]}"; do
    # 僅顯示與 config 相符的實驗（例如 ...__nnUNetPlans__3d_fullres）
    if [[ "$(basename "$exp")" != *"${CFG}"* ]]; then
      continue
    fi
    echo
    echo "🧪 實驗: $(basename "$exp")"
    mapfile -t folds < <(find "$exp" -maxdepth 1 -type d -name 'fold_*' | sort)
    if [[ ${#folds[@]} -eq 0 ]]; then
      echo "  (尚未建立 fold 目錄)"
      continue
    fi
    for fd in "${folds[@]}"; do
      echo "  ▶ $(basename "$fd")"
      best=$(ls "$fd" 2>/dev/null | grep -c '^checkpoint_best\.pth$' || true)
      latest=$(ls "$fd" 2>/dev/null | grep -c '^checkpoint_latest\.pth$' || true)
      echo "    檢查點: best=$best, latest=$latest"
      echo "    最近更新檔："
      ls -lt "$fd" 2>/dev/null | head -n 5 | sed 's/^/      - /' || echo "      (無檔案)"

      # 自動尋找可讀的 log 檔
      log=$(ls "$fd"/*.log "$fd"/*.txt 2>/dev/null | head -n1 || true)
      if [[ -n "${log:-}" ]]; then
        echo "    📝 Log: $(basename "$log")（最後 12 行）"
        tail -n 12 "$log" | sed 's/^/      /'
      else
        echo "    (未找到 log；改列出最近活動清單如上)"
      fi
    done
  done

  if [[ -z "${tpid:-}" ]]; then
    echo
    echo "監控結束（無訓練進程）。"
    exit 0
  fi

  echo
  echo "下次更新: ${REFRESH_INTERVAL} 秒後..."
  sleep "$REFRESH_INTERVAL"
done
