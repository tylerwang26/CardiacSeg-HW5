#!/usr/bin/env bash
set -euo pipefail

# 簡介：
# - 不再依賴外部 log 檔；改以目錄與檔案數量估算 nnUNet 預處理進度。
# - 自動偵測環境變數（優先 NNUNET_*，其次 nnUNet_*）。
# - 顯示每個 plans 目錄（例如 nnUNetPlans_2d、3d_fullres、3d_lowres）的處理比例與最新檔案。

REFRESH_INTERVAL=${REFRESH_INTERVAL:-10}  # 預設每 10 秒更新
DATASET_DIR_NAME=${1:-Dataset001_CardiacSeg}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 載入 .env.nnunet.sh（若存在）
if [[ -f "$SCRIPT_DIR/.env.nnunet.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.nnunet.sh"
fi

# 解析環境變數（兼容大小寫）
RAW_ROOT=${NNUNET_RAW:-${nnUNet_raw:-"$SCRIPT_DIR/nnUNet_raw"}}
PRE_ROOT=${NNUNET_PREPROCESSED:-${nnUNet_preprocessed:-"$SCRIPT_DIR/nnUNet_preprocessed"}}

RAW_DATASET="$RAW_ROOT/$DATASET_DIR_NAME"
PRE_DATASET="$PRE_ROOT/$DATASET_DIR_NAME"

if [[ ! -d "$RAW_DATASET" ]]; then
    echo "找不到原始資料夾：$RAW_DATASET"
    echo "請確認 .env.nnunet.sh 的路徑或參數（第一個參數為資料集資料夾名稱）。"
    exit 1
fi

# 計算預期案例數（使用 dataset.json；若不存在則以 imagesTr+imagesTs 計數）
calc_expected_cases() {
    local json="$RAW_DATASET/dataset.json"
    if [[ -f "$json" ]]; then
        python3 - "$json" <<'PY' 2>/dev/null || true
import json,sys
js=json.load(open(sys.argv[1],encoding='utf-8'))
train=len(js.get('training',[]))
test=len(js.get('test',[]))
print(train+test)
PY
        return 0
    fi
    local tr="$RAW_DATASET/imagesTr" ts="$RAW_DATASET/imagesTs"
    local a=0 b=0
    [[ -d "$tr" ]] && a=$(find "$tr" -maxdepth 1 -type f -name "*_0000.nii.gz" | wc -l | tr -d ' ')
    [[ -d "$ts" ]] && b=$(find "$ts" -maxdepth 1 -type f -name "*_0000.nii.gz" | wc -l | tr -d ' ')
    echo $((a+b))
}

EXPECTED_TOTAL=$(calc_expected_cases)
if [[ -z "$EXPECTED_TOTAL" || "$EXPECTED_TOTAL" -eq 0 ]]; then
    echo "警告：無法推得預期案例數，將僅顯示檔案活動狀況。"
fi

echo "======================================"
echo "  nnU-Net 預處理即時監控"
echo "======================================"
echo "資料集: $DATASET_DIR_NAME"
echo "RAW: $RAW_DATASET"
echo "PRE: $PRE_DATASET"
echo "按 Ctrl+C 停止監控"
echo

progress_bar() {
    local current=$1 total=$2 width=${3:-30}
    if [[ "$total" -le 0 ]]; then
        printf "(未知進度)"
        return
    fi
    local perc=$(( 100 * current / total ))
    local done=$(( width * current / total ))
    local rest=$(( width - done ))
    printf "[%s%s] %3d%% (%d/%d)" "$(printf '#%.0s' $(seq 1 $done))" "$(printf '.%.0s' $(seq 1 $rest))" "$perc" "$current" "$total"
}

while true; do
    clear
    echo "--------------------------------------"
    echo "更新時間: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------"

    # 檢查預處理進程
    if pgrep -f "nnUNetv2_plan_and_preprocess" >/dev/null 2>&1; then
        echo "✓ 預處理進程運行中"
        ps -o pid,pcpu,pmem,etime,command -p "$(pgrep -f "nnUNetv2_plan_and_preprocess" | head -n1)" | sed -n '1p;$p'
    else
        echo "✗ 未偵測到預處理進程（可能已完成或尚未開始）"
    fi

    echo
    if [[ ! -d "$PRE_DATASET" ]]; then
        echo "尚未建立預處理輸出目錄：$PRE_DATASET"
        echo "下次更新: ${REFRESH_INTERVAL} 秒後..."
        sleep "$REFRESH_INTERVAL"
        continue
    fi

    plans=( )
    while IFS= read -r d; do plans+=("$d"); done < <(find "$PRE_DATASET" -maxdepth 1 -type d -name 'nnUNetPlans*' | sort)
    if [[ ${#plans[@]} -eq 0 ]]; then
        echo "尚未產生任何 plans 目錄（nnUNetPlans_*）。"
    fi

    for pdir in "${plans[@]}"; do
        pname=$(basename "$pdir")
        # 以非 _seg 的 .b2nd 檔數量代表已處理案例數
        processed=$(find "$pdir" -type f -name '*.b2nd' 2>/dev/null | grep -v '_seg\.b2nd$' | wc -l | tr -d ' ')
        echo
        echo "📦 $pname"
        if [[ -n "$EXPECTED_TOTAL" && "$EXPECTED_TOTAL" -gt 0 ]]; then
            progress_bar "$processed" "$EXPECTED_TOTAL" 40; echo
        else
            echo "已處理案例數：$processed (總數未知)"
        fi
        echo "最近產生的檔案："
        ls -1t "$pdir"/*.b2nd 2>/dev/null | head -n 5 | sed 's/^/  - /' || echo "  (尚無檔案)"
    done

    echo
    echo "下次更新: ${REFRESH_INTERVAL} 秒後..."
    echo "======================================"
    sleep "$REFRESH_INTERVAL"
done

