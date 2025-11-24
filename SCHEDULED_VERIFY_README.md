# 排程驗證說明

## 已啟動背景排程驗證

我已經建立並啟動了 `scheduled_verify_preprocessing.sh`，它會：

### 功能
1. **智能等待**：自動偵測是否有 `nnUNetv2_train` 進程在運行
   - 若偵測到訓練中，每 5 分鐘檢查一次，直到訓練完成
   - 若無訓練進程，立即開始驗證

2. **驗證預處理完整性**：執行 `nnUNetv2_plan_and_preprocess --verify_dataset_integrity`
   - 檢查資料集結構、檔案完整性
   - 不會重新預處理（除非發現嚴重錯誤）

3. **完整日誌記錄**：
   - 主輸出：`scheduled_verify.out`
   - 詳細日誌：`scheduled_verify_YYYYMMDD_HHMMSS.log`

### 目前狀態
- ✅ 排程腳本已在背景運行（PID: 5237）
- 🔄 正在等待訓練完成（每 5 分鐘檢查一次）
- 📝 輸出導向：`scheduled_verify.out`

### 如何監控

**查看即時狀態**：
```bash
tail -f scheduled_verify.out
```

**檢查排程是否還在運行**：
```bash
ps aux | grep scheduled_verify_preprocessing | grep -v grep
```

**手動觸發（不等待訓練完成）**：
```bash
WAIT_IF_TRAINING=false ./scheduled_verify_preprocessing.sh
```

### 預期行為

- 訓練完成後會自動執行驗證
- 若驗證發現問題，會在日誌中詳細記錄
- 若一切正常，會顯示 "verify_dataset_integrity Done"
- 腳本使用 `nohup` 運行，關閉終端機也會繼續執行

### 完成後的動作

驗證完成後，我建議：
1. 檢查日誌檔確認無錯誤
2. 若有問題，可以手動執行預處理修復
3. 繼續進行推論或評估

---

**提示**：由於之前 `plan_and_preprocess` 在 planning 階段有 TypeError，這次只做 verify（驗證已有檔案完整性），不重新規劃。如果需要重新規劃，請在訓練完成後手動執行完整預處理。
