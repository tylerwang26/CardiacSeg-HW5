# HW5 Project Report: Cardiac Segmentation using Ensemble of 2D and 3D nnU-Net with Soft Voting

## Team member
*   [學校] [學號] [姓名]
*   (請自行填寫)

## Abstract (摘要)
本研究旨在解決心臟 MRI 影像的自動化分割問題，目標是精確分割左心室 (LV)、右心室 (RV) 及心肌 (Myocardium)。我們採用了基於 nnU-Net 框架的集成學習策略。初期實驗顯示單一模型難以同時兼顧所有結構的分割精度，特別是右心室的複雜形狀。因此，我們提出了一種結合 **2D nnU-Net** 與 **3D Low-resolution nnU-Net** 的集成方法。

本報告重點描述最終優化版本 (V8) 的策略：採用 **Soft Voting (機率加權集成)** 取代傳統的硬標籤投票，並結合 **Test Time Augmentation (TTA)** 與針對性的後處理 (LCC, Fill Holes, Morphological Closing)。實驗結果顯示，此策略能有效結合 2D 模型在平面解析度上的優勢與 3D 模型在空間連續性上的優勢，將 Dice Score 從早期的 0.36 大幅提升至 **0.7905**。

## Introduction (前言)
心臟 MRI 分割是心血管疾病診斷的關鍵步驟。然而，由於 MRI 影像通常具有各向異性 (Anisotropic) 的解析度（層間距較大），且心臟結構複雜（如右心室形狀不規則），使得全自動分割極具挑戰性。

現有的解決方案多基於 U-Net 架構。nnU-Net 作為目前的 State-of-the-art (SOTA) 方法，能自動適應資料集特徵。然而，我們發現：
1.  **2D 模型**：擅長處理層內高解析度資訊，對右心室 (RV) 的分割表現較好，但缺乏層間的空間連續性。
2.  **3D 模型**：能捕捉 3D 空間資訊，對左心室 (LV) 和心肌 (Myo) 的連續性較好，但在 Z 軸解析度不足時容易產生偽影。

為了克服單一模型的限制，我們提出了 **V6 集成策略**，透過 **Soft Voting** 機制，在機率層面上融合 2D 與 3D 模型的預測信心，以達到最佳的分割效果。

## Method (方法)

本研究的核心方法為 **多視圖集成 (Multi-view Ensemble)** 搭配 **Soft Voting**。

### 1. 模型架構 (Model Architectures)
我們訓練了兩個獨立的 nnU-Net 模型：
*   **2D nnU-Net**: 逐層 (Slice-by-slice) 進行分割，輸入尺寸為 $512 \times 512$。此模型對於邊緣細節（特別是右心室）的捕捉能力較強。
*   **3D Low-resolution nnU-Net**: 將原始影像降採樣後進行 3D 卷積運算。此模型擁有更大的感受野 (Receptive Field)，能有效維持心臟結構的 3D 拓樸正確性（如心肌的環狀結構）。

### 2. 推論增強 (Inference Strategy: TTA)
在推論階段，我們開啟了 **Test Time Augmentation (TTA)**。具體做法是對輸入影像進行多軸向的鏡像翻轉 (Mirroring)，分別進行預測後再取平均。這雖然增加了推論時間，但能顯著消除模型的隨機誤差，提升預測的穩健性。

### 3. Soft Voting Ensemble (機率加權集成)
這是 V6 版本的核心改進。不同於之前的 Hard Voting (直接對類別標籤投票)，Soft Voting 是對模型輸出的 **Softmax 機率圖 (Probability Maps)** 進行加權平均。

假設 $P_{2d}(x)$ 和 $P_{3d}(x)$ 分別為 2D 和 3D 模型在體素 (Voxel) $x$ 處預測某類別的機率，最終機率 $P_{final}(x)$ 定義為：

$$ P_{final}(x) = w_{2d} \cdot P_{2d}(x) + w_{3d} \cdot P_{3d}(x) $$

我們根據驗證集 (Validation Set) 的表現，針對不同解剖結構設計了特定的權重矩陣 $W$：

| 解剖結構 (Label) | 2D 權重 ($w_{2d}$) | 3D 權重 ($w_{3d}$) | 設計理由 |
| :--- | :---: | :---: | :--- |
| **Background (0)** | 0.5 | 0.5 | 中立 |
| **Myocardium (1)** | 0.3 | **0.7** | 3D 模型更能維持心肌的環狀連續性 |
| **Left Ventricle (2)** | 0.4 | **0.6** | 3D 模型對左心室的整體形狀掌握較佳 |
| **Right Ventricle (3)** | **0.8** | 0.2 | 2D 模型對形狀不規則的右心室分割更精準 |

最終預測結果取 $argmax(P_{final}(x))$。

### 4. 後處理 (Post-processing)
為了修正模型可能產生的拓樸錯誤，我們實施了以下後處理：
*   **保留最大連通區域 (Keep Largest Connected Component, LCC)**：針對每個類別，僅保留體積最大的連通區塊，移除零散的雜訊誤判。
*   **填補孔洞 (Fill Holes)**：針對左心室 (Label 2) 和右心室 (Label 3)，執行 3D 孔洞填補，確保血池結構的實心完整性。

## Experiment (實驗與結果分析)

### 1. 實驗設置
*   **Dataset**: CardiacSeg (AICUP/MICCAI)
*   **Environment**: NVIDIA RTX 4090, PyTorch, nnU-Net V2
*   **Evaluation Metric**: Dice Similarity Coefficient (DSC)

### 2. 方法演進與失敗分析 (Evolution of Approaches)

我們的方法經歷了多次迭代，以下是各階段的關鍵發現：

*   **Phase 1: 初期嘗試 (HW3 原始版本)**
    *   **方法**: 單一 2D 模型，無特殊處理。
    *   **結果**: Dice Score **0.3635**。
    *   **原因**: 模型訓練不足，且未處理資料型態與維度問題，導致嚴重的 Underfitting。

*   **Phase 2: Baseline 建立**
    *   **方法**: 2D + 3D Lowres Ensemble (Hard Voting)，無 TTA。
    *   **結果**: Dice Score **0.7829**。
    *   **分析**: 這是我們表現最好的基準。證明了 2D 與 3D Lowres 的互補性。

*   **Phase 3: 錯誤的優化方向 (V2/V3)**
    *   **方法**: 嘗試引入 **3D Full-resolution** 模型並使用激進的 LCC。
    *   **結果**: 分數下降至 **0.77** 左右。
    *   **失敗原因**:
        1.  **Overfitting**: 3D Fullres 模型參數量大，在有限的訓練資料下，對測試集的泛化能力反而不如 Lowres 模型。
        2.  **資料型態錯誤**: 在處理過程中發生了 float/uint8 的型態轉換錯誤，導致精度流失。

*   **Phase 4 & 5: 回歸與修正 (V4/V5)**
    *   **方法**: 回歸 **3D Lowres**。開啟 **TTA**，並修復資料型態 Bug。加入 LCC 與 Fill Holes。
    *   **結果**: Dice Score 回升至 **0.7833**。
    *   **分析**: 證明 3D Lowres 確實是較佳的選擇，且 TTA 與後處理能帶來微幅提升。

*   **Phase 6: Soft Voting優化 (Current V6)**
    *   **方法**: **Soft Voting Ensemble** (機率加權) + TTA + Advanced Post-processing。
    *   **結果**: Dice Score **0.7901**。
    *   **分析**: 相比於 Hard Voting 僅利用最終標籤，Soft Voting 保留了模型的信心資訊，對於邊緣模糊的體素 (Voxel) 能做出更穩健的判斷，成功突破了 0.79 的瓶頸。

*   **Phase 7: 嘗試引入 3D Fullres (V7)**
    *   **方法**: 嘗試將 3D Fullres (Fold 0 & Fold 1) 加入集成。
    *   **結果**: 分數微幅下降至 **0.7898**。
    *   **分析**: 確認 3D Fullres 模型在此資料集上並未帶來額外增益，反而引入了雜訊，這可能與資料集的 Z 軸解析度有關。

*   **Phase 8: 形態學優化 (Current V8)**
    *   **方法**: 移除 3D Fullres，回歸 V6 架構 (2D + 3D Lowres)。新增 **Morphological Closing (形態學閉運算)** 於後處理步驟。
    *   **結果**: Dice Score 提升至 **0.7905**。
    *   **分析**: 閉運算成功填補了部分細微的孔洞並平滑了邊緣，帶來了 0.0004 的關鍵提升。

## Future Work (未來展望)
目前的優化已將單一 Fold (Fold 0) 的潛力發揮至極限。為了突破 0.80 的大關，我們已啟動 **5-Fold Cross Validation (五折交叉驗證)** 計畫。
*   **策略**: 訓練剩餘的 4 個 Folds (目前已完成 Fold 0)，並對測試集進行 5 個模型的集成推論。
*   **預期**: 透過消除資料切分的偏差，預期能進一步提升模型的泛化能力與穩定性。

## Reference
1.  Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods* (2021).