# CardiacSeg：心臟影像分割專案

> ⚡ 新增：快速訓練與時間策略（若你只想在有限時間內得到可用結果，先讀這一段）

## ⚡ 快速訓練與時間策略

針對課堂/展示或只有 4~8 小時的運算時間，以下是建議的「漸進式」策略，避免一開始就投入極長的 1000 epochs 完整 3D 全解析度訓練。

### 1. 目標分層

- 快速可視化結果（30~60 分鐘）：確認流程正確與標籤合理。
- 初步可用模型（2~4 小時）：Dice 有意義 (>0.6~0.7)。
- 改善與調參（4~8 小時）：加入 early stopping，視資料大小決定是否升級到 fullres。

### 2. 訓練配置選擇指南

| 目的 | 建議 configuration | Epoch 範圍 | 特性 | 何時升級 |
|------|--------------------|-----------|------|-----------|
| 最快跑通 | `2d` | 30~80 | 速度最快，空間資訊較弱 | 需要 3D 空間品質時 |
| 折衷 | `3d_lowres` | 60~150 | 訓練較快，解析度較低 | 想提升邊界細節時 |
| 最終模型 | `3d_fullres` | 150~400 (早停) | 最佳細節與 Dice | 已確認流程 + 有足夠時間 |

> 如果記憶體或時間吃緊，就用 `2d` 或 `3d_lowres` 先產生一個 baseline，再決定是否投資 `3d_fullres`。

> 實務建議：先以 `3d_lowres` 或 `2d` 進行特徵學習與快速驗證，滿意後再升級至 `3d_fullres` 精緻邊界與細節。

### 3. 建議漸進實作流程

1. 第一步：`2d` 或 `3d_lowres`，設定 `--epochs 80`，啟用 early stopping（patience 15）。
2. 觀察 validation pseudo Dice（log 檔或終端輸出）。若 30~50 epochs 即趨於平穩，可視為收斂。
3. 第二步（可選）：改用 `3d_fullres`，載入同資料預處理結果（不需重跑 rename），設定 `--epochs 250` + early stopping。
4. 如需更高品質：可再執行 cascade（`3d_lowres` → `3d_fullres`），但務必確認有額外時間。

### 4. 時間預估（以 Apple Silicon / 中階 GPU 為例）

- 預處理 (Dataset001 中小型)：10~30 分鐘。
- `2d` 訓練：每 10 epochs 約 5~12 分鐘。
- `3d_lowres`：每 10 epochs 約 10~20 分鐘。
- `3d_fullres`：首個 epoch 最慢（資料載入與 cache），後續每 10 epochs 約 20~40 分鐘。

> 實際時間受資料集大小、I/O、GPU/記憶體影響；首 epoch 往往比後續慢 1.3~2 倍。

### 5. Early Stopping 使用提示

- 已提供 `early_stopping_monitor.sh`（若在專案中）可解析訓練 log，檢測若連續 PATIENCE 個 epochs 未提升則送出終止訊號。
- 好處：避免長期尾端微幅改善浪費時間；在探索階段尤其有效。
- 建議參數：`patience = 15`，`min_delta = 0.002`（Dice 微提升再算改進）。

### 6. 指令範例

```bash
# 2D 快速 baseline
python nnunet_train.py --config 2d --epochs 80 --fold 0

# 3D 低解析度（較平衡）
python nnunet_train.py --config 3d_lowres --epochs 120 --fold 0

# 3D 全解析度（啟用早停理念，自行啟動監控腳本）
python nnunet_train.py --config 3d_fullres --epochs 250 --fold 0
```

（macOS/Linux 可將 `python` 換成 `python3`，如使用 MPS 會自動啟用。）

### 7. 何時結束訓練？

- validation pseudo Dice 長時間（>10~15 epochs）無提升。
- Loss 緩慢下降但 Dice 停滯 → 已達資料與模型容量瓶頸。
- 在展示或報告截止前保留足夠時間完成推論與評估。

### 8. 常見加速決策

- 確認使用單一 Fold（`--fold 0`）避免 5-Fold 全部展開。
- 減少 epochs 上限並依 early stopping 實際收斂點提前結束。
- 若 I/O 成為瓶頸，可將資料放置於更快的 SSD。
- 先產生 baseline 指標再投入高解析度，降低浪費。

### 9. 推薦快速開始組合

| 步驟 | 配置 | 目的 | 結果期望 |
|------|------|------|-----------|
| 1 | 2d | 確認流程、標籤合理 | Dice 粗略 >0.5 |
| 2 | 3d_lowres | 建立 3D baseline | Dice 提升 0.05~0.15 |
| 3 | 3d_fullres | 精緻化 | 邊界細節 & 穩定提升 |

### 10. 產出品質 vs 時間的心法

「先得到一個不錯的答案，再尋找更好的答案」：用 `2d`/`3d_lowres` 驗證資料乾淨、標籤可信後，`3d_fullres` 的時間投入才有意義。避免直接投入 1000 epochs 導致等待過長卻不確定資料是否正確。

---

本專案使用 **nnU-Net v2** 進行 3D 心臟影像分割，支援自動環境配置、資料預處理、模型訓練、推論與評估。

---

## 📋 目錄

<!-- toc:start -->
<!-- 目錄由 scripts/generate_toc.py 自動產生，請勿手動編輯 -->
- [CardiacSeg：心臟影像分割專案](#cardiacseg心臟影像分割專案)
  - [⚡ 快速訓練與時間策略](#-快速訓練與時間策略)
    - [1. 目標分層](#1-目標分層)
    - [2. 訓練配置選擇指南](#2-訓練配置選擇指南)
    - [3. 建議漸進實作流程](#3-建議漸進實作流程)
    - [4. 時間預估（以 Apple Silicon / 中階 GPU 為例）](#4-時間預估以-apple-silicon--中階-gpu-為例)
    - [5. Early Stopping 使用提示](#5-early-stopping-使用提示)
    - [6. 指令範例](#6-指令範例)
    - [7. 何時結束訓練？](#7-何時結束訓練)
    - [8. 常見加速決策](#8-常見加速決策)
    - [9. 推薦快速開始組合](#9-推薦快速開始組合)
    - [10. 產出品質 vs 時間的心法](#10-產出品質-vs-時間的心法)
  - [📋 目錄](#-目錄)
  - [📝 HW5 專案報告 (Project Report)](#-hw5-專案報告-project-report)
    - [0. 團隊成員 (Team Member)](#0-團隊成員-team-member)
    - [1. 摘要 (Abstract)](#1-摘要-abstract)
    - [2. 方法論 (Methodology)](#2-方法論-methodology)
      - [模型架構](#模型架構)
      - [Soft Voting Ensemble (機率加權集成)](#soft-voting-ensemble-機率加權集成)
      - [推論增強與後處理](#推論增強與後處理)
    - [3. 實驗過程與演進 (Evolution of Approaches)](#3-實驗過程與演進-evolution-of-approaches)
    - [4. 未來展望 (Future Work)](#4-未來展望-future-work)
  - [🚀 快速開始 {#quick-start}](#-快速開始-quick-start)
    - [前置需求](#前置需求)
    - [一鍵安裝環境](#一鍵安裝環境)
      - [Windows 用戶](#windows-用戶)
      - [macOS / Linux 用戶](#macos--linux-用戶)
        - [方式 1：使用 bash 腳本](#方式-1使用-bash-腳本)
        - [方式 2：使用 PowerShell 腳本（需先安裝 PowerShell）](#方式-2使用-powershell-腳本需先安裝-powershell)
    - [✅ 分步腳本（適合教學/分享）](#-分步腳本適合教學分享)
      - [macOS / Linux](#macos--linux)
      - [Windows（PowerShell）](#windowspowershell)
  - [📁 專案結構 {#project-structure}](#-專案結構-project-structure)
  - [🔧 腳本說明 {#scripts}](#-腳本說明-scripts)
    - [1. `setup_environment.ps1` / `setup_environment.sh`（環境安裝腳本）](#1-setup_environmentps1--setup_environmentsh環境安裝腳本)
    - [2. `rename_dataset.py`（資料集下載與標準化）](#2-rename_datasetpy資料集下載與標準化)
    - [3. `nnunet_train.py`（訓練腳本）](#3-nnunet_trainpy訓練腳本)
    - [4. `nnunet_infer.py`（推論腳本）](#4-nnunet_inferpy推論腳本)
    - [5. `nnunet_evaluate.py`（評估腳本）](#5-nnunet_evaluatepy評估腳本)
  - [🔄 使用流程 {#usage}](#-使用流程-usage)
    - [完整訓練流程（從零開始）](#完整訓練流程從零開始)
  - [📚 原理解析 {#concepts}](#-原理解析-concepts)
    - [nnU-Net 簡介](#nnu-net-簡介)
    - [資料格式要求](#資料格式要求)
    - [資料預處理流程](#資料預處理流程)
    - [訓練策略](#訓練策略)
  - [❓ 常見問題 {#faq}](#-常見問題-faq)
    - [Q1：執行 `setup_environment.ps1` 時顯示「無法載入，因為這個系統上已停用指令碼執行」？](#q1執行-setup_environmentps1-時顯示無法載入因為這個系統上已停用指令碼執行)
    - [Q2：訓練過程中出現「CUDA out of memory」或記憶體不足錯誤？](#q2訓練過程中出現cuda-out-of-memory或記憶體不足錯誤)
    - [Q3：如何查看訓練進度？](#q3如何查看訓練進度)
    - [Q4：如何恢復中斷的訓練？](#q4如何恢復中斷的訓練)
    - [Q5：資料集太大，預處理很慢怎麼辦？](#q5資料集太大預處理很慢怎麼辦)
    - [Q6：macOS 上執行腳本時顯示「command not found: python」？](#q6macos-上執行腳本時顯示command-not-found-python)
    - [Q7：如何分享此專案給同學？](#q7如何分享此專案給同學)
  - [📖 參考資料](#-參考資料)
  - [📧 聯絡資訊](#-聯絡資訊)
<!-- toc:end -->

---

## 📝 HW5 專案報告 (Project Report)

### 0. 團隊成員 (Team Member)
*   [學校] [學號] [姓名]
*   (請自行填寫)

### 1. 摘要 (Abstract)

本研究旨在解決心臟 MRI 影像的自動化分割問題，目標是精確分割左心室 (LV)、右心室 (RV) 及心肌 (Myocardium)。我們採用了基於 nnU-Net 框架的集成學習策略。初期實驗顯示單一模型難以同時兼顧所有結構的分割精度，特別是右心室的複雜形狀。因此，我們提出了一種結合 **2D nnU-Net** 與 **3D Low-resolution nnU-Net** 的集成方法。

本報告重點描述最終優化版本 (V8) 的策略：採用 **Soft Voting (機率加權集成)** 取代傳統的硬標籤投票，並結合 **Test Time Augmentation (TTA)** 與針對性的後處理 (LCC, Fill Holes, Morphological Closing)。實驗結果顯示，此策略能有效結合 2D 模型在平面解析度上的優勢與 3D 模型在空間連續性上的優勢，將 Dice Score 從早期的 0.36 大幅提升至 **0.7905**。

### 2. 方法論 (Methodology)

本研究的核心方法為 **多視圖集成 (Multi-view Ensemble)** 搭配 **Soft Voting**。

#### 模型架構
我們訓練了兩個獨立的 nnU-Net 模型：
*   **2D nnU-Net**: 逐層 (Slice-by-slice) 進行分割，輸入尺寸為 $512 \times 512$。此模型對於邊緣細節（特別是右心室）的捕捉能力較強。
*   **3D Low-resolution nnU-Net**: 將原始影像降採樣後進行 3D 卷積運算。此模型擁有更大的感受野 (Receptive Field)，能有效維持心臟結構的 3D 拓樸正確性（如心肌的環狀結構）。

#### Soft Voting Ensemble (機率加權集成)
這是 V6 版本的核心改進。不同於之前的 Hard Voting (直接對類別標籤投票)，Soft Voting 是對模型輸出的 **Softmax 機率圖 (Probability Maps)** 進行加權平均。

假設 $P_{2d}(x)$ 和 $P_{3d}(x)$ 分別為 2D 和 3D 模型在體素 (Voxel) $x$ 處預測某類別的機率，最終機率 $P_{final}(x)$ 定義為：

$$ P_{final}(x) = w_{2d} \cdot P_{2d}(x) + w_{3d} \cdot P_{3d}(x) $$

我們根據驗證集 (Validation Set) 的表現，針對不同解剖結構設計了特定的權重矩陣 $W$：

| 解剖結構 (Label) | 2D 權重 ($w_{2d}$) | 3D 權重 ($w_{3d}$) | 設計理由 |
| :--- | :---: | :---: | :--- |
| **Background (0)** | 0.5 | 0.5 | 中立 |
| **Myocardium (1)** | 0.3 | **0.7** | 3D 模型更能維持心肌的環狀連續性 |
| **Left Ventricle (2)** | 0.3 | **0.7** | 3D 模型對左心室的整體形狀掌握較佳 |
| **Right Ventricle (3)** | **0.65** | 0.35 | 2D 模型對形狀不規則的右心室分割更精準 |

#### 推論增強與後處理
*   **TTA (Test Time Augmentation)**：推論時開啟多軸向鏡像翻轉，消除隨機誤差。
*   **LCC (Keep Largest Connected Component)**：針對每個類別僅保留最大連通區域，移除雜訊。
*   **Fill Holes**：填補左/右心室內部的孔洞。
*   **Morphological Closing**：使用形態學閉運算平滑邊緣 (V8 新增)。

### 3. 實驗過程與演進 (Evolution of Approaches)

我們的方法經歷了多次迭代，以下是各階段的關鍵發現與困難解決：

*   **Phase 1: 初期嘗試 (HW3 原始版本)**
    *   **結果**: Dice Score **0.3635** (Underfitting)。
    *   **問題**: 模型訓練不足，未處理資料型態與維度問題。

*   **Phase 2: Baseline 建立**
    *   **方法**: 2D + 3D Lowres Ensemble (Hard Voting)。
    *   **結果**: Dice Score **0.7829**。
    *   **發現**: 證明了 2D 與 3D Lowres 的互補性。

*   **Phase 3: 錯誤的優化方向 (V2/V3)**
    *   **方法**: 引入 3D Full-resolution 模型。
    *   **結果**: 分數下降至 **0.77**。
    *   **困難**: 3D Fullres 模型過擬合 (Overfitting)，且發生 float/uint8 型態轉換錯誤。
    *   **解決**: 放棄 Fullres，專注於優化 Lowres 與 2D 的結合，並修復程式碼 Bug。

*   **Phase 4 & 5: 回歸與修正 (V4/V5)**
    *   **方法**: 回歸 3D Lowres + TTA + LCC/Fill Holes。
    *   **結果**: Dice Score 回升至 **0.7833**。

*   **Phase 6: Soft Voting 優化 (Current V6)**
    *   **方法**: 改用 **Soft Voting** (機率加權)。
    *   **結果**: Dice Score **0.7901**。
    *   **突破**: 成功突破 0.79 瓶頸，證明保留信心資訊比單純投票更有效。

*   **Phase 7: 過度集成 (V7)**
    *   **方法**: 嘗試加入 3D Fullres (Fold 0 & Fold 1) 進行 4 模型集成。
    *   **結果**: Dice Score 下降至 **0.7898**。
    *   **分析**: 證實 3D Fullres 模型在此資料集上表現不佳，加入後反而引入雜訊，拖累整體表現。

*   **Phase 8: 形態學優化 (Current V8)**
    *   **方法**: 新增 **Morphological Closing**。
    *   **結果**: Dice Score 提升至 **0.7905**。
    *   **結論**: 細微的形態學處理能進一步提升邊界品質。

### 4. 未來展望 (Future Work)
目前的優化已將單一 Fold (Fold 0) 的潛力發揮至極限。為了突破 0.80 的大關，我們已啟動 **5-Fold Cross Validation** 計畫，預期透過消除資料切分的偏差，能進一步提升模型的泛化能力。

---

## 🚀 快速開始 {#quick-start}

### 前置需求

- **作業系統**：Windows 10/11、macOS（包括 Apple Silicon M1/M2/M3/M4）、Linux
- **Python**：3.8 ~ 3.11（建議 3.11）
- **硬體**：
  - CPU：多核心處理器（訓練時會使用所有核心）
  - RAM：至少 16GB（建議 32GB）
  - 硬碟空間：至少 50GB（用於資料集、預處理結果和模型檔案）
  - GPU：
    - **Windows/Linux**：NVIDIA GPU（選配，可加速訓練）
    - **macOS**：Apple Silicon M1/M2/M3/M4（自動使用 MPS 加速）

### 一鍵安裝環境

#### Windows 用戶

1. **解壓縮本專案資料夾至任意位置**（例如 `C:\CardiacSeg`）

2. **開啟 PowerShell**（以系統管理員身分執行），切換到專案目錄：

  
  ```powershell
  cd C:\CardiacSeg
  ```

1. **允許執行腳本**（僅需執行一次）：

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

1. **執行環境安裝腳本（會自動建立 .venv）**：

  ```powershell
  .\setup_environment.ps1
  ```

1. **重新開啟終端**（或登出後重新登入，確保 PATH 生效），即可開始使用！

   - 啟用虛擬環境（PowerShell）：

     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

#### macOS / Linux 用戶

1. **解壓縮本專案資料夾至任意位置**（例如 `~/CardiacSeg`）

1. **開啟終端（Terminal）**，切換到專案目錄：

  ```bash
  cd ~/CardiacSeg
  ```

1. **執行環境安裝腳本**（會自動建立 .venv；選擇以下任一方式）：

##### 方式 1：使用 bash 腳本


   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

##### 方式 2：使用 PowerShell 腳本（需先安裝 PowerShell）


   ```bash
   pwsh setup_environment.ps1
   ```

1. **啟用虛擬環境**（建議）：

  ```bash
  source .venv/bin/activate
  ```

1. **（可選）設定永久環境變數**：將以下內容加入 `~/.bashrc` 或 `~/.zshrc`：

  ```bash
  export nnUNet_raw="~/CardiacSeg/nnUNet_raw"
  export nnUNet_preprocessed="~/CardiacSeg/nnUNet_preprocessed"
  export nnUNet_results="~/CardiacSeg/nnUNet_results"
  ```

安裝腳本會自動：

- ✅ 偵測作業系統（Windows/macOS/Linux）
- ✅ 偵測或提示安裝 Python 3.8+
- ✅ 升級 pip 到最新版本
- ✅ 偵測 GPU/加速器類型並安裝對應的 PyTorch：
  - **Windows**：NVIDIA GPU → CUDA 版本；AMD/無 GPU → CPU 版本
  - **macOS**：Apple Silicon → MPS 加速；Intel → CPU 版本
  - **Linux**：自動偵測 CUDA
- ✅ 安裝 nnU-Net v2 及所有依賴套件（monai, nibabel, SimpleITK, scikit-image 等）
- ✅ 建立專案虛擬環境 `.venv` 並在其中安裝依賴
- ✅ 設定 nnU-Net 環境變數（`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`）
- ✅ 驗證所有套件是否正確安裝

---

### ✅ 分步腳本（適合教學/分享）

若要用更清楚的四步驟來安裝與驗證，專案已提供分步腳本：

小提醒：所有指令都請在專案根目錄執行（`CardiacSeg/`）。

#### macOS / Linux

```bash
# 只需執行一次：賦予腳本執行權限
chmod +x scripts/macos_linux/*.sh

# Step 1：建立虛擬環境 .venv
bash scripts/macos_linux/01_create_venv.sh

# Step 2：安裝核心套件（torch/torchvision、nnUNetv2、monai、nibabel、blosc2 等）
bash scripts/macos_linux/02_install_core.sh

# Step 3：設定 nnU-Net 需要的環境變數，並產生 .env 檔
bash scripts/macos_linux/03_set_env.sh

# 建議在之後的每個新終端都執行，載入環境變數
source .env.nnunet.sh

# Step 4：驗證安裝（檢查 torch/monai/nibabel/nnunetv2 與裝置後端）
bash scripts/macos_linux/04_verify.sh
```

#### Windows（PowerShell）

```powershell
# Step 1：建立虛擬環境 .venv
scripts\windows\01_create_venv.ps1

# Step 2：安裝核心套件（會自動偵測 NVIDIA 並裝 CUDA 版 torch）
scripts\windows\02_install_core.ps1

# Step 3：設定環境變數並產生 .env 檔
scripts\windows\03_set_env.ps1

# 建議在之後的每個新視窗都 dot-source 載入變數
. .\.env.nnunet.ps1

# Step 4：驗證安裝
scripts\windows\04_verify.ps1
```

完成上述四步後，即可繼續執行：

```bash
# Windows
python rename_dataset.py && python nnunet_train.py

# macOS/Linux
python3 rename_dataset.py && python3 nnunet_train.py
```

---

## 📁 專案結構 {#project-structure}

```text
CardiacSeg/
├── setup_environment.ps1          # 環境安裝腳本（Windows/macOS/Linux 通用）
├── setup_environment.sh           # 環境安裝腳本（bash 版本，macOS/Linux 專用）
├── rename_dataset.py              # 資料集下載與標準化腳本
├── nnunet_train.py                # 訓練腳本（預處理 + 訓練）
├── nnunet_infer.py                # 推論腳本（對測試集預測）
├── nnunet_evaluate.py             # 評估腳本（計算 Dice、IoU）
├── README.md                      # 本說明文件
├── dataset.json                   # （可選）手動準備的資料集描述檔
├── nnUNet_raw/                    # 原始資料集目錄
│   └── Dataset001_CardiacSeg/
│       ├── imagesTr/              # 訓練影像（.nii.gz）
│       ├── labelsTr/              # 訓練標籤（.nii.gz）
│       ├── imagesTs/              # 測試影像（.nii.gz）
│       └── dataset.json           # nnU-Net 資料集描述檔
├── nnUNet_preprocessed/           # 預處理後的資料（自動生成）
└── nnUNet_results/                # 訓練結果與模型權重（自動生成）
```

---

## 🔧 腳本說明 {#scripts}

### 1. `setup_environment.ps1` / `setup_environment.sh`（環境安裝腳本）

**功能**：跨平台一鍵配置完整的 Python 環境，包含 nnU-Net 及所有依賴套件。

**原理**：

- **作業系統偵測**：自動識別 Windows、macOS 或 Linux，執行對應的安裝邏輯。
- **Python 探測**：
  - **Windows**：優先搜尋本機已安裝的 Python 3.x（排除 Windows Store Alias 和 venv 虛擬環境），若未找到則透過 `winget` 自動安裝 Python 3.11。
  - **macOS/Linux**：檢查 `python3` 是否存在，若無則提示安裝。
- **GPU/加速器偵測**：
  - **Windows**：查詢系統 GPU 型號（透過 WMI），若為 NVIDIA GPU 則安裝 PyTorch CUDA 版本，AMD GPU 或無 GPU 則使用 CPU 版本。
  - **macOS**：偵測 Apple Silicon（M1/M2/M3/M4），自動安裝支援 MPS 加速的 PyTorch。
  - **Linux**：安裝 PyTorch 並自動偵測 CUDA（若有）。
- **環境變數設定**：將 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 設為當前專案目錄，供 nnU-Net 辨識資料位置。
- **PATH 更新**（僅 Windows）：將 Python 主目錄和 `Scripts` 子目錄（含 nnU-Net 執行檔）加入使用者環境變數 PATH。

**執行方式**：

```bash
# Windows
.\setup_environment.ps1

# macOS/Linux（bash）
./setup_environment.sh

# macOS/Linux（PowerShell）
pwsh setup_environment.ps1
```

**輸出範例（Windows）**：

```text
========================================
CardiacSeg Environment Setup
========================================
Detected: Windows

[1/5] Checking/Installing Python...
OK Using Python: C:\Users\Wang\AppData\Local\Programs\Python\Python311\python.exe (Python 3.11.9)
PATH updated with Python directory: C:\Users\Wang\AppData\Local\Programs\Python\Python311
PATH updated with Scripts directory: C:\Users\Wang\AppData\Local\Programs\Python\Python311\Scripts

[2/5] Upgrading pip...
Requirement already satisfied: pip in ...

[3/5] Installing Python packages (this may take a few minutes)...
Detected NVIDIA GPU. Installing CUDA-enabled PyTorch (cu121).
    OK torch/torchvision (CUDA) installed
  Installing nnunetv2...
    OK nnunetv2 installed successfully
  ...

[4/5] Setting nnU-Net environment variables...
OK Environment variables set:
  nnUNet_raw = C:\CardiacSeg\nnUNet_raw
  nnUNet_preprocessed = C:\CardiacSeg\nnUNet_preprocessed
  nnUNet_results = C:\CardiacSeg\nnUNet_results

========================================
Verifying Installation
========================================
Checking torch... ✓ Version: 2.2.1+cu121
Checking nnunetv2... ✓ Version: 2.6.2
...

Backend status:
torch version: 2.2.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3060

========================================
Setup Complete!
========================================

You can now run:
  python rename_dataset.py    (Rename dataset files)
  python nnunet_train.py      (Train model)
  python nnunet_infer.py      (Inference)
  python nnunet_evaluate.py   (Evaluation)
```

**輸出範例（macOS Apple Silicon）**：

```text
========================================
CardiacSeg Environment Setup (macOS)
========================================

[1/5] Checking Python installation...
✓ Using Python: /opt/homebrew/bin/python3 (Python 3.11.5)

[2/5] Upgrading pip...
Requirement already satisfied: pip in ...

[3/5] Installing Python packages (this may take a few minutes)...
Detected Apple Silicon (M1/M2/M3/M4). Installing MPS-enabled PyTorch.
  Installing torch and torchvision...
  ✓ torch/torchvision installed (Apple Silicon MPS support)
  Installing nnunetv2...
    ✓ nnunetv2 installed successfully
  ...

[4/5] Setting nnU-Net environment variables...
✓ Environment variables set:
  nnUNet_raw = /Users/tyler/CardiacSeg/nnUNet_raw
  nnUNet_preprocessed = /Users/tyler/CardiacSeg/nnUNet_preprocessed
  nnUNet_results = /Users/tyler/CardiacSeg/nnUNet_results

========================================
Verifying Installation
========================================
Checking torch... ✓ Version: 2.2.1
Checking nnunetv2... ✓ Version: 2.6.2
...

Backend status:
torch version: 2.2.1
MPS (Apple Silicon GPU) available: True
Using Apple Silicon GPU acceleration

========================================
Setup Complete!
========================================

You can now run:
  python3 rename_dataset.py    (Rename dataset files)
  python3 nnunet_train.py      (Train model)
  python3 nnunet_infer.py      (Inference)
  python3 nnunet_evaluate.py   (Evaluation)
```

---

### 2. `rename_dataset.py`（資料集下載與標準化）

**功能**：自動下載資料集（支援 Google Drive、ZIP 壓縮檔）並將檔名標準化為 nnU-Net 格式。

**原理**：

- **快速跳過邏輯**（Fast Skip）：檢查 `imagesTr/`、`labelsTr/`、`imagesTs/` 中的檔案是否已符合 nnU-Net 命名規則（影像：`_0000.nii.gz` 結尾；標籤：無 `_gt` 後綴）。若全部符合且 `dataset.json` 已存在，則直接跳過重新處理。
- **檔名標準化**：
  - **影像檔**：`patient0001.nii.gz` → `patient0001_0000.nii.gz`（`_0000` 表示單一影像模態）
  - **標籤檔**：`patient0001_gt.nii.gz` → `patient0001.nii.gz`（移除 `_gt` 後綴）
- **資料集描述檔生成**（`dataset.json`）：
  - 動態掃描所有訓練影像和標籤檔，建立對應清單。
  - 讀取標籤檔中的所有唯一標籤值（如 0, 1, 2, 3），自動生成 `labels` 字典：

    ```json
    {
      "labels": {
        "background": 0,
        "label1": 1,
        "label2": 2,
        "label3": 3
      },
      "numTraining": 50,
      "file_ending": ".nii.gz",
      "training": [
        {"image": "./imagesTr/patient0001_0000.nii.gz", "label": "./labelsTr/patient0001.nii.gz"},
        ...
      ]
    }
    ```

**執行方式**：

```bash
# Windows
python rename_dataset.py           # 自動跳過已標準化的檔案
python rename_dataset.py --force   # 強制重新處理所有檔案

# macOS/Linux
python3 rename_dataset.py           # 自動跳過已標準化的檔案
python3 rename_dataset.py --force   # 強制重新處理所有檔案
```

**輸出範例**：

```text
腳本 base_dir: /Users/tyler/CardiacSeg
已生成/更新 dataset.json 於 /Users/tyler/CardiacSeg/nnUNet_raw/Dataset001_CardiacSeg/dataset.json
檢測到的標籤：[0, 1, 2, 3]
重命名完成！
```

---

### 3. `nnunet_train.py`（訓練腳本）

**功能**：執行 nnU-Net 資料預處理（Planning & Preprocessing）並啟動模型訓練。

**原理**：

- **預處理階段**（`nnUNetv2_plan_and_preprocess`）：
  - 分析資料集的影像尺寸、間距（spacing）、強度分佈。
  - 自動設計網路架構（如 3D U-Net 的卷積核大小、池化層數）。
  - 將原始資料重新取樣（resampling）至統一間距，並裁剪/填補至固定尺寸。
  - 生成訓練所需的預處理資料（保存於 `nnUNet_preprocessed/`）。
- **訓練階段**（`nnUNetv2_train`）：
  - 使用 5-Fold 交叉驗證（預設訓練 Fold 0）。
  - 自動使用所有 CPU 核心進行資料增強（Data Augmentation）。
  - 訓練過程會定期儲存檢查點（checkpoint）於 `nnUNet_results/`。

**執行方式**：

```bash
# Windows
python nnunet_train.py

# macOS/Linux
python3 nnunet_train.py
```

**關鍵參數**（可於腳本中修改）：

- `dataset_id = "001"`：資料集 ID。
- `configuration = "3d_fullres"`：3D 全解析度配置。
- `fold = "0"`：訓練第 0 個 Fold（可設為 `"all"` 訓練所有 Fold）。
- `planner = "nnUNetPlannerResEncL"`：使用 Residual Encoder 大型架構（適合高記憶體 GPU）。

**輸出範例**：

```text
腳本 base_dir: /Users/tyler/CardiacSeg
環境變數已設定：
  NNUNET_RAW: /Users/tyler/CardiacSeg/nnUNet_raw
  NNUNET_PREPROCESSED: /Users/tyler/CardiacSeg/nnUNet_preprocessed
  NNUNET_RESULTS: /Users/tyler/CardiacSeg/nnUNet_results

運行 nnUNetv2_plan_and_preprocess...
[nnU-Net] Planning experiment...
[nnU-Net] Preprocessing data...
預處理完成！

運行 nnUNetv2_train...
[nnU-Net] Starting training...
Epoch 1/1000: loss=0.5234, Dice=0.7845
...
訓練完成！模型已儲存至 nnUNet_results/
```

---

### 4. `nnunet_infer.py`（推論腳本）

**功能**：使用訓練好的模型對測試集進行預測。

**原理**：

- 載入 `nnUNet_results/` 中的最佳模型權重（checkpoint）。
- 對 `imagesTs/` 中的每張測試影像進行分割預測。
- 輸出預測結果（.nii.gz 格式）於 `predictions/` 資料夾。

**執行方式**：

```bash
# Windows
python nnunet_infer.py

# macOS/Linux
python3 nnunet_infer.py
```

---

### 5. `nnunet_evaluate.py`（評估腳本）

**功能**：計算模型預測結果與真實標籤的相似度指標（Dice Coefficient、IoU）。

**原理**：

- 逐一讀取 `predictions/` 中的預測遮罩和對應的真實標籤。
- 針對每個類別（label1, label2, label3）分別計算：
  - **Dice Coefficient**：衡量預測與真實的重疊程度（範圍 0~1，越接近 1 越好）。
  - **IoU (Intersection over Union)**：交集除以聯集（範圍 0~1）。
- 輸出平均指標至終端並儲存為 CSV 報表。

**執行方式**：

```bash
# Windows
python nnunet_evaluate.py

# macOS/Linux
python3 nnunet_evaluate.py
```

**輸出範例**：

```text
========================================
評估結果
========================================
patient0001 | label1 Dice: 0.8523, IoU: 0.7834
patient0001 | label2 Dice: 0.9012, IoU: 0.8501
...
平均 Dice: 0.8634
平均 IoU: 0.7912
結果已儲存至 evaluation_results.csv
```

---

## 🔄 使用流程 {#usage}

### 完整訓練流程（從零開始）

1. **準備資料集**：
   - 將原始影像放入 `nnUNet_raw/Dataset001_CardiacSeg/imagesTr/`。
   - 將標籤放入 `nnUNet_raw/Dataset001_CardiacSeg/labelsTr/`。
   - （選配）將測試影像放入 `imagesTs/`。

2. **標準化資料集**：

   ```bash
   # Windows
   python rename_dataset.py
   
   # macOS/Linux
   python3 rename_dataset.py
   ```

3. **訓練模型**：

   ```bash
   # Windows
   python nnunet_train.py
   
   # macOS/Linux
   python3 nnunet_train.py
   ```

   - 預處理階段約需 10~30 分鐘（取決於資料集大小）。
   - 訓練階段約需數小時至數天（取決於 GPU/加速器效能和資料量）。

4. **預測測試集**：

   ```bash
   # Windows
   python nnunet_infer.py
   
   # macOS/Linux
   python3 nnunet_infer.py
   ```

5. **評估模型**：

   ```bash
   # Windows
   python nnunet_evaluate.py
   
   # macOS/Linux
   python3 nnunet_evaluate.py
   ```

---

## 📚 原理解析 {#concepts}

### nnU-Net 簡介

**nnU-Net（no-new-Net）** 是由德國癌症研究中心（DKFZ）開發的自適應醫學影像分割框架，無需手動調參即可達到頂尖效能。

**核心特點**：

- **自動配置**：根據資料集特性自動設計網路架構、資料增強策略和訓練超參數。
- **3D U-Net 架構**：編碼器（Encoder）逐層降低解析度提取特徵，解碼器（Decoder）逐層恢復解析度並結合跳躍連接（Skip Connections）保留細節。
- **5-Fold 交叉驗證**：將訓練集分為 5 份，輪流用 4 份訓練、1 份驗證，確保模型泛化能力。

### 資料格式要求

- **影像格式**：NIfTI（`.nii.gz`），支援 3D 醫學影像（如 CT、MRI）。
- **命名規則**：
  - 影像：`<case_id>_<modality>.nii.gz`（如 `patient0001_0000.nii.gz`，`0000` 表示第一個模態）。
  - 標籤：`<case_id>.nii.gz`（如 `patient0001.nii.gz`）。
- **標籤值**：
  - 背景（background）必須為 `0`。
  - 前景類別從 `1` 開始遞增（如 1, 2, 3 代表三個器官）。

### 資料預處理流程

1. **影像重新取樣**：統一所有影像的體素間距（spacing），避免不同掃描設備導致的尺寸差異。
2. **強度正規化**：根據資料集的強度分佈（如 HU 值範圍）進行 Z-score 標準化或百分位數裁剪。
3. **裁剪與填補**：將影像裁剪至包含前景的最小邊界框（Bounding Box），並填補至固定尺寸（如 128x128x128）。

### 訓練策略

- **資料增強**：隨機旋轉、縮放、彈性變形、亮度/對比度調整，增加模型對變異的魯棒性。
- **損失函數**：Dice Loss + Cross-Entropy Loss 的組合，平衡類別不平衡問題。
- **學習率調度**：多項式衰減（Polynomial Decay），初始學習率較高，訓練後期逐漸降低。

---

## ❓ 常見問題 {#faq}

### Q1：執行 `setup_environment.ps1` 時顯示「無法載入，因為這個系統上已停用指令碼執行」？

**A**：需要調整 PowerShell 執行政策：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q2：訓練過程中出現「CUDA out of memory」或記憶體不足錯誤？

**A**：GPU/記憶體不足，可嘗試：

- 在 `nnunet_train.py` 中將 `fold` 改為單一 Fold（如 `"0"`）而非 `"all"`。
- 關閉其他佔用 GPU 的程式。
- 使用較小的 Planner（如 `nnUNetPlanner` 取代 `nnUNetPlannerResEncL`）。
- **macOS 用戶**：若使用 Apple Silicon，可能需要減少 batch size（編輯 `nnUNet_results` 中的配置檔）。

### Q3：如何查看訓練進度？

**A**：訓練過程會在終端即時顯示每個 Epoch 的 loss 和 Dice 分數。亦可查看 `nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log.txt`。

### Q4：如何恢復中斷的訓練？

**A**：nnU-Net 會自動儲存檢查點（checkpoint），重新執行訓練腳本即可從上次中斷處繼續訓練：

```bash
# Windows
python nnunet_train.py

# macOS/Linux
python3 nnunet_train.py
```

### Q5：資料集太大，預處理很慢怎麼辦？

**A**：可手動指定使用更少的 CPU 核心（編輯 `nnunet_train.py` 的 `--num_processes` 參數），或減少驗證資料集完整性的檢查（移除 `--verify_dataset_integrity`）。

### Q6：macOS 上執行腳本時顯示「command not found: python」？

**A**：macOS 和 Linux 系統使用 `python3` 而非 `python`。請在所有命令中使用 `python3`：

```bash
python3 rename_dataset.py
python3 nnunet_train.py
```

### Q7：如何分享此專案給同學？

**A**：

1. 壓縮整個 `CardiacSeg` 資料夾。
2. 同學解壓縮後，根據作業系統執行對應的安裝腳本：
   - **Windows**：`.\setup_environment.ps1`
   - **macOS/Linux**：`./setup_environment.sh` 或 `pwsh setup_environment.ps1`
3. 若資料集已預處理完成，可一併分享 `nnUNet_preprocessed/` 和 `nnUNet_results/` 資料夾，節省重新訓練時間。

---

## 📖 參考資料

- **nnU-Net 論文**：[Isensee et al., Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z)
- **nnU-Net v2 官方文件**：[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **NIfTI 格式說明**：[https://nifti.nimh.nih.gov/](https://nifti.nimh.nih.gov/)

---

## 📧 聯絡資訊

如有任何問題，請聯繫專案維護者或參考 nnU-Net 官方 GitHub Issues。
