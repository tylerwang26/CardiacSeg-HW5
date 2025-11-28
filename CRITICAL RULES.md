# CRITICAL RULES
1. nnU-Net輸出(Depth, Height, Width)必須 Transpose to 比賽要求的 (Height, Width, epth)  //nnU-Net 的輸出通常是 (D, H, W)，在儲存成 NIfTI 檔案之前，必須執行 transpose(2, 1, 0) 或類似的操作來還原成原始影像的座標系

ex:         # Transpose Check (Critical Fix)
        # final_seg is (Z, Y, X) from npz
        # ref_img is (X, Y, Z)
        if final_seg.shape != ref_img.shape:
            # print(f"Transposing {case} from {final_seg.shape} to {ref_img.shape}")
            final_seg = final_seg.transpose(2, 1, 0)

---

# 生成規格

## v8

2D(F0) + 3D Low(F0) | Weights: Myo/LV(3D:0.7), RV(2D:0.65) | Closing: Yes | Fill: Yes | TTA: Yes | LCC: Yes | Voting: Soft	0.79051468

## V16

生成V16 submmision file, 規格為:
2D: 5-Fold Ensemble (Folds 0-4).
3D Lowres: 5-Fold Ensemble (Folds 0-4).
3D Fullres: 2-Fold Ensemble (Folds 0+1).
Weights:
Myo/LV: Fullres (0.7) + Lowres (0.3) + 2D (0.0). (Trust Fullres heavily).
RV: 2D (0.65) + Lowres (0.35) + Fullres (0.0). (Trust 2D heavily).
Post-processing: V8 Standard (Closing ⭢ LCC ⭢ Fill).
Transpose Check: MANDATORY.

### 結果

V16 Submission 檔案已成功生成。

檔案位置:
submission_final_v16_updated_20251128.zip

執行細節:

Ensemble Components:

2D: 5-Fold Ensemble (Folds 0-4) [剛剛生成]
3D Lowres: 5-Fold Ensemble (Folds 0-4) [使用 11/26 生成的概率圖，確認新於 11/23 Checkpoint]
3D Fullres: 2-Fold Ensemble (Folds 0+1) [使用現有概率圖]
Weights (已確認):

Myo (1) / LV (2): Fullres (0.7) + Lowres (0.3) + 2D (0.0)
RV (3): 2D (0.65) + Lowres (0.35) + Fullres (0.0)
Post-processing:

V8 Standard (Closing ⭢ LCC ⭢ Fill) 已應用。
Transpose Check:

MANDATORY CHECK ACTIVE: 所有 50 個案例均觸發了轉置修正 (例如: `Transposing patient005
1 from (386, 512, 512) to (512, 512, 386)`)，確保了最終輸出維度正確 (X, Y, Z)。