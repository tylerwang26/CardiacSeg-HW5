# CardiacSeg Submission Analysis Report (V4 - V17)

## 1. Strategy Overview (Extracted from `monitor_training_live.ps1`)

| Version | Strategy Description | Score |
| :--- | :--- | :--- |
| **V4** | 2D(F0) + 3D Low(F0) \| Weights: Myo/LV(3D:0.7), RV(2D:0.65) \| Closing: No \| Fill: No \| TTA: Yes \| LCC: No \| Voting: Hard | 0.78303919 |
| **V5** | 2D(F0) + 3D Low(F0) \| Weights: Myo/LV(3D:0.7), RV(2D:0.65) \| Closing: No \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Hard | 0.78330995 |
| **V6** | 2D(F0) + 3D Low(F0) \| Weights: Myo/LV(3D:0.7), RV(2D:0.65) \| Closing: No \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.79010228 |
| **V7** | 2D(F0) + 3D Low(F0) + 3D Full(F0+F1) \| Weights: Soft Voting \| Closing: No \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.78984334 |
| **V8** | 2D(F0) + 3D Low(F0) \| Weights: Myo/LV(3D:0.7), RV(2D:0.65) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.79051468 |
| **V9** | 2D(F0) + 3D Low(5-Fold) \| Weights: Myo/LV(3D:0.7), RV(2D:0.65) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.78705121 |
| **V10** | 2D(F0) + 3D Low(5-Fold) \| Weights: Myo(3D:0.8), LV(3D:0.6), RV(2D:1.0) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.78730098 |
| **V11** | 2D(F0) + 3D Low(5-Fold) \| Weights: Myo/LV(3D:0.7), RV(2D:1.0) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.78702595 |
| **V12** | 2D(F0) + 3D Low(F0) \| Weights: Myo/LV(3D:0.7), RV(2D:1.0) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | 0.79048839 |
| **V13** | 2D(F0) + 3D Low(F0) \| Weights: Myo(2D:0.6), LV(3D:0.7), RV(2D:1.0) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | TBD |
| **V14** | 2D(F0) + 3D Full(F0) \| Weights: Myo/LV(Full:0.7), RV(2D:1.0)\| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | TBD |
| **V15** | 2D(F0) + 3D Low(5-Fold) \| Weights: Myo/LV(3D:0.7), RV(2D:0.65) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | TBD |
| **V16** | 2D(F0) + 3D Low(5-Fold) + 3D Full(F0+F1) \| Weights: Myo/LV(Full:0.6), RV(2D:0.7) \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | TBD |
| **V17** | 2D(5-Fold) + 3D Low(5-Fold) + 3D Full(5-Fold) \| Weights: TBD \| Closing: Yes \| Fill: Yes \| TTA: Yes \| LCC: Yes \| Voting: Soft | TBD |

## 2. Detailed Script Analysis & Verification

| Version | Script File | TTA | Voting | Closing | Fill Holes | LCC | Transpose Check (D,H,W)->(H,W,D) | Submission Zip File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V4** | `run_optimized_submission_v4.py` | Yes | Hard | No | No | No | ⚠️ No Explicit Check | `submission_optimized_v4_20251122.zip` |
| **V5** | `run_postprocessing_v5.py` | Yes* | Hard* | No | Yes | Yes | ⚠️ No Explicit Check | `submission_optimized_v5_20251122.zip` |
| **V6** | `run_optimized_submission_v6.py` | Yes | Soft | No | Yes | Yes | ✅ Yes | `submission_optimized_v6_20251122.zip` |
| **V7** | `run_optimized_submission_v7.py` | Yes | Soft | No | Yes | Yes | ✅ Yes | `submission_optimized_v7_20251123.zip` |
| **V8** | `run_optimized_submission_v8.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_optimized_v8_20251123.zip` |
| **V9** | `run_final_v9_ensemble.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v9_20251125.zip` |
| **V10** | `run_final_v10_ensemble.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v10_20251125.zip` |
| **V11** | `run_final_v11_hybrid.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v11_20251125.zip` |
| **V12** | `run_final_v12_single_fold_fixed.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v12_20251126.zip` |
| **V13** | `run_final_v13_optimized_weights.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v13_20251126.zip` |
| **V14** | `run_final_v14_fullres_hybrid.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v14_20251126.zip` |
| **V15** | `run_v15_submission.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v15_20251126.zip` |
| **V16** | `run_v16_submission.py` | Yes | Soft | Yes | Yes | Yes | ✅ Yes | `submission_final_v16_20251126.zip` |
| **V17** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*\* V5 inherits TTA/Voting from V4 results.*


### Key Findings

1. **Transposition Issue**: V4 and V5 do **not** have explicit code to check for `(D, H, W)` vs `(H, W, D)` mismatch. They rely on `nibabel` preserving the affine and shape of the input/reference image. If the reference image (from `imagesTs`) is correct, the output is likely correct, but there is no safety check like in V6+.
2. **Post-Processing Evolution**:
    * **V4**: Raw ensemble.
    * **V5**: Added Fill Holes & LCC.
    * **V6**: Switched to Soft Voting.
    * **V8**: Added Morphological Closing (kernel 3x3x3). This configuration (Soft Voting + Closing + Fill + LCC) became the standard for all subsequent versions.
3. **V17 Status**: Currently a placeholder in the monitoring script; no corresponding python script was found.


