# Ensemble Test Results (2025-11-20)

## 1. Execution Summary
- **3D Lowres Inference**: Completed on 10 validation cases (`inference_3d_lowres_validation`).
- **2D Inference**: Completed on 10 validation cases (`inference_2d_validation_final`).
- **Ensemble**: Executed using `label_specific` weights (L1: 3D=0.7, L2: 3D=0.6, L3: 2D=0.8).

## 2. Evaluation Results (Mean Dice)

| Case | Ensemble Dice | L1 (Myo) | L2 (LV) | L3 (RV) |
|------|---------------|----------|---------|---------|
| patient0009 | 0.5467 | 0.9026 | 0.7375 | 0.0000 |
| patient0013 | **0.8081** | 0.9369 | 0.7962 | 0.6911 |
| patient0022 | 0.5667 | 0.9357 | 0.7644 | 0.0000 |
| patient0023 | 0.5683 | 0.9355 | 0.7694 | 0.0000 |
| patient0031 | 0.5383 | 0.8882 | 0.7266 | 0.0000 |
| patient0033 | 0.5377 | 0.8854 | 0.7278 | 0.0000 |
| patient0034 | 0.5564 | 0.9258 | 0.7435 | 0.0000 |
| patient0039 | 0.5590 | 0.9103 | 0.7668 | 0.0000 |
| patient0043 | 0.5061 | 0.7244 | 0.7939 | 0.0000 |
| patient0046 | 0.5774 | 0.9209 | 0.8114 | 0.0000 |
| **Average** | **0.5765** | **0.8966** | **0.7638** | **0.0691** |

## 3. Comparison
- **3D Lowres Only**: ~0.5545 (from training log)
- **Ensemble**: 0.5765 (+0.022)

## 4. Observations
- **Label 3 (Right Ventricle)**: Still a major challenge. Only detected in `patient0013`.
- **Label 1 & 2**: Very high accuracy (L1 ~0.90, L2 ~0.76).
- **Improvement**: Ensemble provides a modest boost, primarily by refining L1/L2 or slightly helping L3 in one case.

## 5. Next Steps
- Investigate why 2D model missed L3 in 9/10 cases (check 2D individual performance).
- Consider **3D Fullres** training to improve resolution for small structures (L3).
- Submit current Ensemble result to leaderboard to get test set feedback.
