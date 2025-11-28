
import numpy as np
import os

# Paths
prob_2d_path = r"C:\CardiacSeg\inference_2d_test_prob\patient0051.nii.gz.npz"
prob_3d_path = r"C:\CardiacSeg\inference_3d_lowres_test_prob\patient0051.nii.gz.npz"

print("Checking RV (Label 3) signals...")

# 2D
if os.path.exists(prob_2d_path):
    data = np.load(prob_2d_path)['probabilities']
    rv_prob = data[3]
    print(f"2D RV Max Prob: {np.max(rv_prob)}")
    print(f"2D RV Mean Prob: {np.mean(rv_prob)}")
    print(f"2D RV > 0.5 count: {np.sum(rv_prob > 0.5)}")
else:
    print("2D prob not found")

# 3D
if os.path.exists(prob_3d_path):
    data = np.load(prob_3d_path)['probabilities']
    rv_prob = data[3]
    print(f"3D RV Max Prob: {np.max(rv_prob)}")
    print(f"3D RV Mean Prob: {np.mean(rv_prob)}")
    print(f"3D RV > 0.5 count: {np.sum(rv_prob > 0.5)}")
else:
    print("3D prob not found")
