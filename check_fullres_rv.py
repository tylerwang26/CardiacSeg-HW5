
import numpy as np
import os

# Paths
prob_fullres_path = r"C:\CardiacSeg\inference_3d_fullres_f0_prob\patient0051.nii.gz.npz"

print("Checking Fullres RV (Label 3) signals...")

if os.path.exists(prob_fullres_path):
    data = np.load(prob_fullres_path)['probabilities']
    # Check shape
    print(f"Shape: {data.shape}")
    
    rv_prob = data[3]
    print(f"Fullres RV Max Prob: {np.max(rv_prob)}")
    print(f"Fullres RV Mean Prob: {np.mean(rv_prob)}")
    print(f"Fullres RV > 0.5 count: {np.sum(rv_prob > 0.5)}")
    
    # Also check Myo (1) and LV (2) to compare confidence
    print(f"Fullres Myo (1) Max: {np.max(data[1])}")
    print(f"Fullres LV (2) Max: {np.max(data[2])}")
    
else:
    print(f"Fullres prob not found at {prob_fullres_path}")
