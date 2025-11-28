
import nibabel as nib
import numpy as np
import os

# Paths
img_ts_path = r"C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg\imagesTs\patient0051_0000.nii.gz"
prob_3d_ref_path = r"C:\CardiacSeg\inference_3d_lowres_test_prob\patient0051.nii.gz"
prob_3d_ensemble_ref_path = r"C:\CardiacSeg\inference_3d_lowres_ensemble_test_prob\patient0051.nii.gz"

print("Checking Affines...")

# 1. Original Image
if os.path.exists(img_ts_path):
    img = nib.load(img_ts_path)
    print(f"\nOriginal Image (imagesTs):")
    print(f"Shape: {img.shape}")
    print(f"Affine:\n{img.affine}")
else:
    print(f"Original image not found at {img_ts_path}")

# 2. 3D Lowres Prob Ref (Used in V8)
if os.path.exists(prob_3d_ref_path):
    ref_v8 = nib.load(prob_3d_ref_path)
    print(f"\n3D Lowres Ref (V8 used this):")
    print(f"Shape: {ref_v8.shape}")
    print(f"Affine:\n{ref_v8.affine}")
else:
    print(f"3D Lowres Ref not found at {prob_3d_ref_path}")

# 3. 3D Ensemble Ref (Used in V10)
if os.path.exists(prob_3d_ensemble_ref_path):
    ref_v10 = nib.load(prob_3d_ensemble_ref_path)
    print(f"\n3D Ensemble Ref (V10 used this):")
    print(f"Shape: {ref_v10.shape}")
    print(f"Affine:\n{ref_v10.affine}")
else:
    print(f"3D Ensemble Ref not found at {prob_3d_ensemble_ref_path}")
