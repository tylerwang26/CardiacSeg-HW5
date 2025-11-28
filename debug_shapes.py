
import numpy as np
import nibabel as nib
import os

prob_path = r"C:\CardiacSeg\inference_2d_test_prob\patient0051.nii.gz.npz"
img_path = r"C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg\imagesTs\patient0051_0000.nii.gz"

print(f"Checking shapes for patient0051...")

# Check Original Image
img = nib.load(img_path)
print(f"Original Image Shape (nibabel): {img.shape}")
print(f"Original Image Affine:\n{img.affine}")

# Check Probabilities
data = np.load(prob_path)
probs = data['probabilities']
print(f"Probabilities Shape (numpy): {probs.shape}")

# Check if transposition is needed
if img.shape == probs.shape[1:]:
    print("Shapes match (ignoring channel dim).")
else:
    print("Shapes DO NOT match.")
    print(f"Expected (from img): {img.shape}")
    print(f"Got (from probs): {probs.shape[1:]}")

