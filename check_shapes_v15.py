
import numpy as np
import nibabel as nib
import os

case = "patient0051"
img_path = fr"C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg\imagesTs\{case}_0000.nii.gz"
p2d_path = fr"C:\CardiacSeg\inference_2d_test_prob\{case}.nii.gz.npz"
p3d_path = fr"C:\CardiacSeg\inference_3d_lowres_5fold_prob\{case}.nii.gz.npz"

nii = nib.load(img_path)
print(f"Original Image Shape: {nii.shape}")

if os.path.exists(p2d_path):
    p2d = np.load(p2d_path)['probabilities']
    print(f"2D Prob Shape: {p2d.shape}")
else:
    print("2D Prob not found")

if os.path.exists(p3d_path):
    p3d = np.load(p3d_path)['probabilities']
    print(f"3D Prob Shape: {p3d.shape}")
else:
    print("3D Prob not found")
