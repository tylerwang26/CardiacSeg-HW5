
import numpy as np
import nibabel as nib
from pathlib import Path

BASE_DIR = Path(r'C:\CardiacSeg')
OUTPUT_3D_PROB_DIR = BASE_DIR / 'inference_3d_lowres_test_prob'
case = "patient0051"

npz_path = OUTPUT_3D_PROB_DIR / f"{case}.nii.gz.npz"
nii_path = OUTPUT_3D_PROB_DIR / f"{case}.nii.gz"

print(f"Checking {case}...")

if npz_path.exists():
    data = np.load(str(npz_path))
    probs = data['probabilities']
    print(f"Probabilities shape: {probs.shape}")
else:
    print("NPZ not found")

if nii_path.exists():
    img = nib.load(str(nii_path))
    print(f"NIfTI shape: {img.shape}")
    print(f"NIfTI affine:\n{img.affine}")
else:
    print("NIfTI not found")
