import nibabel as nib
import numpy as np
import glob
import os
from pathlib import Path

base_dir = Path(r'C:\CardiacSeg')
dir_2d = base_dir / 'inference_2d_test_tta'
dir_3d = base_dir / 'inference_3d_fullres_test'
dir_ensemble_raw = base_dir / 'temp_ensemble_raw'
dir_v2 = base_dir / 'submission_optimized'
dir_v3 = base_dir / 'submission_optimized_v3'

cases = sorted([os.path.basename(f).replace(".nii.gz", "") for f in glob.glob(str(dir_3d / "*.nii.gz"))])

print(f"{'Case':<15} {'Vol 2D':<15} {'Vol 3D F0':<15} {'Vol V2':<15} {'Vol V3':<15}")

for case in cases[:10]: # Check first 10
    p2 = dir_2d / f"{case}.nii.gz"
    p3 = dir_3d / f"{case}.nii.gz"
    pv2 = dir_v2 / f"{case}.nii.gz"
    pv3 = dir_v3 / f"{case}.nii.gz"
    
    v2 = np.sum(nib.load(p2).get_fdata() == 3) if p2.exists() else 0
    v3 = np.sum(nib.load(p3).get_fdata() == 3) if p3.exists() else 0
    vv2 = np.sum(nib.load(pv2).get_fdata() == 3) if pv2.exists() else 0
    vv3 = np.sum(nib.load(pv3).get_fdata() == 3) if pv3.exists() else 0
    
    print(f"{case:<15} {v2:<15} {v3:<15} {vv2:<15} {vv3:<15}")
