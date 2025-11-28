
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import zipfile
import glob
from scipy.ndimage import label as label_cc
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_closing

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'

# Input Probabilities
PROB_2D_DIR = BASE_DIR / 'inference_2d_test_prob'
PROB_3D_LOWRES_5FOLD_DIR = BASE_DIR / 'inference_3d_lowres_5fold_prob'
PROB_3D_FULLRES_F0_DIR = BASE_DIR / 'inference_3d_fullres_f0_prob'
PROB_3D_FULLRES_F1_DIR = BASE_DIR / 'inference_3d_fullres_f1_prob'

SUBMISSION_DIR = BASE_DIR / 'submission_final_v16'
ZIP_FILE = BASE_DIR / 'submission_final_v16_20251126.zip'

# Weights
# Fullres = (F0 + F1) / 2
# Lowres = 5-Fold
# 2D = Fold 0
WEIGHTS = {
    0: {'2d': 0.4, 'low': 0.3, 'full': 0.3},
    1: {'2d': 0.2, 'low': 0.2, 'full': 0.6}, # Myo: Fullres is best
    2: {'2d': 0.2, 'low': 0.2, 'full': 0.6}, # LV: Fullres is best
    3: {'2d': 0.7, 'low': 0.3, 'full': 0.0}  # RV: Trust 2D, Lowres helps slightly. Fullres is 0.
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def postprocess_segmentation(segmentation):
    """
    V8 Post-processing
    """
    cleaned_seg = np.zeros_like(segmentation)
    
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        mask = binary_closing(mask, structure=np.ones((3,3,3)))
            
        labeled_mask, num_features = label_cc(mask)
        if num_features > 1:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component = component_sizes.argmax()
            mask = (labeled_mask == largest_component)
            
        if label_id in [2, 3]:
            mask = binary_fill_holes(mask)
            
        cleaned_seg[mask] = label_id
        
    return cleaned_seg

def load_prob(base_dir, case):
    p_path = base_dir / f"{case}.nii.gz.npz"
    if not p_path.exists(): p_path = base_dir / f"{case}.npz"
    if not p_path.exists(): return None
    return np.load(str(p_path))['probabilities']

def run_ensemble():
    print(f"=== Running V16 Ensemble ===")
    print("Components: 3D Fullres (F0+F1) + 3D Lowres (5-Fold) + 2D (Fold 0)")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    cases = get_test_cases()
    
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Processing {case}...")
        
        p2d = load_prob(PROB_2D_DIR, case)
        plow = load_prob(PROB_3D_LOWRES_5FOLD_DIR, case)
        pfull0 = load_prob(PROB_3D_FULLRES_F0_DIR, case)
        pfull1 = load_prob(PROB_3D_FULLRES_F1_DIR, case)
        
        if any(x is None for x in [p2d, plow, pfull0, pfull1]):
            print(f"Missing probabilities for {case}")
            continue
            
        # Average Fullres
        pfull = (pfull0 + pfull1) / 2.0
        
        # Ensemble
        num_classes = p2d.shape[0]
        ensemble_prob = np.zeros_like(p2d)
        
        for c in range(num_classes):
            w2d = WEIGHTS[c]['2d']
            wlow = WEIGHTS[c]['low']
            wfull = WEIGHTS[c]['full']
            ensemble_prob[c] = w2d * p2d[c] + wlow * plow[c] + wfull * pfull[c]
            
        seg = np.argmax(ensemble_prob, axis=0)
        seg = postprocess_segmentation(seg)
        
        img_path = IMAGES_TS_DIR / f"{case}_0000.nii.gz"
        nii = nib.load(str(img_path))
        
        # Transpose Check (Critical Fix from V8)
        if seg.shape != nii.shape:
            print(f"Transposing {case} from {seg.shape} to {nii.shape}")
            seg = seg.transpose(2, 1, 0)
            
        seg_nii = nib.Nifti1Image(seg.astype(np.uint8), nii.affine, nii.header)
        nib.save(seg_nii, str(SUBMISSION_DIR / f"{case}.nii.gz"))

    print(f"Creating Zip: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in SUBMISSION_DIR.glob('*.nii.gz'):
            zf.write(f, arcname=f.name)
    print("Done.")

if __name__ == '__main__':
    run_ensemble()
