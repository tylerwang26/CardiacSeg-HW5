
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
PROB_2D_DIR = BASE_DIR / 'inference_2d_test_prob' # Fold 0
PROB_3D_LOWRES_5FOLD_DIR = BASE_DIR / 'inference_3d_lowres_5fold_prob' # 5-Fold

SUBMISSION_DIR = BASE_DIR / 'submission_final_v15'
ZIP_FILE = BASE_DIR / 'submission_final_v15_20251126.zip'

# Weights (Based on V8, but using 5-Fold 3D)
WEIGHTS = {
    0: {'2d': 0.5, '3d': 0.5},
    1: {'2d': 0.3, '3d': 0.7}, # Myo: Trust 3D (5-Fold)
    2: {'2d': 0.3, '3d': 0.7}, # LV: Trust 3D (5-Fold)
    3: {'2d': 0.65, '3d': 0.35} # RV: Trust 2D (Fold 0) more
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def postprocess_segmentation(segmentation):
    """
    V8 Post-processing: LCC + Fill Holes + Closing
    """
    cleaned_seg = np.zeros_like(segmentation)
    
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        # 1. Morphological Closing
        mask = binary_closing(mask, structure=np.ones((3,3,3)))
            
        # 2. Keep Largest Connected Component
        labeled_mask, num_features = label_cc(mask)
        if num_features > 1:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component = component_sizes.argmax()
            mask = (labeled_mask == largest_component)
            
        # 3. Fill Holes (Only for LV=2 and RV=3)
        if label_id in [2, 3]:
            mask = binary_fill_holes(mask)
            
        cleaned_seg[mask] = label_id
        
    return cleaned_seg

def run_ensemble():
    print(f"=== Running V15 Ensemble (3D Lowres 5-Fold + 2D Fold 0) ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    cases = get_test_cases()
    
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Processing {case}...")
        
        # Load 2D
        p2d_path = PROB_2D_DIR / f"{case}.nii.gz.npz"
        if not p2d_path.exists(): p2d_path = PROB_2D_DIR / f"{case}.npz"
        
        # Load 3D
        p3d_path = PROB_3D_LOWRES_5FOLD_DIR / f"{case}.nii.gz.npz"
        if not p3d_path.exists(): p3d_path = PROB_3D_LOWRES_5FOLD_DIR / f"{case}.npz"
        
        if not (p2d_path.exists() and p3d_path.exists()):
            print(f"Missing probabilities for {case}")
            continue
            
        prob_2d = np.load(str(p2d_path))['probabilities']
        prob_3d = np.load(str(p3d_path))['probabilities']
        
        # Load geometry
        img_path = IMAGES_TS_DIR / f"{case}_0000.nii.gz"
        nii = nib.load(str(img_path))
        
        # Ensemble
        num_classes = prob_2d.shape[0]
        ensemble_prob = np.zeros_like(prob_2d)
        
        for c in range(num_classes):
            w2d = WEIGHTS[c]['2d']
            w3d = WEIGHTS[c]['3d']
            ensemble_prob[c] = w2d * prob_2d[c] + w3d * prob_3d[c]
            
        # Argmax
        seg = np.argmax(ensemble_prob, axis=0)
        
        # Post-processing
        seg = postprocess_segmentation(seg)
        
        # Transpose Check (Critical Fix from V8)
        # seg is (Z, Y, X) from npz, ref_img is (X, Y, Z)
        if seg.shape != nii.shape:
            print(f"Transposing {case} from {seg.shape} to {nii.shape}")
            seg = seg.transpose(2, 1, 0)

        # Save
        seg_nii = nib.Nifti1Image(seg.astype(np.uint8), nii.affine, nii.header)
        nib.save(seg_nii, str(SUBMISSION_DIR / f"{case}.nii.gz"))

    # Zip
    print(f"Creating Zip: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in SUBMISSION_DIR.glob('*.nii.gz'):
            zf.write(f, arcname=f.name)
    print("Done.")

if __name__ == '__main__':
    run_ensemble()
