
import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import zipfile
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join
import glob
from scipy.ndimage import label as label_cc
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_closing

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'

# Output Dirs for Probabilities
OUTPUT_2D_PROB_DIR = BASE_DIR / 'inference_2d_test_prob'
# SWITCH TO FULLRES FOLD 0
OUTPUT_3D_FULLRES_PROB_DIR = BASE_DIR / 'inference_3d_fullres_f0_prob'

SUBMISSION_DIR = BASE_DIR / 'submission_final_v14_fullres_hybrid'
ZIP_FILE = BASE_DIR / 'submission_final_v14_20251126.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

# Soft Voting Weights (V14: Fullres Superiority)
# Validation Stats (Fold 0):
# Myo (1): Fullres (0.922) > 2D (0.911) > Lowres (0.909)
# LV (2):  Fullres (0.765) > Lowres (0.755) > 2D (0.737)
# RV (3):  All ~0.0
#
# Strategy:
# Use Fullres as the 3D backbone instead of Lowres.
WEIGHTS = {
    0: {'2d': 0.5, '3d_full': 0.5}, # Background
    1: {'2d': 0.3, '3d_full': 0.7}, # Myocardium: Fullres is best (0.922). Trust it.
    2: {'2d': 0.3, '3d_full': 0.7}, # LV: Fullres is best (0.765). Trust it.
    3: {'2d': 1.0, '3d_full': 0.0}  # RV: Trust 2D (Fullres also has 0.0 signal)
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def postprocess_segmentation(segmentation):
    """
    Applies LCC, Fill Holes, and Morphological Closing.
    Matches V8 implementation exactly (3D operations).
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
    print(f"Running V14 Ensemble (Fullres Fold 0 + 2D)...")
    print(f"Weights: {WEIGHTS}")
    
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    test_cases = get_test_cases()
    print(f"Found {len(test_cases)} test cases.")
    
    for case_id in test_cases:
        print(f"Processing {case_id}...")
        
        # Load probabilities
        prob_2d_file = OUTPUT_2D_PROB_DIR / f"{case_id}.nii.gz.npz"
        prob_3d_file = OUTPUT_3D_FULLRES_PROB_DIR / f"{case_id}.nii.gz.npz"
        
        if not prob_2d_file.exists(): prob_2d_file = OUTPUT_2D_PROB_DIR / f"{case_id}.npz"
        if not prob_3d_file.exists(): prob_3d_file = OUTPUT_3D_FULLRES_PROB_DIR / f"{case_id}.npz"
        
        if not prob_2d_file.exists() or not prob_3d_file.exists():
            print(f"Missing probabilities for {case_id}, skipping.")
            continue
            
        prob_2d = np.load(prob_2d_file)['probabilities']
        prob_3d = np.load(prob_3d_file)['probabilities']
        
        # Ensemble
        num_classes, d, h, w = prob_2d.shape
        ensemble_prob = np.zeros_like(prob_2d)
        
        for c in range(num_classes):
            w_2d = WEIGHTS[c]['2d']
            w_3d = WEIGHTS[c]['3d_full']
            ensemble_prob[c] = (prob_2d[c] * w_2d) + (prob_3d[c] * w_3d)
            
        # Argmax
        final_seg = np.argmax(ensemble_prob, axis=0).astype(np.uint8)
        
        # Post-processing
        final_seg = postprocess_segmentation(final_seg)
        
        # Save
        img_nib = nib.load(os.path.join(IMAGES_TS_DIR, f"{case_id}_0000.nii.gz"))
        
        # Transpose Check
        if final_seg.shape != img_nib.shape:
            print(f"Transposing {case_id} from {final_seg.shape} to {img_nib.shape}")
            final_seg = final_seg.transpose(2, 1, 0)
            
        nifti_img = nib.Nifti1Image(final_seg, img_nib.affine, img_nib.header)
        nib.save(nifti_img, SUBMISSION_DIR / f"{case_id}.nii.gz")
        
    print("Ensemble completed.")
    
    # Create Zip
    print(f"Creating submission zip: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in SUBMISSION_DIR.glob('*.nii.gz'):
            zf.write(file_path, arcname=file_path.name)
    print("Done!")

if __name__ == "__main__":
    run_ensemble()
