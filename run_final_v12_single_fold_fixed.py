
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
# V12 uses the Single Fold (Fold 0) probabilities, same as V8
OUTPUT_2D_PROB_DIR = BASE_DIR / 'inference_2d_test_prob'
OUTPUT_3D_LOWRES_PROB_DIR = BASE_DIR / 'inference_3d_lowres_test_prob'

SUBMISSION_DIR = BASE_DIR / 'submission_final_v12_single_fold_fixed'
ZIP_FILE = BASE_DIR / 'submission_final_v12_20251126.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

# Soft Voting Weights (V12: V8 Strategy + RV Fix)
# Strategy: 
# 1. Use Single Fold (Fold 0) because 5-Fold Ensemble (V9/V11) performed worse (0.7870 vs 0.7905).
# 2. Use V8 weights for Myo/LV (where 3D is strong).
# 3. Use 100% 2D for RV (Label 3) because 3D has 0.0 Dice in validation.
WEIGHTS = {
    0: {'2d': 0.5, '3d_low': 0.5}, # Background
    1: {'2d': 0.3, '3d_low': 0.7}, # Myocardium (V8 weights)
    2: {'2d': 0.3, '3d_low': 0.7}, # LV (V8 weights)
    3: {'2d': 1.0, '3d_low': 0.0}  # RV (Fix: Trust 2D completely)
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def postprocess_segmentation(segmentation):
    """
    Applies LCC, Fill Holes, and Morphological Closing.
    """
    cleaned_seg = np.zeros_like(segmentation)
    
    # Process each class
    for class_idx in [1, 2, 3]:
        class_mask = (segmentation == class_idx)
        
        if not np.any(class_mask):
            continue
            
        # 1. Keep Largest Connected Component
        labeled_mask, num_features = label_cc(class_mask)
        if num_features > 1:
            sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
            largest_component_label = np.argmax(sizes) + 1
            class_mask = (labeled_mask == largest_component_label)
            
        # 2. Fill Holes
        # Fill holes in each slice to avoid 3D topology issues
        for i in range(class_mask.shape[2]):
            class_mask[:, :, i] = binary_fill_holes(class_mask[:, :, i])
            
        # 3. Morphological Closing (smooth boundaries)
        class_mask = binary_closing(class_mask, iterations=1)
        
        cleaned_seg[class_mask] = class_idx
        
    return cleaned_seg

def run_ensemble():
    print(f"Running V12 Ensemble (Single Fold + RV Fix)...")
    print(f"Weights: {WEIGHTS}")
    
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    test_cases = get_test_cases()
    print(f"Found {len(test_cases)} test cases.")
    
    for case_id in test_cases:
        print(f"Processing {case_id}...")
        
        # Load probabilities
        # Note: nnUNet saves probabilities as filename.nii.gz.npz
        prob_2d_file = OUTPUT_2D_PROB_DIR / f"{case_id}.nii.gz.npz"
        prob_3d_file = OUTPUT_3D_LOWRES_PROB_DIR / f"{case_id}.nii.gz.npz"
        
        if not prob_2d_file.exists():
             # Try without .nii.gz just in case (some versions differ)
             prob_2d_file = OUTPUT_2D_PROB_DIR / f"{case_id}.npz"

        if not prob_3d_file.exists():
             prob_3d_file = OUTPUT_3D_LOWRES_PROB_DIR / f"{case_id}.npz"
        
        if not prob_2d_file.exists() or not prob_3d_file.exists():
            print(f"Missing probabilities for {case_id}, skipping.")
            print(f"Checked: {prob_2d_file} and {prob_3d_file}")
            continue
            
        prob_2d = np.load(prob_2d_file)['probabilities']
        prob_3d = np.load(prob_3d_file)['probabilities']
        
        # Ensemble
        num_classes, d, h, w = prob_2d.shape
        ensemble_prob = np.zeros_like(prob_2d)
        
        for c in range(num_classes):
            w_2d = WEIGHTS[c]['2d']
            w_3d = WEIGHTS[c]['3d_low']
            ensemble_prob[c] = (prob_2d[c] * w_2d) + (prob_3d[c] * w_3d)
            
        # Argmax
        final_seg = np.argmax(ensemble_prob, axis=0).astype(np.uint8)
        
        # Post-processing
        final_seg = postprocess_segmentation(final_seg)
        
        # Save
        img_nib = nib.load(os.path.join(IMAGES_TS_DIR, f"{case_id}_0000.nii.gz"))
        
        # Transpose Check (Critical Fix)
        # final_seg is (Z, Y, X) from npz, img_nib is (X, Y, Z)
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
