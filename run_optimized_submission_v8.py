
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

# Output Dirs for Probabilities (Reuse existing)
OUTPUT_2D_PROB_DIR = BASE_DIR / 'inference_2d_test_prob'
OUTPUT_3D_LOWRES_PROB_DIR = BASE_DIR / 'inference_3d_lowres_test_prob'

SUBMISSION_DIR = BASE_DIR / 'submission_optimized_v8'
ZIP_FILE = BASE_DIR / 'submission_optimized_v8_20251123.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

# Soft Voting Weights (V8: Refined 2-Model Ensemble)
# Strategy: Trust 3D Lowres for volume (LV/Myo), Trust 2D for shape/slice details (RV).
WEIGHTS = {
    0: {'2d': 0.5, '3d_low': 0.5}, # Background
    1: {'2d': 0.3, '3d_low': 0.7}, # Myocardium: 3D is better for continuity
    2: {'2d': 0.3, '3d_low': 0.7}, # LV: 3D is better for volume
    3: {'2d': 0.65, '3d_low': 0.35} # RV: 2D is better for complex shape
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
    
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        # 1. Morphological Closing (New in V8)
        # Smooths boundaries and fills small gaps
        # structure=np.ones((3,3,3)) defines connectivity
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

def soft_voting_ensemble(cases):
    print("\n=== Running Soft Voting Ensemble (V8: 2-Model Refined) ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Ensembling {case}...")
        
        # Paths
        p_2d = OUTPUT_2D_PROB_DIR / f"{case}.nii.gz.npz"
        p_low = OUTPUT_3D_LOWRES_PROB_DIR / f"{case}.nii.gz.npz"
        
        # Check existence
        if not (p_2d.exists() and p_low.exists()):
            print(f"Error: Missing probabilities for {case}")
            continue
            
        # Load probabilities
        prob_2d = np.load(str(p_2d))['probabilities']
        prob_low = np.load(str(p_low))['probabilities']
        
        # Load reference image for affine
        ref_img_path = OUTPUT_3D_LOWRES_PROB_DIR / f"{case}.nii.gz"
        ref_img = nib.load(str(ref_img_path))
        
        # Initialize final probability map
        final_prob = np.zeros_like(prob_2d)
        
        # Weighted Average
        for label_id in [0, 1, 2, 3]:
            w_2d = WEIGHTS[label_id]['2d']
            w_low = WEIGHTS[label_id]['3d_low']
            
            term_2d = prob_2d[label_id] * w_2d
            term_low = prob_low[label_id] * w_low
            
            final_prob[label_id] = term_2d + term_low
            
        # Argmax
        final_seg = np.argmax(final_prob, axis=0).astype(np.uint8)
        
        # Post-processing
        final_seg = postprocess_segmentation(final_seg)
        
        # Transpose Check (Critical Fix)
        # final_seg is (Z, Y, X) from npz
        # ref_img is (X, Y, Z)
        if final_seg.shape != ref_img.shape:
            # print(f"Transposing {case} from {final_seg.shape} to {ref_img.shape}")
            final_seg = final_seg.transpose(2, 1, 0)
        
        # Save
        out_img = nib.Nifti1Image(final_seg, ref_img.affine, ref_img.header)
        nib.save(out_img, str(SUBMISSION_DIR / f"{case}.nii.gz"))

def create_submission_zip(cases):
    print(f"\n=== Creating Submission Zip: {ZIP_FILE} ===")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for case in cases:
            file_path = SUBMISSION_DIR / f"{case}.nii.gz"
            if file_path.exists():
                zf.write(file_path, f"{case}.nii.gz")
    print("Done!")

if __name__ == "__main__":
    cases = get_test_cases()
    
    # No new inference needed, reusing V6 probabilities
    
    # 1. Soft Voting
    soft_voting_ensemble(cases)
    
    # 2. Zip
    create_submission_zip(cases)
