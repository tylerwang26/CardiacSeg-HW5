
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import zipfile
from scipy.ndimage import label as label_cc
import glob

# Paths
BASE_DIR = Path(r'C:\CardiacSeg')
DIR_2D = BASE_DIR / 'inference_2d_test_tta'
DIR_3D_F0 = BASE_DIR / 'inference_3d_fullres_test'
DIR_3D_F1 = BASE_DIR / 'inference_3d_fullres_test_fold1'
SUBMISSION_DIR = BASE_DIR / 'submission_optimized_v3'
ZIP_FILE = BASE_DIR / 'submission_optimized_v3_20251122.zip'

# Weights Configuration
# Format: {Label_ID: {Model_Name: Weight}}
# Models: '2d', 'f0', 'f1'
WEIGHTS = {
    0: {'2d': 0.34, 'f0': 0.33, 'f1': 0.33}, # Background: Neutral
    1: {'2d': 0.10, 'f0': 0.45, 'f1': 0.45}, # Myocardium: Trust 3D
    2: {'2d': 0.10, 'f0': 0.45, 'f1': 0.45}, # Left Ventricle: Trust 3D
    3: {'2d': 0.80, 'f0': 0.10, 'f1': 0.10}  # Right Ventricle: Trust 2D (Rescue)
}

def keep_largest_connected_component(segmentation):
    """
    Keeps only the largest connected component for each label (1, 2, 3).
    """
    cleaned_seg = np.zeros_like(segmentation)
    
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        labeled_mask, num_features = label_cc(mask)
        
        if num_features <= 1:
            cleaned_seg[mask] = label_id
            continue
            
        # Find largest component
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0 # Ignore background
        largest_component = component_sizes.argmax()
        
        cleaned_seg[labeled_mask == largest_component] = label_id
        
    return cleaned_seg

def weighted_voting_ensemble(p2, pf0, pf1):
    """
    Performs weighted voting at voxel level.
    p2, pf0, pf1: Numpy arrays of shape (H, W, D) containing integer labels.
    """
    # Initialize score maps for each class [0, 1, 2, 3]
    # Shape: (4, H, W, D)
    # To save memory, we can process flat arrays or chunks, but for these image sizes (512^3 is big, but ours are smaller), it might fit.
    # Cardiac images are typically ~300x512x512. 
    # 4 * 300 * 512 * 512 * 4 bytes (float32) ~ 1.2 GB. It fits in RAM.
    
    shape = p2.shape
    scores = np.zeros((4,) + shape, dtype=np.float32)
    
    models = [('2d', p2), ('f0', pf0), ('f1', pf1)]
    
    for model_name, pred_map in models:
        for label_id in [0, 1, 2, 3]:
            # Get weight for this model predicting this label
            w = WEIGHTS[label_id][model_name]
            
            # Add weight to the score map where the model predicts this label
            mask = (pred_map == label_id)
            scores[label_id][mask] += w
            
    # Argmax to get final label
    final_seg = np.argmax(scores, axis=0).astype(np.uint8)
    return final_seg

def main():
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    # Get case list
    cases = sorted([os.path.basename(f).replace(".nii.gz", "") for f in glob.glob(str(DIR_3D_F0 / "*.nii.gz"))])
    print(f"Found {len(cases)} cases.")
    
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Processing {case}...")
        
        f2 = DIR_2D / f"{case}.nii.gz"
        ff0 = DIR_3D_F0 / f"{case}.nii.gz"
        ff1 = DIR_3D_F1 / f"{case}.nii.gz"
        
        if not (f2.exists() and ff0.exists() and ff1.exists()):
            print(f"Error: Missing predictions for {case}")
            continue
            
        # Load images
        img_obj = nib.load(str(ff0)) # Use F0 as reference for affine
        affine = img_obj.affine
        header = img_obj.header
        
        d2 = nib.load(str(f2)).get_fdata().astype(np.uint8)
        df0 = img_obj.get_fdata().astype(np.uint8)
        df1 = nib.load(str(ff1)).get_fdata().astype(np.uint8)
        
        # Ensemble
        ensemble_data = weighted_voting_ensemble(d2, df0, df1)
        
        # Post-processing
        final_data = keep_largest_connected_component(ensemble_data)
        final_data = final_data.astype(np.uint8)
        
        # Save
        out_path = SUBMISSION_DIR / f"{case}.nii.gz"
        new_img = nib.Nifti1Image(final_data, affine, header)
        new_img.set_data_dtype(np.uint8)
        nib.save(new_img, str(out_path))
        
    # Create Zip
    print(f"\nCreating Zip: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in SUBMISSION_DIR.glob('*.nii.gz'):
            zf.write(file_path, arcname=file_path.name)
            
    print(f"Done! Size: {os.path.getsize(ZIP_FILE) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()
