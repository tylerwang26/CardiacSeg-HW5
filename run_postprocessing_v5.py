
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import zipfile
from scipy.ndimage import label as label_cc
from scipy.ndimage import binary_fill_holes
import glob

# Paths
BASE_DIR = Path(r'C:\CardiacSeg')
INPUT_DIR = BASE_DIR / 'submission_optimized_v4' # Start from V4 results
OUTPUT_DIR = BASE_DIR / 'submission_optimized_v5'
ZIP_FILE = BASE_DIR / 'submission_optimized_v5_20251122.zip'

def postprocess_segmentation(segmentation):
    """
    Applies LCC and Fill Holes to the segmentation.
    """
    cleaned_seg = np.zeros_like(segmentation)
    
    # Process each label
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        # 1. Keep Largest Connected Component
        labeled_mask, num_features = label_cc(mask)
        
        if num_features > 1:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0 # Ignore background
            largest_component = component_sizes.argmax()
            mask = (labeled_mask == largest_component)
            
        # 2. Fill Holes (Only for LV=2 and RV=3, NOT for Myocardium=1 which is a ring)
        if label_id in [2, 3]:
            # Fill holes slice by slice to avoid 3D topology issues or just 3D fill?
            # 3D fill is safer for 3D volumes usually, but let's stick to what's robust.
            # binary_fill_holes works in 3D by default if input is 3D.
            mask = binary_fill_holes(mask)
            
        cleaned_seg[mask] = label_id
        
    return cleaned_seg

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
        
    # Get case list
    cases = sorted([os.path.basename(f) for f in glob.glob(str(INPUT_DIR / "*.nii.gz"))])
    print(f"Found {len(cases)} cases in {INPUT_DIR}")
    
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Post-processing {case}...")
        
        input_path = INPUT_DIR / case
        output_path = OUTPUT_DIR / case
        
        img = nib.load(str(input_path))
        data = img.get_fdata().astype(np.uint8)
        
        # Apply Post-processing
        new_data = postprocess_segmentation(data)
        
        # Save
        out_img = nib.Nifti1Image(new_data, img.affine, img.header)
        nib.save(out_img, str(output_path))

    # Create Zip
    print(f"\n=== Creating Submission Zip: {ZIP_FILE} ===")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for case in cases:
            file_path = OUTPUT_DIR / case
            zf.write(file_path, case)
    print("Done!")

if __name__ == "__main__":
    main()
