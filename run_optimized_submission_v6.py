
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

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'

# Output Dirs for Probabilities
OUTPUT_2D_PROB_DIR = BASE_DIR / 'inference_2d_test_prob'
OUTPUT_3D_PROB_DIR = BASE_DIR / 'inference_3d_lowres_test_prob'
SUBMISSION_DIR = BASE_DIR / 'submission_optimized_v6'
ZIP_FILE = BASE_DIR / 'submission_optimized_v6_20251122.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

# Soft Voting Weights
# Format: {Label_ID: {'2d': Weight, '3d': Weight}}
WEIGHTS = {
    0: {'2d': 0.5, '3d': 0.5}, # Background
    1: {'2d': 0.3, '3d': 0.7}, # Myocardium: Trust 3D
    2: {'2d': 0.4, '3d': 0.6}, # Left Ventricle: Trust 3D
    3: {'2d': 0.8, '3d': 0.2}   # Right Ventricle: Trust 2D
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def run_inference_with_probs(model_type, output_dir, cases):
    print(f"\n=== Running {model_type} Inference (with Probabilities) ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if model_type == '2d':
        trainer = 'nnUNetTrainer__nnUNetPlans__2d'
        config = '2d'
    else:
        trainer = 'nnUNetTrainer__nnUNetPlans__3d_lowres'
        config = '3d_lowres'
        
    model_folder = join(RESULTS_DIR, trainer)
    
    # Check if all npz files exist
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    if len(existing_files) == len(cases):
        print(f"All {model_type} probability files already exist. Skipping.")
        return

    print(f"Initializing {model_type} predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True, # Enable TTA
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    print(f"Loading checkpoint from {model_folder}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )
    
    for i, case in enumerate(cases):
        output_file = join(output_dir, f'{case}.nii.gz') # nnUNet expects .nii.gz extension even if saving probs
        npz_file = join(output_dir, f'{case}.npz')
        
        if os.path.exists(npz_file):
            print(f"[{i+1}/{len(cases)}] Skipping {case}, .npz exists.")
            continue
            
        print(f"[{i+1}/{len(cases)}] Predicting {case}...")
        input_files = [[join(IMAGES_TS_DIR, f'{case}_0000.nii.gz')]]
        
        try:
            predictor.predict_from_files(
                input_files,
                [output_file],
                save_probabilities=True, # CRITICAL: Save probabilities
                overwrite=True,
                num_processes_segmentation_export=1,
                num_processes_preprocessing=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )
        except Exception as e:
            print(f"Error predicting {case}: {e}")

def postprocess_segmentation(segmentation):
    """
    Applies LCC and Fill Holes (V5 Logic).
    """
    cleaned_seg = np.zeros_like(segmentation)
    
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        # 1. Keep Largest Connected Component
        labeled_mask, num_features = label_cc(mask)
        if num_features > 1:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component = component_sizes.argmax()
            mask = (labeled_mask == largest_component)
            
        # 2. Fill Holes (Only for LV=2 and RV=3)
        if label_id in [2, 3]:
            mask = binary_fill_holes(mask)
            
        cleaned_seg[mask] = label_id
        
    return cleaned_seg

def soft_voting_ensemble(cases):
    print("\n=== Running Soft Voting Ensemble (V6) ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Ensembling {case}...")
        
        npz_2d = OUTPUT_2D_PROB_DIR / f"{case}.nii.gz.npz"
        npz_3d = OUTPUT_3D_PROB_DIR / f"{case}.nii.gz.npz"
        
        if not (npz_2d.exists() and npz_3d.exists()):
            print(f"Error: Missing probabilities for {case}")
            continue
            
        # Load probabilities
        # Shape: (4, H, W, D)
        prob_2d = np.load(str(npz_2d))['probabilities']
        prob_3d = np.load(str(npz_3d))['probabilities']
        
        # Load geometry from one of the files (e.g., 3D prediction nii)
        # We need the affine to save the result.
        # The predict_from_files generated a .nii.gz as well.
        ref_img_path = OUTPUT_3D_PROB_DIR / f"{case}.nii.gz"
        ref_img = nib.load(str(ref_img_path))
        
        # Initialize final probability map
        final_prob = np.zeros_like(prob_2d)
        
        # Weighted Average
        for label_id in [0, 1, 2, 3]:
            w2 = WEIGHTS[label_id]['2d']
            w3 = WEIGHTS[label_id]['3d']
            
            final_prob[label_id] = (prob_2d[label_id] * w2) + (prob_3d[label_id] * w3)
            
        # Argmax
        final_seg = np.argmax(final_prob, axis=0).astype(np.uint8)
        
        # Post-processing (V5)
        final_seg = postprocess_segmentation(final_seg)
        
        # Transpose if necessary to match NIfTI shape
        # Probabilities are usually (C, Z, Y, X), so final_seg is (Z, Y, X)
        # NIfTI is usually (X, Y, Z)
        if final_seg.shape != ref_img.shape:
            print(f"Transposing {case} from {final_seg.shape} to {ref_img.shape}")
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
    
    # 1. Run 2D Inference (Probabilities)
    run_inference_with_probs('2d', OUTPUT_2D_PROB_DIR, cases)
    
    # 2. Run 3D Lowres Inference (Probabilities)
    run_inference_with_probs('3d_lowres', OUTPUT_3D_PROB_DIR, cases)
    
    # 3. Soft Voting + Post-processing
    soft_voting_ensemble(cases)
    
    # 4. Zip
    create_submission_zip(cases)
