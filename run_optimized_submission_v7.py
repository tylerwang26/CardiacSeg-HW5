
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
OUTPUT_2D_PROB_DIR = BASE_DIR / 'inference_2d_test_prob' # Reuse V6
OUTPUT_3D_LOWRES_PROB_DIR = BASE_DIR / 'inference_3d_lowres_test_prob' # Reuse V6
OUTPUT_3D_FULLRES_F0_PROB_DIR = BASE_DIR / 'inference_3d_fullres_f0_prob' # New
OUTPUT_3D_FULLRES_F1_PROB_DIR = BASE_DIR / 'inference_3d_fullres_f1_prob' # New

SUBMISSION_DIR = BASE_DIR / 'submission_optimized_v7'
ZIP_FILE = BASE_DIR / 'submission_optimized_v7_20251123.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

# Soft Voting Weights (V7)
# Format: {Label_ID: {'2d': W, '3d_low': W, '3d_full': W}}
# Note: 3d_full weight will be split between F0 and F1 equally
WEIGHTS = {
    0: {'2d': 0.4, '3d_low': 0.4, '3d_full': 0.2}, # Background
    1: {'2d': 0.2, '3d_low': 0.5, '3d_full': 0.3}, # Myocardium: Lowres is stable, Fullres adds detail
    2: {'2d': 0.2, '3d_low': 0.4, '3d_full': 0.4}, # LV: Fullres helps with boundaries
    3: {'2d': 0.7, '3d_low': 0.2, '3d_full': 0.1}  # RV: 2D is still king
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def run_inference_with_probs(model_config, trainer, fold, output_dir, cases):
    print(f"\n=== Running {model_config} Fold {fold} Inference (with Probabilities) ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_folder = join(RESULTS_DIR, trainer)
    
    # Check if all npz files exist
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    if len(existing_files) == len(cases):
        print(f"All probability files in {output_dir} already exist. Skipping.")
        return

    print(f"Initializing predictor...")
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
    
    print(f"Loading checkpoint from {model_folder} (Fold {fold})")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth'
    )
    
    for i, case in enumerate(cases):
        output_file = join(output_dir, f'{case}.nii.gz')
        npz_file = join(output_dir, f'{case}.nii.gz.npz')
        
        if os.path.exists(npz_file):
            print(f"[{i+1}/{len(cases)}] Skipping {case}, .npz exists.")
            continue
            
        print(f"[{i+1}/{len(cases)}] Predicting {case}...")
        input_files = [[join(IMAGES_TS_DIR, f'{case}_0000.nii.gz')]]
        
        try:
            predictor.predict_from_files(
                input_files,
                [output_file],
                save_probabilities=True,
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
    Applies LCC and Fill Holes.
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
    print("\n=== Running Soft Voting Ensemble (V7: 4-Model Ensemble) ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Ensembling {case}...")
        
        # Paths
        p_2d = OUTPUT_2D_PROB_DIR / f"{case}.nii.gz.npz"
        p_low = OUTPUT_3D_LOWRES_PROB_DIR / f"{case}.nii.gz.npz"
        p_full_f0 = OUTPUT_3D_FULLRES_F0_PROB_DIR / f"{case}.nii.gz.npz"
        p_full_f1 = OUTPUT_3D_FULLRES_F1_PROB_DIR / f"{case}.nii.gz.npz"
        
        # Check existence
        if not (p_2d.exists() and p_low.exists() and p_full_f0.exists() and p_full_f1.exists()):
            print(f"Error: Missing probabilities for {case}")
            continue
            
        # Load probabilities
        prob_2d = np.load(str(p_2d))['probabilities']
        prob_low = np.load(str(p_low))['probabilities']
        prob_full_f0 = np.load(str(p_full_f0))['probabilities']
        prob_full_f1 = np.load(str(p_full_f1))['probabilities']
        
        # Load reference image for affine
        ref_img_path = OUTPUT_3D_LOWRES_PROB_DIR / f"{case}.nii.gz"
        ref_img = nib.load(str(ref_img_path))
        
        # Initialize final probability map
        final_prob = np.zeros_like(prob_2d)
        
        # Weighted Average
        for label_id in [0, 1, 2, 3]:
            w_2d = WEIGHTS[label_id]['2d']
            w_low = WEIGHTS[label_id]['3d_low']
            w_full = WEIGHTS[label_id]['3d_full']
            
            # Split fullres weight between F0 and F1
            w_full_split = w_full / 2.0
            
            # Ensemble
            # Note: Ensure shapes match. If not, we might need resizing, but nnU-Net usually handles this if configured correctly.
            # However, 3D Fullres might have slightly different output shape if padding differed?
            # Usually predict_from_files resamples to original image space, so shapes should match (C, Z, Y, X) or (C, X, Y, Z) depending on transpose.
            # nnU-Net .npz is (C, Z, Y, X) usually.
            
            # Safety check on shapes?
            # Assuming all are (4, Z, Y, X)
            
            term_2d = prob_2d[label_id] * w_2d
            term_low = prob_low[label_id] * w_low
            term_full = (prob_full_f0[label_id] * w_full_split) + (prob_full_f1[label_id] * w_full_split)
            
            final_prob[label_id] = term_2d + term_low + term_full
            
        # Argmax
        final_seg = np.argmax(final_prob, axis=0).astype(np.uint8)
        
        # Post-processing
        final_seg = postprocess_segmentation(final_seg)
        
        # Transpose Check (Fix from V6)
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
    
    # 1. Run 3D Fullres Fold 0
    run_inference_with_probs(
        '3d_fullres', 
        'nnUNetTrainer__nnUNetPlans__3d_fullres', 
        0, 
        OUTPUT_3D_FULLRES_F0_PROB_DIR, 
        cases
    )
    
    # 2. Run 3D Fullres Fold 1
    run_inference_with_probs(
        '3d_fullres', 
        'nnUNetTrainer__nnUNetPlans__3d_fullres', 
        1, 
        OUTPUT_3D_FULLRES_F1_PROB_DIR, 
        cases
    )
    
    # 3. Soft Voting (Uses 2D and Lowres from V6 + New Fullres)
    soft_voting_ensemble(cases)
    
    # 4. Zip
    create_submission_zip(cases)
