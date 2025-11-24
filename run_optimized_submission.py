import os
import json
import torch
import multiprocessing
import zipfile
import numpy as np
import nibabel as nib
from pathlib import Path
import glob
from scipy.ndimage import label as label_cc

# Set paths BEFORE importing nnunetv2
BASE_DIR = Path(r'C:\CardiacSeg')
os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from ensemble_model import CardiacEnsemble

# Define directories
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'

OUTPUT_2D_DIR = BASE_DIR / 'inference_2d_test_tta'
OUTPUT_3D_DIR = BASE_DIR / 'inference_3d_fullres_test' # Already exists and has TTA
SUBMISSION_DIR = BASE_DIR / 'submission_optimized'
ZIP_FILE = BASE_DIR / 'submission_optimized_20251121.zip'

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def run_inference_2d_tta(output_dir, cases):
    print(f"\n=== Running 2D Inference on Test Set (with TTA) ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    trainer = 'nnUNetTrainer__nnUNetPlans__2d'
    model_folder = join(RESULTS_DIR, trainer)
    
    print(f"Initializing 2D predictor from {model_folder}...")
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

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Predicting {case}...")
        
        input_files = [[join(IMAGES_TS_DIR, f'{case}_0000.nii.gz')]]
        output_files = [join(output_dir, f'{case}.nii.gz')]
        
        if os.path.exists(output_files[0]):
            print(f"Skipping {case}, already exists.")
            continue

        try:
            predictor.predict_from_files(
                input_files,
                output_files,
                save_probabilities=False,
                overwrite=True,
                num_processes_segmentation_export=1,
                num_processes_preprocessing=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )
        except Exception as e:
            print(f"Error predicting {case}: {e}")

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
        # Ignore background (0)
        component_sizes[0] = 0
        largest_component = component_sizes.argmax()
        
        cleaned_seg[labeled_mask == largest_component] = label_id
        
    return cleaned_seg

def run_ensemble_and_postprocessing(cases):
    print("\n=== Running Ensemble & Post-processing ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    ensemble = CardiacEnsemble(
        model_2d_folder=RESULTS_DIR / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0',
        model_3d_lowres_folder=RESULTS_DIR / 'nnUNetTrainer__nnUNetPlans__3d_fullres' / 'fold_0'
    )
    
    # We manually run ensemble to inject post-processing
    # Or we can use the ensemble class and then post-process the files.
    # Let's use the class to generate raw ensemble, then post-process.
    
    temp_ensemble_dir = BASE_DIR / 'temp_ensemble_raw'
    if not temp_ensemble_dir.exists():
        temp_ensemble_dir.mkdir()
        
    print("Generating raw ensemble...")
    ensemble.ensemble_predictions(
        pred_2d_folder=OUTPUT_2D_DIR,
        pred_3d_folder=OUTPUT_3D_DIR,
        output_folder=temp_ensemble_dir,
        method='label_specific',
        case_list=cases
    )
    
    print("Applying LCC Post-processing...")
    for case in cases:
        src_path = temp_ensemble_dir / f"{case}.nii.gz"
        dst_path = SUBMISSION_DIR / f"{case}.nii.gz"
        
        if not src_path.exists():
            print(f"Warning: {case} missing from ensemble output")
            continue
            
        nii = nib.load(str(src_path))
        data = nii.get_fdata()
        
        cleaned_data = keep_largest_connected_component(data)
        
        # Cast to uint8 to ensure integer labels and remove float artifacts
        cleaned_data = cleaned_data.astype(np.uint8)
        
        # Save
        new_nii = nib.Nifti1Image(cleaned_data, nii.affine, nii.header)
        new_nii.set_data_dtype(np.uint8)
        nib.save(new_nii, str(dst_path))
        
    print("Post-processing done.")

def create_submission_zip():
    print(f"\n=== Creating Submission Zip: {ZIP_FILE} ===")
    
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in SUBMISSION_DIR.glob('*.nii.gz'):
            zf.write(file_path, arcname=file_path.name)
            
    print(f"Zip created. Size: {os.path.getsize(ZIP_FILE) / (1024*1024):.2f} MB")

def main():
    cases = get_test_cases()
    print(f"Found {len(cases)} test cases.")
    
    # 1. Run 2D Inference (TTA)
    run_inference_2d_tta(OUTPUT_2D_DIR, cases)
    
    # 2. Run Ensemble & Post-processing
    # Note: OUTPUT_3D_DIR is assumed to be populated from previous run (which had TTA)
    if not OUTPUT_3D_DIR.exists() or not list(OUTPUT_3D_DIR.glob('*.nii.gz')):
        print("Error: 3D output directory is empty. Please run run_test_submission_fullres.py first or check paths.")
        return

    run_ensemble_and_postprocessing(cases)
    
    # 3. Create Zip
    create_submission_zip()
    
    print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
