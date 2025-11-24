import os
import json
import torch
import multiprocessing
import zipfile
from pathlib import Path
import glob

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

OUTPUT_2D_DIR = BASE_DIR / 'inference_2d_test'
OUTPUT_3D_DIR = BASE_DIR / 'inference_3d_fullres_test'
SUBMISSION_DIR = BASE_DIR / 'submission_final_fullres'
ZIP_FILE = BASE_DIR / 'submission_fullres_20251121.zip'

def get_test_cases():
    # Find all *_0000.nii.gz files in imagesTs
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def run_inference_3d_fullres(output_dir, cases):
    print(f"\n=== Running 3D Fullres Inference on Test Set ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    config = '3d_fullres'
    trainer = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
        
    # Initialize predictor
    print(f"Initializing 3D Fullres predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True, # Enable TTA for best results
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    # Initialize from checkpoint
    model_folder = join(RESULTS_DIR, trainer)
    print(f"Loading checkpoint from {model_folder}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    # Predict loop
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Predicting {case}...")
        
        input_files = [[join(IMAGES_TS_DIR, f'{case}_0000.nii.gz')]]
        output_files = [join(output_dir, f'{case}.nii.gz')]
        
        # Check if already exists
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

def run_ensemble(cases):
    print("\n=== Running Ensemble (2D + 3D Fullres) ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    ensemble = CardiacEnsemble(
        model_2d_folder=RESULTS_DIR / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0',
        model_3d_lowres_folder=RESULTS_DIR / 'nnUNetTrainer__nnUNetPlans__3d_fullres' / 'fold_0'
    )
    
    ensemble.ensemble_predictions(
        pred_2d_folder=OUTPUT_2D_DIR,
        pred_3d_folder=OUTPUT_3D_DIR,
        output_folder=SUBMISSION_DIR,
        method='label_specific',
        case_list=cases
    )

def create_submission_zip():
    print(f"\n=== Creating Submission Zip: {ZIP_FILE} ===")
    
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in SUBMISSION_DIR.glob('*.nii.gz'):
            zf.write(file_path, arcname=file_path.name)
            
    print(f"Zip created. Size: {os.path.getsize(ZIP_FILE) / (1024*1024):.2f} MB")

def main():
    cases = get_test_cases()
    print(f"Found {len(cases)} test cases.")
    
    # 1. Run 3D Fullres Inference
    run_inference_3d_fullres(OUTPUT_3D_DIR, cases)
    
    # 2. Skip Ensemble (3D Fullres is better alone: ~0.79 vs 0.56)
    # run_ensemble(cases)
    
    # 3. Create Zip from 3D Output directly
    print(f"\n=== Creating Submission Zip from 3D Output: {ZIP_FILE} ===")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in OUTPUT_3D_DIR.glob('*.nii.gz'):
            zf.write(file_path, arcname=file_path.name)
            
    print(f"Zip created. Size: {os.path.getsize(ZIP_FILE) / (1024*1024):.2f} MB")
    
    print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
