import os
import json
import torch
import multiprocessing
import zipfile
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from ensemble_model import CardiacEnsemble
import glob

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'

OUTPUT_2D_DIR = BASE_DIR / 'inference_2d_test'
OUTPUT_3D_DIR = BASE_DIR / 'inference_3d_lowres_test'
SUBMISSION_DIR = BASE_DIR / 'submission_final'
ZIP_FILE = BASE_DIR / 'submission_20251120.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

def get_test_cases():
    # Find all *_0000.nii.gz files in imagesTs
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def run_inference(model_name, output_dir, cases):
    print(f"\n=== Running {model_name} Inference on Test Set ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Determine configuration and trainer
    if model_name == '2d':
        config = '2d'
        trainer = 'nnUNetTrainer__nnUNetPlans__2d'
    else:
        config = '3d_lowres'
        trainer = 'nnUNetTrainer__nnUNetPlans__3d_lowres'
        
    # Initialize predictor
    print(f"Initializing {model_name} predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False, # disable TTA for speed
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    # Load checkpoint
    model_folder = join(RESULTS_DIR, trainer)
    print(f"Loading checkpoint from {model_folder}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )
    
    # Predict loop
    for i, case in enumerate(cases):
        output_file = join(output_dir, f'{case}.nii.gz')
        
        if os.path.exists(output_file):
            print(f"[{i+1}/{len(cases)}] Skipping {case}, already exists.")
            continue
            
        print(f"[{i+1}/{len(cases)}] Predicting {case}...")
        input_files = [[join(IMAGES_TS_DIR, f'{case}_0000.nii.gz')]]
        
        try:
            predictor.predict_from_files(
                input_files,
                [output_file],
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
            import traceback
            traceback.print_exc()

def run_ensemble(cases):
    print("\n=== Running Ensemble on Test Set ===")
    
    ensemble = CardiacEnsemble(
        model_2d_folder=RESULTS_DIR / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0',
        model_3d_lowres_folder=RESULTS_DIR / 'nnUNetTrainer__nnUNetPlans__3d_lowres' / 'fold_0'
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
    
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            for file in files:
                if file.endswith('.nii.gz'):
                    file_path = os.path.join(root, file)
                    arcname = file # Add file at root of zip
                    print(f"Adding {file}...")
                    zipf.write(file_path, arcname)
                    
    print(f"Zip created successfully at {ZIP_FILE}")

def main():
    cases = get_test_cases()
    print(f"Found {len(cases)} test cases.")
    
    # 1. Run 2D Inference
    run_inference('2d', OUTPUT_2D_DIR, cases)
    
    # 2. Run 3D Inference
    run_inference('3d_lowres', OUTPUT_3D_DIR, cases)
    
    # 3. Run Ensemble
    run_ensemble(cases)
    
    # 4. Zip
    create_submission_zip()
    
    print("\nAll tasks completed!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
