
import os
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

# Paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'
OUTPUT_DIR = BASE_DIR / 'inference_3d_fullres_test_fold1'

# Set env vars
os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

def run_inference():
    print(f"=== Running 3D Fullres Inference (Fold 1) ===")
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
        
    trainer = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    model_folder = RESULTS_DIR / trainer
    
    print(f"Initializing predictor from {model_folder} (Fold 1)...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True, # TTA
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(1,), # ONLY FOLD 1
        checkpoint_name='checkpoint_best.pth'
    )

    print("Starting prediction...")
    
    # Manually construct lists to avoid issues with folder parsing
    import glob
    files = sorted(glob.glob(str(IMAGES_TS_DIR / "*_0000.nii.gz")))
    input_files = [[f] for f in files]
    output_files = [str(OUTPUT_DIR / os.path.basename(f).replace("_0000.nii.gz", ".nii.gz")) for f in files]
    
    print(f"Found {len(input_files)} cases.")
    
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_files,
        output_folder_or_list_of_truncated_output_files=output_files,
        save_probabilities=False,
        overwrite=False,
        num_processes_segmentation_export=2,
        num_processes_preprocessing=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    print("Done!")

if __name__ == '__main__':
    run_inference()
