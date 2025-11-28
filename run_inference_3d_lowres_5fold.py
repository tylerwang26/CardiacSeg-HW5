
import os
from pathlib import Path

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'
OUTPUT_DIR = BASE_DIR / 'inference_3d_lowres_5fold_prob'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join
import glob

def run_inference():
    print(f"=== Running 3D Lowres 5-Fold Inference ===")
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
        
    # Use the CustomEpochs trainer which has 5 folds
    trainer = 'nnUNetTrainerCustomEpochs__nnUNetPlans__3d_lowres'
    model_folder = RESULTS_DIR / trainer
    
    print(f"Initializing predictor from {model_folder} (Folds 0-4)...")
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

    # Initialize with all 5 folds
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name='checkpoint_best.pth'
    )

    print("Starting prediction...")
    
    files = sorted(glob.glob(str(IMAGES_TS_DIR / "*_0000.nii.gz")))
    input_files = [[f] for f in files]
    output_files = [str(OUTPUT_DIR / os.path.basename(f).replace("_0000.nii.gz", ".nii.gz")) for f in files]
    
    # Check which ones are already done
    todo_input = []
    todo_output = []
    
    for i, out_f in enumerate(output_files):
        if os.path.exists(out_f) and os.path.exists(out_f.replace(".nii.gz", ".npz")):
            print(f"Skipping {os.path.basename(out_f)}, already exists.")
        else:
            todo_input.append(input_files[i])
            todo_output.append(out_f)
            
    if not todo_input:
        print("All files already predicted.")
        return

    print(f"Predicting {len(todo_input)} cases...")
    
    predictor.predict_from_files(
        todo_input,
        todo_output,
        save_probabilities=True,
        overwrite=True,
        num_processes_segmentation_export=1,
        num_processes_preprocessing=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

if __name__ == '__main__':
    run_inference()
