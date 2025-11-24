import os
import json
import torch
import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Set paths
os.environ['nnUNet_raw'] = r'C:\CardiacSeg\nnUNet_raw'
os.environ['nnUNet_preprocessed'] = r'C:\CardiacSeg\nnUNet_preprocessed'
os.environ['nnUNet_results'] = r'C:\CardiacSeg\nnUNet_results'

def main():
    print("Starting 3D Lowres Validation Inference...")
    
    # Load splits
    splits_file = join(os.environ['nnUNet_preprocessed'], 'Dataset001_CardiacSeg', 'splits_final.json')
    splits = load_json(splits_file)
    val_cases = splits[0]['val']
    
    print(f"Found {len(val_cases)} validation cases: {val_cases}")

    # Initialize predictor
    print("Initializing predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False, # disable TTA
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    # Initialize from checkpoint
    model_folder = join(os.environ['nnUNet_results'], 'Dataset001_CardiacSeg', 'nnUNetTrainer__nnUNetPlans__3d_lowres')
    print(f"Loading checkpoint from {model_folder}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    # Output directory
    output_dir = r'C:\CardiacSeg\inference_3d_lowres_validation'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Input directory
    input_dir = join(os.environ['nnUNet_raw'], 'Dataset001_CardiacSeg', 'imagesTr')

    # Predict loop
    for i, case in enumerate(val_cases):
        print(f"[{i+1}/{len(val_cases)}] Predicting {case}...")
        
        input_files = [[join(input_dir, f'{case}_0000.nii.gz')]]
        output_files = [join(output_dir, f'{case}.nii.gz')]
        
        # Check if already exists
        if os.path.exists(output_files[0]):
            print(f"Skipping {case}, already exists.")
            continue

        try:
            # Predict
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
            print(f"Successfully predicted {case}")
        except Exception as e:
            print(f"Error predicting {case}: {e}")
            import traceback
            traceback.print_exc()

    print("Inference completed.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
