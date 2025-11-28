import os
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder

# Explicitly set environment variables in Python to be 100% sure
os.environ['nnUNet_raw'] = r"C:\CardiacSeg\nnUNet_raw"
os.environ['nnUNet_preprocessed'] = r"C:\CardiacSeg\nnUNet_preprocessed"
os.environ['nnUNet_results'] = r"C:\CardiacSeg\nnUNet_results"

# Define parameters
dataset_name_or_id = 'Dataset001_CardiacSeg'
configuration = '3d_lowres'
trainer_name = 'nnUNetTrainerCustomEpochs'
plans_identifier = 'nnUNetPlans'
folds = [0, 1, 2, 3, 4]
input_folder = r"C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg\imagesTs"
output_folder = r"C:\CardiacSeg\inference_3d_lowres_ensemble_test"

# Construct the model training output directory manually to verify
model_training_output_dir = get_output_folder(dataset_name_or_id, trainer_name, plans_identifier, configuration)
print(f"Calculated model output dir: {model_training_output_dir}")

# Initialize Predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False, # disable_tta
    perform_everything_on_device=True,
    device=torch.device('cuda'),
    verbose=True,
    verbose_preprocessing=True,
    allow_tqdm=True
)

print("Initializing from trained model folder...")
predictor.initialize_from_trained_model_folder(
    model_training_output_dir,
    use_folds=folds,
    checkpoint_name='checkpoint_final.pth'
)

print("Running prediction...")
predictor.predict_from_files(
    input_folder,
    output_folder,
    save_probabilities=False,
    overwrite=True,
    num_processes_preprocessing=2,
    num_processes_segmentation_export=2,
    folder_with_segs_from_prev_stage=None,
    num_parts=1,
    part_id=0
)
print("Done!")
