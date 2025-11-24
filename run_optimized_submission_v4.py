
import os
import json
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import zipfile
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join
import glob

# Set paths
BASE_DIR = Path(r'C:\CardiacSeg')
RAW_DIR = BASE_DIR / 'nnUNet_raw' / 'Dataset001_CardiacSeg'
RESULTS_DIR = BASE_DIR / 'nnUNet_results' / 'Dataset001_CardiacSeg'
IMAGES_TS_DIR = RAW_DIR / 'imagesTs'

# Input/Output Dirs
OUTPUT_2D_DIR = BASE_DIR / 'inference_2d_test_tta' # Already generated
OUTPUT_3D_DIR = BASE_DIR / 'inference_3d_lowres_test_tta' # To be generated
SUBMISSION_DIR = BASE_DIR / 'submission_optimized_v4'
ZIP_FILE = BASE_DIR / 'submission_optimized_v4_20251122.zip'

os.environ['nnUNet_raw'] = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(BASE_DIR / 'nnUNet_results')

# Weights from ensemble_model.py (Successful Baseline)
# Format: {Label_ID: {'2d': Weight, '3d': Weight}}
WEIGHTS = {
    1: {'2d': 0.3, '3d': 0.7},  # Myocardium: Trust 3D
    2: {'2d': 0.4, '3d': 0.6},  # Left Ventricle: Trust 3D
    3: {'2d': 0.8, '3d': 0.2}   # Right Ventricle: Trust 2D
}

def get_test_cases():
    files = glob.glob(os.path.join(IMAGES_TS_DIR, "*_0000.nii.gz"))
    cases = [os.path.basename(f).replace("_0000.nii.gz", "") for f in files]
    return sorted(cases)

def run_3d_lowres_tta_inference(cases):
    print(f"\n=== Running 3D Lowres Inference (with TTA) on Test Set ===")
    
    if not os.path.exists(OUTPUT_3D_DIR):
        os.makedirs(OUTPUT_3D_DIR)
        
    trainer = 'nnUNetTrainer__nnUNetPlans__3d_lowres'
    model_folder = join(RESULTS_DIR, trainer)
    
    # Check if all cases are already predicted
    existing_files = [f for f in os.listdir(OUTPUT_3D_DIR) if f.endswith('.nii.gz')]
    if len(existing_files) == len(cases):
        print("All 3D Lowres TTA predictions already exist. Skipping inference.")
        return

    print(f"Initializing 3D Lowres predictor with TTA (Mirroring)...")
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
        output_file = join(OUTPUT_3D_DIR, f'{case}.nii.gz')
        
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

def weighted_voting_ensemble(p2, p3):
    """
    Performs weighted voting.
    p2: 2D prediction (H, W, D)
    p3: 3D prediction (H, W, D)
    """
    shape = p2.shape
    scores = np.zeros((4,) + shape, dtype=np.float32)
    
    # One-hot encoding logic implicitly handled by adding weights to the predicted class channel
    
    # 2D Model Contribution
    for label_id in [1, 2, 3]:
        w = WEIGHTS[label_id]['2d']
        mask = (p2 == label_id)
        scores[label_id][mask] += w
        
    # 3D Model Contribution
    for label_id in [1, 2, 3]:
        w = WEIGHTS[label_id]['3d']
        mask = (p3 == label_id)
        scores[label_id][mask] += w
        
    # Background handling (implicit or explicit)
    # If neither predicts a label, it remains 0.
    # If both predict different labels, the one with higher weight wins.
    # If sum of weights < threshold? No, just argmax.
    # But we need to handle background (0).
    # Let's assume background has a base score or just use the fact that if no label wins, it's 0.
    # Actually, a better way is:
    # Initialize background score to a small epsilon or 0.5?
    # Or just:
    # scores[0] = 0.5 (neutral)
    # If a pixel is 0 in both, scores[1,2,3] will be 0, so scores[0] wins.
    
    # Let's be more precise:
    # If p2==0, it votes for 0 with some weight?
    # The original ensemble logic in ensemble_model.py didn't show the voting implementation, 
    # but usually it's probability averaging. Since we only have hard labels here (nii.gz),
    # we are doing "Hard Voting" with weights.
    
    # Let's assume:
    # If p2 says 0, it adds weight to 0?
    # Let's look at the weights again.
    # L1: 2D(0.3) + 3D(0.7) = 1.0
    # If p2=1 (0.3) and p3=1 (0.7) -> Score 1 = 1.0.
    # If p2=1 (0.3) and p3=0 (?) -> Score 1 = 0.3. Score 0 = ?
    
    # We need a weight for background.
    # Let's assume background weight is complementary.
    # If p2=0, it contributes to class 0.
    # But how much?
    # Let's try a simpler approach:
    # Iterate over each voxel is too slow in Python.
    # Vectorized approach:
    
    # Initialize scores for 0, 1, 2, 3
    # We can assign a default weight to the prediction.
    # But since we have different weights for different classes...
    
    # Let's use the "Confidence" interpretation.
    # If 2D predicts 3, it's 80% confident.
    # If 3D predicts 3, it's 20% confident.
    
    # What if 2D predicts 0?
    # It means it's confident it's NOT 1, 2, or 3.
    
    # Let's try this:
    # Score for class C = Sum(Weight_Model_M_Class_C * I(Model_M predicts C))
    # But we only have weights for 1, 2, 3.
    # What is the weight for 0?
    # Let's assume uniform weight for 0, or derived.
    
    # Alternative:
    # Use the specific logic from the prompt description:
    # "2D Model: 擅長檢測 Label 3 (右心室) - 48.63%"
    # "3D Low-Res: 擅長 Label 1/2 (心肌/左心室) - 88%/67%"
    
    # Let's implement the logic:
    # For each pixel:
    # If 2D says 3 -> High vote for 3.
    # If 3D says 1 -> High vote for 1.
    
    # Let's define a "Vote Map"
    # 2D Votes:
    # 0 -> Vote 0 (Weight ?)
    # 1 -> Vote 1 (0.3)
    # 2 -> Vote 2 (0.4)
    # 3 -> Vote 3 (0.8)
    
    # 3D Votes:
    # 0 -> Vote 0 (Weight ?)
    # 1 -> Vote 1 (0.7)
    # 2 -> Vote 2 (0.6)
    # 3 -> Vote 3 (0.2)
    
    # If we set Background Weight to say 0.5 for both?
    # Case: 2D=3 (0.8), 3D=0 (0.5). Winner=3 (0.8 > 0.5). Correct.
    # Case: 2D=0 (0.5), 3D=1 (0.7). Winner=1 (0.7 > 0.5). Correct.
    # Case: 2D=3 (0.8), 3D=1 (0.7). Winner=3 (0.8 > 0.7). Correct (2D is better at RV).
    # Case: 2D=1 (0.3), 3D=2 (0.6). Winner=2 (0.6 > 0.3). Correct (3D is better at LV).
    
    # This seems robust. Let's use 0.5 for background for both models.
    
    bg_weight_2d = 0.5
    bg_weight_3d = 0.5
    
    # 2D Votes
    mask_0 = (p2 == 0)
    scores[0][mask_0] += bg_weight_2d
    
    for l in [1, 2, 3]:
        mask = (p2 == l)
        scores[l][mask] += WEIGHTS[l]['2d']
        
    # 3D Votes
    mask_0 = (p3 == 0)
    scores[0][mask_0] += bg_weight_3d
    
    for l in [1, 2, 3]:
        mask = (p3 == l)
        scores[l][mask] += WEIGHTS[l]['3d']
        
    final_seg = np.argmax(scores, axis=0).astype(np.uint8)
    return final_seg

def run_ensemble(cases):
    print("\n=== Running Ensemble (V4) ===")
    
    if not SUBMISSION_DIR.exists():
        SUBMISSION_DIR.mkdir()
        
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Ensembling {case}...")
        
        f2 = OUTPUT_2D_DIR / f"{case}.nii.gz"
        f3 = OUTPUT_3D_DIR / f"{case}.nii.gz"
        
        if not (f2.exists() and f3.exists()):
            print(f"Error: Missing predictions for {case}")
            continue
            
        # Load images
        img2 = nib.load(str(f2))
        img3 = nib.load(str(f3))
        
        d2 = img2.get_fdata().astype(np.uint8)
        d3 = img3.get_fdata().astype(np.uint8)
        
        # Ensemble
        final_data = weighted_voting_ensemble(d2, d3)
        
        # Save
        out_img = nib.Nifti1Image(final_data, img3.affine, img3.header)
        nib.save(out_img, str(SUBMISSION_DIR / f"{case}.nii.gz"))

def create_submission_zip(cases):
    print(f"\n=== Creating Submission Zip: {ZIP_FILE} ===")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for case in cases:
            file_path = SUBMISSION_DIR / f"{case}.nii.gz"
            if file_path.exists():
                zf.write(file_path, f"{case}.nii.gz")
            else:
                print(f"Warning: Missing file {file_path}")
    print("Done!")

if __name__ == "__main__":
    cases = get_test_cases()
    
    # 1. Run 3D Lowres TTA Inference
    run_3d_lowres_tta_inference(cases)
    
    # 2. Run Ensemble
    run_ensemble(cases)
    
    # 3. Zip
    create_submission_zip(cases)
