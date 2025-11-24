import os
import numpy as np
import nibabel as nib
from pathlib import Path
from ensemble_model import CardiacEnsemble
import pandas as pd

def main():
    base_dir = Path(r'C:\CardiacSeg')
    # Use the standard 2D validation folder
    pred_2d_folder = base_dir / 'inference_2d_validation_final'
    # Use the new 3D Fullres validation folder
    pred_3d_folder = base_dir / 'inference_3d_fullres_validation'
    output_folder = base_dir / 'ensemble_output_validation_fullres'
    labels_dir = base_dir / 'nnUNet_raw' / 'Dataset001_CardiacSeg' / 'labelsTr'
    
    if not output_folder.exists():
        output_folder.mkdir()

    # 1. Run Ensemble
    print("Running Ensemble (2D + 3D Fullres)...")
    # We pass the fullres folder as the '3d_lowres' argument because the class structure might expect that name,
    # or we can just instantiate it. The class uses these paths mainly for loading plans if needed, 
    # but ensemble_predictions uses the prediction files directly.
    ensemble = CardiacEnsemble(
        model_2d_folder=base_dir / 'nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0',
        model_3d_lowres_folder=base_dir / 'nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0'
    )
    
    # Find common cases
    cases_2d = set(f.stem.replace('.nii', '') for f in pred_2d_folder.glob('*.nii.gz'))
    cases_3d = set(f.stem.replace('.nii', '') for f in pred_3d_folder.glob('*.nii.gz'))
    common_cases = sorted(list(cases_2d.intersection(cases_3d)))
    
    print(f"Found {len(common_cases)} common cases: {common_cases}")
    
    if not common_cases:
        print("No common cases found to ensemble!")
        return

    # Use the same weights as before or adjust?
    # Previous: {'2d': 0.4, '3d_lowres': 0.6} (default in class)
    # But we might want to trust 3D Fullres more?
    # Let's stick to default first or maybe 0.5/0.5?
    # The user's previous ensemble result was 0.5765.
    # Let's use the default weights first.
    
    ensemble.ensemble_predictions(
        pred_2d_folder=pred_2d_folder,
        pred_3d_folder=pred_3d_folder,
        output_folder=output_folder,
        method='label_specific',
        case_list=common_cases
    )
    
    # 2. Evaluate
    print("\nEvaluating Ensemble Results...")
    metrics = []
    for case in common_cases:
        pred_path = output_folder / f"{case}.nii.gz"
        gt_path = labels_dir / f"{case}.nii.gz"
        
        if not gt_path.exists():
            # Try with _gt suffix if not found
            gt_path = labels_dir / f"{case}_gt.nii.gz"
            if not gt_path.exists():
                print(f"Warning: GT not found for {case}")
                continue
        
        # Load images
        pred_nii = nib.load(str(pred_path))
        gt_nii = nib.load(str(gt_path))
        
        pred_data = pred_nii.get_fdata()
        gt_data = gt_nii.get_fdata()
        
        # Calculate Dice for each label
        case_metrics = {'Case': case}
        for label in [1, 2, 3]: # Assuming labels 1, 2, 3
            pred_mask = (pred_data == label)
            gt_mask = (gt_data == label)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = pred_mask.sum() + gt_mask.sum()
            
            if union == 0:
                dice = 1.0 # Both empty
            else:
                dice = 2.0 * intersection / union
            
            case_metrics[f'Dice_L{label}'] = dice
            
        case_metrics['Mean_Dice'] = np.mean([case_metrics[f'Dice_L{label}'] for label in [1, 2, 3]])
        metrics.append(case_metrics)
        print(f"{case}: Mean Dice = {case_metrics['Mean_Dice']:.4f} (L1={case_metrics['Dice_L1']:.4f}, L2={case_metrics['Dice_L2']:.4f}, L3={case_metrics['Dice_L3']:.4f})")

    if metrics:
        df = pd.DataFrame(metrics)
        print("\n=== Ensemble Summary (2D + 3D Fullres) ===")
        print(df.describe())
        print(f"\nOverall Mean Dice: {df['Mean_Dice'].mean():.4f}")
        
        # Save results
        df.to_csv('ensemble_results_fullres.csv', index=False)
    else:
        print("No metrics calculated.")

if __name__ == '__main__':
    main()
