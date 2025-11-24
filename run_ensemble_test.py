import os
import numpy as np
import nibabel as nib
from pathlib import Path
from ensemble_model import CardiacEnsemble
import pandas as pd

def main():
    base_dir = Path(r'C:\CardiacSeg')
    pred_2d_folder = base_dir / 'inference_2d_validation_final'
    pred_3d_folder = base_dir / 'inference_3d_lowres_validation'
    output_folder = base_dir / 'ensemble_output_validation'
    labels_dir = base_dir / 'nnUNet_raw' / 'Dataset001_CardiacSeg' / 'labelsTr'
    
    # 1. Run Ensemble
    print("Running Ensemble...")
    ensemble = CardiacEnsemble(
        model_2d_folder=base_dir / 'nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0',
        model_3d_lowres_folder=base_dir / 'nnUNet_results/Dataset001_CardiacSeg/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0'
    )
    
    # Find common cases
    cases_2d = set(f.stem.replace('.nii', '') for f in pred_2d_folder.glob('*.nii.gz'))
    cases_3d = set(f.stem.replace('.nii', '') for f in pred_3d_folder.glob('*.nii.gz'))
    common_cases = sorted(list(cases_2d.intersection(cases_3d)))
    
    print(f"Found {len(common_cases)} common cases: {common_cases}")
    
    if not common_cases:
        print("No common cases found to ensemble!")
        return

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
            print(f"Warning: GT not found for {case}")
            continue
            
        pred_nii = nib.load(str(pred_path))
        gt_nii = nib.load(str(gt_path))
        pred_data = pred_nii.get_fdata()
        gt_data = gt_nii.get_fdata()
        
        case_metrics = {'case': case}
        mean_dice = 0
        for label in [1, 2, 3]:
            dice, iou = dice_iou_per_label(pred_data, gt_data, label)
            case_metrics[f'Dice_L{label}'] = dice
            mean_dice += dice
        case_metrics['Mean_Dice'] = mean_dice / 3
        metrics.append(case_metrics)
        print(f"{case}: Mean Dice = {case_metrics['Mean_Dice']:.4f} (L1={case_metrics['Dice_L1']:.4f}, L2={case_metrics['Dice_L2']:.4f}, L3={case_metrics['Dice_L3']:.4f})")
        
    if metrics:
        df = pd.DataFrame(metrics)
        print("\nEnsemble Evaluation Summary:")
        print(df.describe())
        df.to_csv(output_folder / 'evaluation_metrics.csv', index=False)
        print(f"Metrics saved to {output_folder / 'evaluation_metrics.csv'}")
    else:
        print("No metrics calculated.")

def dice_iou_per_label(pred, gt, label_val):
    pred_mask = (pred == label_val)
    gt_mask = (gt == label_val)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    union = pred_sum + gt_sum - inter
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    return dice, iou

if __name__ == '__main__':
    main()
