import os
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd

def main():
    base_dir = Path(r'C:\CardiacSeg')
    pred_folder = base_dir / 'inference_3d_fullres_validation'
    labels_dir = base_dir / 'nnUNet_raw' / 'Dataset001_CardiacSeg' / 'labelsTr'
    
    print("Evaluating 3D Fullres Results...")
    metrics = []
    
    # Get list of cases
    cases = sorted([f.stem.replace('.nii', '') for f in pred_folder.glob('*.nii.gz')])
    
    for case in cases:
        pred_path = pred_folder / f"{case}.nii.gz"
        gt_path = labels_dir / f"{case}.nii.gz"
        
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
        for label in [1, 2, 3]:
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
        print("\n=== 3D Fullres Summary ===")
        print(df.describe())
        print(f"\nOverall Mean Dice: {df['Mean_Dice'].mean():.4f}")
    else:
        print("No metrics calculated.")

if __name__ == '__main__':
    main()
