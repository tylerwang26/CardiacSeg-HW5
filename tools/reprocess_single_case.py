"""重新處理單一 case 的 preprocessed 資料"""
import os
import sys
from pathlib import Path

# 設定環境變數
os.environ['nnUNet_raw'] = r'c:\CardiacSeg\nnUNet_raw'
os.environ['nnUNet_preprocessed'] = r'c:\CardiacSeg\nnUNet_preprocessed'
os.environ['nnUNet_results'] = r'c:\CardiacSeg\nnUNet_results'

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset, preprocess_dataset
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import load_json
import numpy as np

def reprocess_single_case(dataset_id: int, case_identifier: str, configuration: str = '2d'):
    """
    重新處理單一 case
    
    Args:
        dataset_id: 資料集 ID (例如 1 代表 Dataset001)
        case_identifier: case 名稱 (例如 'patient0001')
        configuration: 配置名稱 ('2d', '3d_fullres', '3d_lowres')
    """
    dataset_name = f'Dataset{dataset_id:03d}_CardiacSeg'
    preprocessed_folder = Path(nnUNet_preprocessed) / dataset_name
    
    # 刪除損壞的 npz 檔案
    config_folder = preprocessed_folder / f'nnUNetPlans_{configuration}'
    corrupted_file = config_folder / f'{case_identifier}.npz'
    
    if corrupted_file.exists():
        print(f"刪除損壞檔案: {corrupted_file}")
        corrupted_file.unlink()
    
    # 使用 nnUNet 的 API 重新 preprocess 整個配置
    # 由於 nnUNet 沒有提供單一 case 的 API，我們需要重新跑整個配置
    # 但它會跳過已存在的檔案
    print(f"\n重新處理 {dataset_name} 的 {configuration} 配置...")
    print("注意：只會重新生成缺失或損壞的檔案\n")
    
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset
    
    # 讀取 plans 檔案
    plans_file = preprocessed_folder / 'nnUNetPlans.json'
    if not plans_file.exists():
        print(f"錯誤: Plans 檔案不存在: {plans_file}")
        return False
    
    plans = load_json(plans_file)
    
    # 只處理指定的配置
    configurations = [configuration]
    
    print(f"開始重新 preprocessing {configuration}...")
    try:
        preprocess_dataset(
            dataset_id=dataset_id,
            plans_identifier='nnUNetPlans',
            configurations=configurations,
            num_processes=(4,),  # 使用 4 個 CPU 核心 (必須是 tuple)
            verbose=True
        )
        print(f"\n完成！檢查修復後的檔案...")
        
        # 驗證修復
        if corrupted_file.exists():
            try:
                data = np.load(corrupted_file)
                print(f"✓ {case_identifier}.npz 已成功修復！")
                print(f"  包含的 keys: {list(data.keys())}")
                return True
            except Exception as e:
                print(f"✗ 檔案仍然損壞: {e}")
                return False
        else:
            print(f"✗ 檔案未生成")
            return False
            
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 修復 patient0001 的 2d 配置
    success = reprocess_single_case(
        dataset_id=1,
        case_identifier='patient0001',
        configuration='2d'
    )
    
    sys.exit(0 if success else 1)
