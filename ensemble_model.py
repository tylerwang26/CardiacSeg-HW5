"""
Ensemble Model for Cardiac Segmentation
結合 2D 和 3D Low-Resolution models 的優勢

策略：
- 2D Model: 擅長檢測 Label 3 (右心室) - 48.63%
- 3D Low-Res: 擅長 Label 1/2 (心肌/左心室) - 88%/67%
- Label-specific weighting
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Dict
import json
from tqdm import tqdm

class CardiacEnsemble:
    """
    心臟分割 Ensemble 模型
    """
    
    def __init__(
        self,
        model_2d_folder: Path,
        model_3d_lowres_folder: Path,
        weights: Dict[str, float] = None
    ):
        """
        初始化 Ensemble
        
        Args:
            model_2d_folder: 2D model checkpoint 資料夾
            model_3d_lowres_folder: 3D lowres model checkpoint 資料夾
            weights: 各模型權重 {'2d': 0.4, '3d_lowres': 0.6}
        """
        self.model_2d_folder = Path(model_2d_folder)
        self.model_3d_lowres_folder = Path(model_3d_lowres_folder)
        
        # 預設權重（根據實驗結果調整）
        if weights is None:
            self.weights = {
                '2d': 0.4,         # 2D 對 Label 3 貢獻大
                '3d_lowres': 0.6   # 3D 對 Label 1/2 貢獻大
            }
        else:
            self.weights = weights
            
        # Label-specific 權重（進階策略）
        self.label_weights = {
            1: {'2d': 0.3, '3d_lowres': 0.7},  # 心肌：3D 更準
            2: {'2d': 0.4, '3d_lowres': 0.6},  # 左心室：3D 稍優
            3: {'2d': 0.8, '3d_lowres': 0.2}   # 右心室：2D 大幅領先
        }
        
        print("=" * 70)
        print("Cardiac Segmentation Ensemble")
        print("=" * 70)
        print(f"2D Model: {self.model_2d_folder}")
        print(f"3D LowRes Model: {self.model_3d_lowres_folder}")
        print(f"\nGlobal Weights: {self.weights}")
        print(f"Label-Specific Weights:")
        for label, w in self.label_weights.items():
            print(f"  Label {label}: 2D={w['2d']:.1f}, 3D={w['3d_lowres']:.1f}")
        print("=" * 70)
    
    def load_predictions(
        self, 
        pred_2d_path: Path, 
        pred_3d_path: Path
    ) -> Tuple[sitk.Image, sitk.Image]:
        """
        載入兩個模型的預測結果
        
        Returns:
            (pred_2d, pred_3d): SimpleITK Image objects
        """
        pred_2d = sitk.ReadImage(str(pred_2d_path))
        pred_3d = sitk.ReadImage(str(pred_3d_path))
        
        return pred_2d, pred_3d
    
    def simple_ensemble(
        self,
        pred_2d: np.ndarray,
        pred_3d: np.ndarray
    ) -> np.ndarray:
        """
        簡單加權平均 Ensemble（Global weights）
        
        Args:
            pred_2d: 2D prediction array (H, W, D)
            pred_3d: 3D prediction array (H, W, D)
            
        Returns:
            ensemble_pred: 合成預測 (H, W, D)
        """
        w_2d = self.weights['2d']
        w_3d = self.weights['3d_lowres']
        
        # 加權投票
        ensemble = np.zeros_like(pred_2d)
        
        for label in [1, 2, 3]:
            mask_2d = (pred_2d == label).astype(float)
            mask_3d = (pred_3d == label).astype(float)
            
            # 加權平均
            combined = w_2d * mask_2d + w_3d * mask_3d
            
            # 將這個 label 的機率加入
            ensemble = np.where(combined > 0.5, label, ensemble)
        
        return ensemble
    
    def label_specific_ensemble(
        self,
        pred_2d: np.ndarray,
        pred_3d: np.ndarray
    ) -> np.ndarray:
        """
        Label-specific 權重 Ensemble（進階版）
        
        每個 label 使用不同的權重組合
        """
        ensemble = np.zeros_like(pred_2d)
        
        # 對每個 label 分別處理
        for label in [1, 2, 3]:
            w_2d = self.label_weights[label]['2d']
            w_3d = self.label_weights[label]['3d_lowres']
            
            mask_2d = (pred_2d == label).astype(float)
            mask_3d = (pred_3d == label).astype(float)
            
            # 加權平均
            combined = w_2d * mask_2d + w_3d * mask_3d
            
            # 此 label 機率 > 0.5 的位置
            label_mask = combined > 0.5
            
            # 寫入 ensemble（後面的 label 會覆蓋前面的，所以順序很重要）
            ensemble[label_mask] = label
        
        return ensemble
    
    def post_process(self, pred: np.ndarray) -> np.ndarray:
        """
        後處理：移除小連通區域、形態學操作等
        
        TODO: 實作後處理邏輯
        """
        # 簡單版本：直接返回
        return pred
    
    def ensemble_predictions(
        self,
        pred_2d_folder: Path,
        pred_3d_folder: Path,
        output_folder: Path,
        method: str = 'label_specific',
        case_list: List[str] = None
    ):
        """
        對所有案例執行 ensemble
        
        Args:
            pred_2d_folder: 2D 預測資料夾
            pred_3d_folder: 3D 預測資料夾
            output_folder: 輸出資料夾
            method: 'simple' or 'label_specific'
            case_list: 要處理的案例列表（None = 全部）
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # 找到所有預測檔案
        pred_2d_files = sorted(pred_2d_folder.glob("*.nii.gz"))
        
        if case_list:
            pred_2d_files = [f for f in pred_2d_files 
                           if f.stem.replace('.nii', '') in case_list]
        
        print(f"\n處理 {len(pred_2d_files)} 個案例...")
        print(f"Ensemble 方法: {method}\n")
        
        results = []
        
        for pred_2d_path in tqdm(pred_2d_files):
            case_name = pred_2d_path.name
            pred_3d_path = pred_3d_folder / case_name
            
            if not pred_3d_path.exists():
                print(f"警告: 找不到 3D 預測 {pred_3d_path}")
                continue
            
            # 載入預測
            pred_2d_img = sitk.ReadImage(str(pred_2d_path))
            pred_3d_img = sitk.ReadImage(str(pred_3d_path))
            
            pred_2d_arr = sitk.GetArrayFromImage(pred_2d_img)
            pred_3d_arr = sitk.GetArrayFromImage(pred_3d_img)
            
            # 執行 Ensemble
            if method == 'simple':
                ensemble_arr = self.simple_ensemble(pred_2d_arr, pred_3d_arr)
            elif method == 'label_specific':
                ensemble_arr = self.label_specific_ensemble(pred_2d_arr, pred_3d_arr)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # 後處理
            ensemble_arr = self.post_process(ensemble_arr)
            
            # 保存結果
            ensemble_img = sitk.GetImageFromArray(ensemble_arr)
            ensemble_img.CopyInformation(pred_2d_img)  # 複製 metadata
            
            output_path = output_folder / case_name
            sitk.WriteImage(ensemble_img, str(output_path))
            
            results.append({
                'case': case_name,
                'output': str(output_path)
            })
        
        print(f"\n✓ Ensemble 完成！")
        print(f"輸出資料夾: {output_folder}")
        print(f"產生 {len(results)} 個預測檔案")
        
        return results


def main():
    """
    測試 Ensemble
    """
    base_dir = Path(__file__).parent
    
    # 模型資料夾
    model_2d = base_dir / "nnUNet_results" / "Dataset001_CardiacSeg" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0"
    model_3d_lowres = base_dir / "nnUNet_results" / "Dataset001_CardiacSeg" / "nnUNetTrainer__nnUNetPlans__3d_lowres" / "fold_0"
    
    # 預測資料夾（需要先執行 inference）
    pred_2d_folder = base_dir / "inference_2d_validation"
    pred_3d_folder = base_dir / "inference_output"
    
    # 輸出資料夾
    output_folder = base_dir / "ensemble_output"
    
    # 檢查資料夾是否存在
    if not pred_2d_folder.exists():
        print(f"錯誤: 2D 預測資料夾不存在: {pred_2d_folder}")
        print("請先執行 2D model inference")
        return
    
    if not pred_3d_folder.exists():
        print(f"錯誤: 3D 預測資料夾不存在: {pred_3d_folder}")
        print("請先執行 3D model inference")
        return
    
    # 創建 Ensemble
    ensemble = CardiacEnsemble(
        model_2d_folder=model_2d,
        model_3d_lowres_folder=model_3d_lowres
    )
    
    # 執行 Ensemble
    results = ensemble.ensemble_predictions(
        pred_2d_folder=pred_2d_folder,
        pred_3d_folder=pred_3d_folder,
        output_folder=output_folder,
        method='label_specific'  # 使用 label-specific 權重
    )
    
    print("\n" + "=" * 70)
    print("Ensemble 測試完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
