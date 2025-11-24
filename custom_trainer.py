"""
創建自定義 nnUNet Trainer 類，可以從環境變數讀取 epochs 數量
"""
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerCustomEpochs(nnUNetTrainer):
    """
    允許透過環境變數 nnUNet_n_epochs 設定 epochs 數量的自定義 Trainer
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, **kwargs)
        
        # 從環境變數讀取 epochs（如果有設定）
        custom_epochs = os.environ.get('nnUNet_n_epochs', None)
        if custom_epochs is not None:
            try:
                self.num_epochs = int(custom_epochs)
                print(f"✓ 使用自定義 epochs 數量: {self.num_epochs}")
            except ValueError:
                print(f"警告: nnUNet_n_epochs='{custom_epochs}' 無法轉換為整數，使用預設值 {self.num_epochs}")
        else:
            print(f"使用預設 epochs 數量: {self.num_epochs}")

if __name__ == '__main__':
    # 測試
    print("自定義 Trainer 類別建立成功！")
    print(f"類別名稱: {nnUNetTrainerCustomEpochs.__name__}")
    print("\n使用方式：")
    print("1. 設定環境變數: export nnUNet_n_epochs=80 (Linux/Mac) 或 $env:nnUNet_n_epochs=\"80\" (Windows)")
    print("2. 使用 -tr 參數指定 trainer: nnUNetv2_train ... -tr nnUNetTrainerCustomEpochs")
