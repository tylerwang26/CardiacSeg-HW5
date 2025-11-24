# filepath: c:\CardiacSeg\generate_dataset_json.py
import os
import json

dataset_dir = r"C:\CardiacSeg\nnUNet_raw\Dataset001_CardiacSeg"
json_path = os.path.join(dataset_dir, "dataset.json")

# 動態掃描 imagesTr 和 labelsTr
images_tr_dir = os.path.join(dataset_dir, "imagesTr")
labels_tr_dir = os.path.join(dataset_dir, "labelsTr")
images_ts_dir = os.path.join(dataset_dir, "imagesTs")

training = []
for img_file in sorted(os.listdir(images_tr_dir)):
    if img_file.endswith(".nii.gz"):
        case_id = img_file.split("_")[0]  # 假設檔名如 case_0000.nii.gz
        label_file = f"{case_id}.nii.gz"  # 調整為實際標籤檔名
        if os.path.exists(os.path.join(labels_tr_dir, label_file)):
            training.append({"image": f"./imagesTr/{img_file}", "label": f"./labelsTr/{label_file}"})

test = [f"./imagesTs/{f}" for f in sorted(os.listdir(images_ts_dir)) if f.endswith(".nii.gz")]

dataset_json = {
    "channel_names": {"0": "CT"},  # 調整為您的模態 (e.g., "MRI")
    "labels": {
        "background": 0,
        "label1": 1  # 調整為實際標籤 (e.g., "heart": 1, "aorta": 2)
    },
    "numTraining": len(training),
    "file_ending": ".nii.gz",
    "name": "CardiacSeg",
    "description": "Cardiac segmentation dataset",
    "reference": "Your reference",
    "licence": "Your licence",
    "release": "1.0",
    "tensorImageSize": "3D",
    "training": training,
    "test": test
}

with open(json_path, 'w') as f:
    json.dump(dataset_json, f, indent=4)
print(f"已生成 {json_path}。請手動驗證內容！")