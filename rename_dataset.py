# cite: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

import os
import subprocess
import multiprocessing
import json
import nibabel as nib
import numpy as np
import json as pyjson
import zipfile
import shutil
import tempfile
from urllib.parse import urlparse
import subprocess
import sys
import argparse
try:
    import requests
except Exception:
    requests = None
try:
    import gdown  # for Google Drive links
except Exception:
    gdown = None
try:
    from tqdm import tqdm
except ImportError:
    print("安裝 tqdm 以顯示進度條...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 跨平台：使用當前腳本所在目錄作為 base_dir
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"腳本 base_dir: {base_dir}")  # 診斷打印

# nnU-Net v2 環境變數設定
os.environ["NNUNET_RAW"] = os.path.join(base_dir, "nnUNet_raw")
os.environ["NNUNET_PREPROCESSED"] = os.path.join(base_dir, "nnUNet_preprocessed")
os.environ["NNUNET_RESULTS"] = os.path.join(base_dir, "nnUNet_results")

# 數據集 ID 和名稱
dataset_id = "001"
dataset_name = "CardiacSeg"
dataset_dir = os.path.join(os.environ["NNUNET_RAW"], f"Dataset{dataset_id}_{dataset_name}")
images_tr_dir = os.path.join(dataset_dir, "imagesTr")
labels_tr_dir = os.path.join(dataset_dir, "labelsTr")
images_ts_dir = os.path.join(dataset_dir, "imagesTs")
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)
os.makedirs(images_ts_dir, exist_ok=True)

def install_dependencies():
    # 保留為空：環境安裝改由 setup_environment.ps1 處理
    return

def _is_already_standard(fname: str, is_image: bool) -> bool:
    if not fname.endswith('.nii.gz'):
        return False
    if is_image:
        return fname.count('_') >= 2 and fname.endswith('_0000.nii.gz')
    # label: should not contain _gt and not end with _0000
    return (not fname.endswith('_0000.nii.gz')) and ('_gt' not in fname)

def _all_files_standard(images_tr_dir, labels_tr_dir, images_ts_dir) -> bool:
    for f in os.listdir(images_tr_dir):
        if f.endswith('.nii.gz') and not _is_already_standard(f, True):
            return False
    for f in os.listdir(labels_tr_dir):
        if f.endswith('.nii.gz') and not _is_already_standard(f, False):
            return False
    for f in os.listdir(images_ts_dir):
        if f.endswith('.nii.gz') and not _is_already_standard(f, True):
            return False
    return True

def _quick_dataset_json_ok(json_path: str, images_tr_dir: str, labels_tr_dir: str) -> bool:
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        if 'labels' not in js or 'background' not in js['labels']:
            return False
        # basic count match
        train_images = [x for x in os.listdir(images_tr_dir) if x.endswith('_0000.nii.gz')]
        train_labels = [x for x in os.listdir(labels_tr_dir) if x.endswith('.nii.gz') and not x.endswith('_0000.nii.gz')]
        if len(js.get('training', [])) != min(len(train_images), len(train_labels)):
            return False
        return True
    except Exception:
        return False

def generate_dataset_json(force_regenerate=False, fast_skip=True):
    json_path = os.path.join(dataset_dir, "dataset.json")
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    labels_tr_dir = os.path.join(dataset_dir, "labelsTr")
    images_ts_dir = os.path.join(dataset_dir, "imagesTs")

    # 快速跳過：檔名都標準 & dataset.json 有效 且未要求強制
    if fast_skip and not force_regenerate and _all_files_standard(images_tr_dir, labels_tr_dir, images_ts_dir) and _quick_dataset_json_ok(json_path, images_tr_dir, labels_tr_dir):
        print("✓ 檢查：檔案與 dataset.json 均已標準，快速跳過重建。")
        return

    if os.path.exists(json_path) and not force_regenerate:
        print(f"dataset.json 已存在於 {json_path}，但因檢查未達快速跳過條件，重新掃描生成。")

    training = []
    all_labels = set()

    print("\n處理訓練影像和標籤...")
    img_files = sorted([f for f in os.listdir(images_tr_dir) if f.endswith('.nii.gz')])
    for img_file in tqdm(img_files, desc="🔄 掃描訓練資料", unit="檔案"):
        if '_' not in img_file or not img_file.split('_')[-1].startswith('0000'):
            old_img = os.path.join(images_tr_dir, img_file)
            new_img_file = img_file.replace('.nii.gz', '_0000.nii.gz')
            new_img = os.path.join(images_tr_dir, new_img_file)
            os.rename(old_img, new_img)
            tqdm.write(f"  ✓ 重命名影像：{img_file} -> {new_img_file}")
            std_img_file = new_img_file
        else:
            std_img_file = img_file

        base_case = std_img_file.replace('_0000.nii.gz', '').replace('.nii.gz', '')
        label_candidates = [f for f in os.listdir(labels_tr_dir) if f.startswith(base_case) and f.endswith('.nii.gz')]
        if not label_candidates:
            continue
        original_label = label_candidates[0]
        std_label_file = f"{base_case}.nii.gz"
        old_label = os.path.join(labels_tr_dir, original_label)
        new_label = os.path.join(labels_tr_dir, std_label_file)
        if original_label != std_label_file:
            os.rename(old_label, new_label)
            tqdm.write(f"  ✓ 重命名標籤：{original_label} -> {std_label_file}")

        training.append({"image": f"./imagesTr/{std_img_file}", "label": f"./labelsTr/{std_label_file}"})
        try:
            lbl_img = nib.load(new_label)
            lbl_data = lbl_img.get_fdata()
            unique_labels = np.unique(lbl_data)
            # 以 int() 轉成原生 Python int
            for ul in unique_labels:
                try:
                    all_labels.add(int(ul))
                except Exception:
                    pass
        except Exception as e:
            tqdm.write(f"  ⚠ 載入標籤失敗 {new_label}: {e}")

    print("\n處理測試影像...")
    test = []
    test_files = sorted([f for f in os.listdir(images_ts_dir) if f.endswith('.nii.gz')])
    for f in tqdm(test_files, desc="🔄 掃描測試資料", unit="檔案"):
        if '_' not in f or not f.split('_')[-1].startswith('0000'):
            # 重命名測試影像也加 _0000
            old_test = os.path.join(images_ts_dir, f)
            new_test_file = f.replace('.nii.gz', '_0000.nii.gz')
            new_test = os.path.join(images_ts_dir, new_test_file)
            if not os.path.exists(new_test):
                try:
                    os.rename(old_test, new_test)
                    tqdm.write(f"  ✓ 重命名測試影像：{f} -> {new_test_file}")
                except FileExistsError:
                    pass
            test.append(f"./imagesTs/{new_test_file}")
        else:
            test.append(f"./imagesTs/{f}")

    # 建立標籤名稱 (背景 + 其它) 正確方向: 名稱 -> 整數值
    label_name_map = {"background": 0}
    for l in sorted(all_labels):
        if l == 0:
            continue
        label_name_map[f"label{l}"] = int(l)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": label_name_map,
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Cardiac segmentation dataset",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "tensorImageSize": "3D",
        "training": training,
        "test": test
    }

    def _normalize(o):
        if isinstance(o, dict):
            return {str(k): _normalize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_normalize(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    dataset_json = _normalize(dataset_json)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
    print(f"\n✓ 已生成/更新 dataset.json 於 {json_path}")
    print(f"  - 訓練資料數量：{len(training)}")
    print(f"  - 測試資料數量：{len(test)}")
    print(f"  - 檢測到的標籤：{sorted(all_labels)}")

def _has_data():
    def _has_nii(p):
        return os.path.isdir(p) and any(fn.endswith('.nii') or fn.endswith('.nii.gz') for fn in os.listdir(p))
    return _has_nii(images_tr_dir) and _has_nii(labels_tr_dir)

def _download_file(url, dst):
    # Prefer gdown for Google Drive links
    if 'drive.google.com' in url or 'uc?id=' in url:
        gd = gdown
        if gd is None:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown==4.6.0'])
                import gdown as gd  # type: ignore
            except Exception as e:
                raise RuntimeError(f"無法安裝 gdown 用於 Google Drive 下載: {e}")
        try:
            print(f"📥 下載中：{os.path.basename(dst)}")
            gd.download(url, dst, quiet=False)
            if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                raise RuntimeError("gdown 下載結果無效或為空檔案。")
            return
        except Exception as e:
            raise RuntimeError(f"gdown 下載失敗: {e}")
    # Fallback to requests for regular URLs
    if requests is None:
        raise RuntimeError("requests 未安裝，無法下載。請先安裝或改用本機檔案來源。")
    print(f"📥 下載中：{os.path.basename(dst)}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(dst, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dst)) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def _extract_zip(zip_path, target_dir):
    print(f"📦 解壓縮：{os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="解壓縮檔案", unit="檔案"):
            zf.extract(member, target_dir)

def _copy_nii_tree(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    nii_files = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.endswith('.nii') or fn.endswith('.nii.gz'):
                nii_files.append((root, fn))
    
    if nii_files:
        print(f"📋 複製 {len(nii_files)} 個 NIfTI 檔案到 {os.path.basename(dst_dir)}")
        for root, fn in tqdm(nii_files, desc="複製檔案", unit="檔案"):
            src = os.path.join(root, fn)
            dst = os.path.join(dst_dir, fn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

def _find_candidate_dirs(root_dir):
    image_dirs = []
    label_dirs = []
    test_dirs = []
    for current_root, dirs, _ in os.walk(root_dir):
        name = os.path.basename(current_root).lower()
        if any(k in name for k in ['imagestr', 'image_tr', 'train_images', 'training_image', 'trainimage', 'image']):
            image_dirs.append(current_root)
        if any(k in name for k in ['labelstr', 'label_tr', 'train_labels', 'training_label', 'gt', 'label']):
            label_dirs.append(current_root)
        if any(k in name for k in ['imagests', 'image_ts', 'test_images', 'testing_image', 'testimage', 'images_ts']):
            test_dirs.append(current_root)
    return image_dirs, label_dirs, test_dirs

def ensure_original_data():
    if _has_data():
        print("偵測到既有資料，跳過下載。")
        return
    print("未找到原始資料。嘗試依設定下載並整理至 nnU-Net 結構...")

    # 讀取可選的設定檔 data_sources.json（位於腳本目錄）
    config_path = os.path.join(base_dir, 'data_sources.json')
    cfg = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = pyjson.load(f)

    # 支援兩種設定：
    # 1) dataset_zip: 包含 imagesTr/labelsTr/imagesTs 結構的壓縮檔 URL 或本機路徑
    # 2) train_images_url / train_labels_url / test_images_url：各自的壓縮檔或資料夾 URL/路徑
    def _is_url(path_or_url):
        try:
            return urlparse(path_or_url).scheme in ('http', 'https')
        except Exception:
            return False

    def _prepare_from_zip(path_or_url):
        tmpdir = tempfile.mkdtemp(prefix='aicup_ds_')
        try:
            zip_path = os.path.join(tmpdir, 'dataset.zip')
            if _is_url(path_or_url):
                print(f"下載資料壓縮檔: {path_or_url}")
                _download_file(path_or_url, zip_path)
            else:
                zip_path = path_or_url
            print("解壓縮...")
            _extract_zip(zip_path, tmpdir)
            # 嘗試匹配子資料夾
            _copy_nii_tree(os.path.join(tmpdir, 'imagesTr'), images_tr_dir)
            _copy_nii_tree(os.path.join(tmpdir, 'labelsTr'), labels_tr_dir)
            if os.path.isdir(os.path.join(tmpdir, 'imagesTs')):
                _copy_nii_tree(os.path.join(tmpdir, 'imagesTs'), images_ts_dir)
            # 若標準資料夾不存在，嘗試基於名稱關鍵字搜尋
            if not any(os.scandir(images_tr_dir)) or not any(os.scandir(labels_tr_dir)):
                imgs, lbls, tsts = _find_candidate_dirs(tmpdir)
                if imgs:
                    for d in imgs:
                        _copy_nii_tree(d, images_tr_dir)
                if lbls:
                    for d in lbls:
                        _copy_nii_tree(d, labels_tr_dir)
                if tsts:
                    for d in tsts:
                        _copy_nii_tree(d, images_ts_dir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # 優先 dataset_zip
    used = False
    if cfg and 'dataset_zip' in cfg and cfg['dataset_zip']:
        _prepare_from_zip(cfg['dataset_zip'])
        used = True
    else:
        # 分項來源
        tri = (cfg.get('train_images_url') if cfg else os.environ.get('TRAIN_IMAGES_URL'))
        trl = (cfg.get('train_labels_url') if cfg else os.environ.get('TRAIN_LABELS_URL'))
        tsi = (cfg.get('test_images_url') if cfg else os.environ.get('TEST_IMAGES_URL'))
        for name, url_or_path, target in (
            ('train images', tri, images_tr_dir),
            ('train labels', trl, labels_tr_dir),
            ('test images', tsi, images_ts_dir),
        ):
            if not url_or_path:
                continue
            print(f"準備 {name} 來源: {url_or_path}")
            if os.path.isdir(url_or_path):
                _copy_nii_tree(url_or_path, target)
                used = True
            elif url_or_path.endswith('.zip') or _is_url(url_or_path):
                _prepare_from_zip(url_or_path)
                used = True
            else:
                print(f"警告：未知的來源格式，已跳過: {url_or_path}")

    if not _has_data():
        print(f"未能自動下載/整理資料。請在 {base_dir} 建立 data_sources.json，例如:\n" \
              "{\n  \"dataset_zip\": \"https://.../your_dataset.zip\"\n}\n" \
              "或提供 train_images_url / train_labels_url / test_images_url。亦可手動放入 nnUNet_raw/Dataset001_CardiacSeg 底下。")
    else:
        print("原始資料已到位。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize dataset and generate nnU-Net dataset.json')
    parser.add_argument('--force', action='store_true', help='強制重建 dataset.json 與檔名')
    parser.add_argument('--no-fast-skip', action='store_true', help='停用快速跳過檢查')
    args = parser.parse_args()

    multiprocessing.freeze_support()
    install_dependencies()
    ensure_original_data()
    generate_dataset_json(force_regenerate=args.force, fast_skip=not args.no_fast_skip)
    print("\n" + "="*60)
    print("✓ 完成：資料檢查/重命名與 dataset.json 處理")
    print("  後續預處理/訓練請使用 nnunet_train.py")
    print("="*60)