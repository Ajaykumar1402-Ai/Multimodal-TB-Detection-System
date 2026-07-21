import os
import urllib.request
import zipfile
import shutil
import glob
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')

URL_MONTGOMERY = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
URL_SHENZHEN = "http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"
KAGGLE_ZIP = r"C:\Users\itsak\OneDrive\tb\archive.zip"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    else:
        print(f"File {dest} already exists.")

def extract_zip(zip_path, dest_dir):
    print(f"Extracting {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print("Extraction complete.")

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def setup_directories():
    splits = ['train', 'val', 'test']
    classes = ['tb', 'normal']
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)
            
def process_montgomery(temp_dir):
    print("Processing Montgomery Dataset (Test set only)...")
    # All Montgomery images go to Test
    montgomery_dir = os.path.join(temp_dir, 'MontgomerySet')
    if not os.path.exists(montgomery_dir):
        # Extract
        zip_path = os.path.join(TEMP_DIR, 'NLM-MontgomeryCXRSet.zip')
        extract_zip(zip_path, temp_dir)
    
    cxr_dir = os.path.join(montgomery_dir, 'CXR_png')
    images = glob.glob(os.path.join(cxr_dir, '*.png'))
    for img in images:
        filename = os.path.basename(img)
        # labels: filename ending _1.png = TB+, _0.png = TB-
        if filename.endswith('_1.png'):
            shutil.copy(img, os.path.join(DATA_DIR, 'test', 'tb', filename))
        elif filename.endswith('_0.png'):
            shutil.copy(img, os.path.join(DATA_DIR, 'test', 'normal', filename))

def process_train_val(temp_dir):
    print("Processing Shenzhen & Kaggle Datasets (Train/Val sets)...")
    tb_images = []
    normal_images = []
    
    # 1. Shenzhen Dataset
    shenzhen_dir = os.path.join(temp_dir, 'ChinaSet_AllFiles')
    if not os.path.exists(shenzhen_dir):
        zip_path = os.path.join(TEMP_DIR, 'ChinaSet_AllFiles.zip')
        extract_zip(zip_path, temp_dir)
        
    shenzhen_cxr = os.path.join(shenzhen_dir, 'CXR_png')
    sz_images = glob.glob(os.path.join(shenzhen_cxr, '*.png'))
    for img in sz_images:
        filename = os.path.basename(img)
        if filename.endswith('_1.png'):
            tb_images.append(img)
        elif filename.endswith('_0.png'):
            normal_images.append(img)
            
    # 2. Kaggle Dataset
    # Already extracted to temp_dir? No, let's extract it now
    kaggle_extract_dir = os.path.join(temp_dir, 'Kaggle')
    if not os.path.exists(kaggle_extract_dir):
        extract_zip(KAGGLE_ZIP, kaggle_extract_dir)
        
    # Kaggle struct: TB_Chest_Radiography_Database/Tuberculosis/ and Normal/
    kaggle_tb = glob.glob(os.path.join(kaggle_extract_dir, '**', 'Tuberculosis', '*.png'), recursive=True)
    kaggle_normal = glob.glob(os.path.join(kaggle_extract_dir, '**', 'Normal', '*.png'), recursive=True)
    
    tb_images.extend(kaggle_tb)
    normal_images.extend(kaggle_normal)
    
    # Shuffle and split (approx 80/20 train/val)
    random.seed(42)
    random.shuffle(tb_images)
    random.shuffle(normal_images)
    
    tb_train_idx = int(len(tb_images) * 0.82) # Adjust to get close to 820
    normal_train_idx = int(len(normal_images) * 0.82)
    
    tb_train = tb_images[:tb_train_idx]
    tb_val = tb_images[tb_train_idx:]
    
    normal_train = normal_images[:normal_train_idx]
    normal_val = normal_images[normal_train_idx:]
    
    print(f"TB: {len(tb_train)} train, {len(tb_val)} val")
    print(f"Normal: {len(normal_train)} train, {len(normal_val)} val")
    
    def copy_files(file_list, split, cls):
        for i, img in enumerate(file_list):
            filename = os.path.basename(img)
            # Prefix with index to avoid name collisions between datasets
            new_filename = f"{cls}_{split}_{i}_{filename}"
            shutil.copy(img, os.path.join(DATA_DIR, split, cls, new_filename))
            
    copy_files(tb_train, 'train', 'tb')
    copy_files(tb_val, 'val', 'tb')
    copy_files(normal_train, 'train', 'normal')
    copy_files(normal_val, 'val', 'normal')

def main():
    print("Starting data collection...")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Clear existing splits
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        clear_directory(split_dir)
        
    setup_directories()
    
    # Download datasets
    montgomery_zip = os.path.join(TEMP_DIR, 'NLM-MontgomeryCXRSet.zip')
    shenzhen_zip = os.path.join(TEMP_DIR, 'ChinaSet_AllFiles.zip')
    
    download_file(URL_MONTGOMERY, montgomery_zip)
    download_file(URL_SHENZHEN, shenzhen_zip)
    
    process_montgomery(TEMP_DIR)
    process_train_val(TEMP_DIR)
    
    print("Data collection and splitting complete.")

if __name__ == "__main__":
    main()
