import os
import shutil
import json
from pathlib import Path

# Define the dataset structure
CATEGORIES = {
    "natural": [
        "Animals", "People", "Selfies", "Landscapes", "Buildings", "Food", "Vehicles"
    ],
    "documents": [
        "ID_Cards", "Passports", "Reports", "Screenshots", "PDF_Scans", "Invoices"
    ],
    "medical_other": [
        "CT", "MRI", "Ultrasound", "Dental_Xrays", "Hand_Xrays", "Knee_Xrays", "Spine_Xrays", "Abdominal_Xrays"
    ],
    "synthetic": [
        "AI_Generated_Images", "Cartoons", "Digital_Art"
    ],
    "hard_negatives": [
        # Reserved for manual additions of images that failed in production
    ]
}

DATA_ROOT = Path(os.path.dirname(__file__)).parent / 'data'
GUARD_DATASET_ROOT = DATA_ROOT / 'train_guard'

def initialize_structure():
    """Create the initial folder structure for the Guard AI V2.0 dataset."""
    print("Initializing Guard Dataset Structure...")
    
    # Create CXR positive folder
    cxr_dir = GUARD_DATASET_ROOT / 'cxr'
    cxr_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Non-CXR negative folder hierarchy
    non_cxr_dir = GUARD_DATASET_ROOT / 'non_cxr'
    non_cxr_dir.mkdir(parents=True, exist_ok=True)
    
    for main_cat, subcats in CATEGORIES.items():
        main_dir = non_cxr_dir / main_cat
        main_dir.mkdir(exist_ok=True)
        for subcat in subcats:
            sub_dir = main_dir / subcat
            sub_dir.mkdir(exist_ok=True)
            
    print(f"Structure created successfully at: {GUARD_DATASET_ROOT.resolve()}")

def import_images(source_dir, main_cat, subcat):
    """
    Import curated, openly licensed images into the dataset pipeline.
    Usage: import_images('/path/to/downloads/ct_scans', 'medical_other', 'CT')
    """
    if main_cat not in CATEGORIES and main_cat != "hard_negatives":
        raise ValueError(f"Unknown main category: {main_cat}")
        
    if main_cat != "hard_negatives" and subcat not in CATEGORIES[main_cat]:
        # Support manual addition of new categories seamlessly
        print(f"Notice: Adding new subcategory '{subcat}' to '{main_cat}'")
        
    dest_dir = GUARD_DATASET_ROOT / 'non_cxr' / main_cat / subcat
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
        
    imported = 0
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for f in source.rglob('*'):
        if f.is_file() and f.suffix.lower() in valid_exts:
            try:
                dest_file = dest_dir / f.name
                # Avoid overwriting by appending a unique ID if file exists
                if dest_file.exists():
                    import uuid
                    dest_file = dest_dir / f"{f.stem}_{uuid.uuid4().hex[:6]}{f.suffix}"
                    
                shutil.copy2(f, dest_file)
                imported += 1
            except Exception as e:
                print(f"Failed to import {f.name}: {e}")
                
    print(f"Imported {imported} images into {main_cat}/{subcat}.")

def integrate_hard_negative(image_path, rejection_reason):
    """
    Automatically integrates a manually verified hard negative into the dataset.
    """
    import_images(os.path.dirname(image_path), 'hard_negatives', rejection_reason.replace(" ", "_"))

if __name__ == '__main__':
    # When run directly, initialize the structure
    initialize_structure()
