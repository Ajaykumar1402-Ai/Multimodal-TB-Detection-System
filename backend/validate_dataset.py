import os
import glob
import cv2
import hashlib
import numpy as np

DATA_DIR = "./data"

def get_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def validate():
    print("="*50)
    print("TB Vision Pro - Dataset Integrity Check")
    print("="*50)

    splits = ['train', 'val', 'test']
    classes = ['normal', 'tb']
    
    total_images = 0
    corrupted = []
    hashes = {}
    duplicates = []
    
    class_counts = {'normal': 0, 'tb': 0}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for split in splits:
        for c in classes:
            folder = os.path.join(DATA_DIR, split, c)
            if not os.path.exists(folder):
                continue
                
            for file in os.listdir(folder):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                path = os.path.join(folder, file)
                total_images += 1
                class_counts[c] += 1
                split_counts[split] += 1
                
                # Check corruption
                img = cv2.imread(path)
                if img is None:
                    corrupted.append(path)
                    continue
                
                # Check duplicates
                h = get_hash(path)
                if h in hashes:
                    duplicates.append((path, hashes[h]))
                else:
                    hashes[h] = path

    print(f"\nTotal Images Found: {total_images}")
    print(f"Class Distribution: Normal={class_counts['normal']}, TB={class_counts['tb']}")
    print(f"Split Distribution: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    
    print(f"\nCorrupted Images: {len(corrupted)}")
    for c in corrupted[:5]:
        print(f"  - {c}")
        
    print(f"Duplicate Images: {len(duplicates)}")
    for d in duplicates[:5]:
        print(f"  - {d[0]} is duplicate of {d[1]}")
        
    if len(corrupted) > 0 or len(duplicates) > 0:
        print("\n[FAILED] Dataset validation failed. Fix errors before training.")
        return False
        
    print("\n[PASSED] Dataset validation successful.")
    return True

if __name__ == "__main__":
    validate()
