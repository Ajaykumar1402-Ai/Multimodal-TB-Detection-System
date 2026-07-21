import os
import hashlib
import cv2

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

def clean_dataset():
    print("="*50)
    print("TB Vision Pro - Cleaning Dataset")
    print("="*50)

    splits = ['train', 'val', 'test']
    classes = ['normal', 'tb']
    
    hashes = {}
    duplicates = []
    
    for split in splits:
        for c in classes:
            folder = os.path.join(DATA_DIR, split, c)
            if not os.path.exists(folder):
                continue
                
            for file in os.listdir(folder):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                path = os.path.join(folder, file)
                
                # Check duplicates
                h = get_hash(path)
                if h in hashes:
                    duplicates.append(path)
                else:
                    hashes[h] = path

    print(f"Found {len(duplicates)} duplicate images.")
    for d in duplicates:
        try:
            os.remove(d)
        except Exception as e:
            print(f"Could not remove {d}: {e}")
            
    print("Duplicates removed successfully. Please rerun validation.")

if __name__ == "__main__":
    clean_dataset()
