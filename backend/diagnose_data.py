import os
from pathlib import Path

data_dir = "./data"   # change to your data folder path

for split in ["train", "val", "test"]:
    split_path = Path(data_dir) / split
    if not split_path.exists():
        print(f"{split}: folder not found")
        continue
    
    # We will use "tb" as positive and "normal" as negative
    for label in ["tb", "normal"]:
        label_path = split_path / label
        if label_path.exists():
            count = len(list(label_path.glob("*.png")) +
                       list(label_path.glob("*.jpg")) +
                       list(label_path.glob("*.jpeg")))
            print(f"{split}/{label}: {count} images")
        else:
            print(f"{split}/{label}: folder not found")
