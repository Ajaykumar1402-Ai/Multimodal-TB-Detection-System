"""
=============================================================
TB-Vision Pro: Dataset Preparation for DenseNet-121 Training
=============================================================
Combines:
  1. Montgomery County CXR Set (NIH OpenI)
  2. Shenzhen CXR Set (NIH OpenI)
  3. Kaggle TB Chest X-Ray Dataset (manual download)
  4. Synthetic data (fallback)

Output: backend/data/train|val|test / normal|tb/
=============================================================
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import random
import json
from pathlib import Path

# ── CONFIG ──
BASE_DIR      = Path(__file__).parent
TRAINING_DATA = BASE_DIR / "training_data"
OUTPUT_DIR    = BASE_DIR / "data"

KAGGLE_DIR    = TRAINING_DATA / "kaggle"
MONTGOMERY_DIR= TRAINING_DATA / "montgomery"
SHENZHEN_DIR  = TRAINING_DATA / "shenzhen"
SYNTHETIC_DIR = TRAINING_DATA / "synthetic"

RANDOM_SEED   = 42
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15

random.seed(RANDOM_SEED)


def log(msg):
    """Safe print for Windows cp1252 consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


def download_file(url, dest, label):
    """Download only if dest is missing or stub (<100KB)."""
    if dest.exists() and dest.stat().st_size > 100_000:
        log(f"  [SKIP] {label} already downloaded.")
        return True
    log(f"  >> Downloading {label}...")
    try:
        def progress(bn, bs, ts):
            dl = bn * bs
            pct = min(100, dl * 100 / ts) if ts > 0 else 0
            sys.stdout.write(f"\r    {pct:.1f}%  ({dl//1024//1024} MB)")
            sys.stdout.flush()
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
        return dest.stat().st_size > 100_000
    except Exception as e:
        log(f"\n  FAILED: {e}")
        return False


def extract_zip(zip_path, dest, label):
    marker = dest / ".extracted"
    if marker.exists():
        log(f"  [SKIP] {label} already extracted.")
        return
    if not zip_path.exists() or zip_path.stat().st_size < 100_000:
        log(f"  [SKIP] {label} zip stub/missing, skipping.")
        return
    log(f"  >> Extracting {label}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest)
        marker.touch()
        log(f"  [OK] {label} extracted.")
    except Exception as e:
        log(f"  FAILED: {e}")


def copy_image(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def make_splits(images, train_r=0.70, val_r=0.15):
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return images[:n_train], images[n_train:n_train+n_val], images[n_train+n_val:]


# ── SETUP ──
MONTGOMERY_DIR.mkdir(parents=True, exist_ok=True)
SHENZHEN_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

log("\n" + "="*60)
log("  TB-Vision Pro: Dataset Preparation")
log("="*60)

# ── STEP 1: DOWNLOAD NIH ──
log("\n[ STEP 1 ] Downloading NIH CXR Datasets")
log("-" * 50)

mont_zip = TRAINING_DATA / "montgomery.zip"
shen_zip = TRAINING_DATA / "shenzhen.zip"

mont_ok = download_file(
    "https://openi.nlm.nih.gov/ftp/NLM-MontgomerySet.zip",
    mont_zip, "Montgomery County CXR"
)
if mont_ok:
    extract_zip(mont_zip, MONTGOMERY_DIR, "Montgomery")

shen_ok = download_file(
    "https://openi.nlm.nih.gov/ftp/ChinaSet_AllFiles.zip",
    shen_zip, "Shenzhen CXR"
)
if shen_ok:
    extract_zip(shen_zip, SHENZHEN_DIR, "Shenzhen")


# ── STEP 2: COLLECT IMAGES ──
log("\n[ STEP 2 ] Collecting Images")
log("-" * 50)

normal_images = []
tb_images     = []


def collect_nih(base_dir, name):
    n, t = 0, 0
    for ext in ["*.png", "*.jpg"]:
        for img in base_dir.rglob(ext):
            stem = img.stem
            if stem.endswith("_0"):
                normal_images.append(img); n += 1
            elif stem.endswith("_1"):
                tb_images.append(img); t += 1
    if n + t > 0:
        log(f"  {name:20s}: {n} normal, {t} TB")


def collect_synthetic(base_dir):
    if not base_dir.exists():
        return
    n, t = 0, 0
    for img in base_dir.glob("*.png"):
        stem = img.stem
        if stem.endswith("_0"):
            normal_images.append(img); n += 1
        elif stem.endswith("_1"):
            tb_images.append(img); t += 1
    if n + t > 0:
        log(f"  {'Synthetic':20s}: {n} normal, {t} TB")


def collect_kaggle(base_dir):
    if not base_dir.exists() or not any(base_dir.iterdir()):
        log(f"  Kaggle: NOT FOUND at {base_dir}")
        log(f"    Place Kaggle TB dataset in: {base_dir}")
        log(f"    Continuing with available data only...")
        return
    n, t = 0, 0
    norm_kw = ["normal", "negative"]
    tb_kw   = ["tuberculosis", "tb", "positive"]
    for folder in base_dir.rglob("*"):
        if not folder.is_dir():
            continue
        fname = folder.name.lower()
        imgs = (list(folder.glob("*.png")) +
                list(folder.glob("*.jpg")) +
                list(folder.glob("*.jpeg")))
        if any(kw in fname for kw in norm_kw):
            normal_images.extend(imgs); n += len(imgs)
        elif any(kw in fname for kw in tb_kw):
            tb_images.extend(imgs); t += len(imgs)
    if n + t > 0:
        log(f"  {'Kaggle':20s}: {n} normal, {t} TB")
    else:
        log(f"  Kaggle: No images detected (ensure Normal/Tuberculosis subfolders)")


collect_nih(MONTGOMERY_DIR, "Montgomery")
collect_nih(SHENZHEN_DIR, "Shenzhen")
collect_kaggle(KAGGLE_DIR)
collect_synthetic(SYNTHETIC_DIR)

total = len(normal_images) + len(tb_images)
log(f"\n  TOTAL: {len(normal_images)} normal, {len(tb_images)} TB ({total} images)")

if total == 0:
    log("  No images found. Exiting.")
    sys.exit(1)

if len(tb_images) < 50:
    log(f"  WARNING: Only {len(tb_images)} TB images. Add Kaggle for best accuracy.")


# ── STEP 3: SPLIT ──
log("\n[ STEP 3 ] Stratified 70/15/15 Split")
log("-" * 50)

norm_train, norm_val, norm_test = make_splits(normal_images)
tb_train,   tb_val,   tb_test   = make_splits(tb_images)

log(f"  {'':15s}  Train    Val    Test")
log(f"  {'Normal':15s}: {len(norm_train):5d}  {len(norm_val):5d}  {len(norm_test):5d}")
log(f"  {'TB':15s}: {len(tb_train):5d}  {len(tb_val):5d}  {len(tb_test):5d}")
log(f"  {'TOTAL':15s}: {len(norm_train)+len(tb_train):5d}  {len(norm_val)+len(tb_val):5d}  {len(norm_test)+len(tb_test):5d}")


# ── STEP 4: BUILD IMAGEFOLDER ──
log("\n[ STEP 4 ] Building ImageFolder Structure")
log("-" * 50)

splits_map = {
    "train": {"normal": norm_train, "tb": tb_train},
    "val":   {"normal": norm_val,   "tb": tb_val},
    "test":  {"normal": norm_test,  "tb": tb_test},
}

total_copied = 0
for split_name, classes in splits_map.items():
    for class_name, images in classes.items():
        dest_dir = OUTPUT_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(images):
            ext  = src.suffix.lower() or ".png"
            dest = dest_dir / f"{split_name}_{class_name}_{i:05d}{ext}"
            if not dest.exists():
                copy_image(src, dest)
            total_copied += 1
        log(f"  {split_name}/{class_name}: {len(images)} images")

log(f"\n  Total files: {total_copied}")


# ── STEP 5: MANIFEST ──
manifest = {
    "total": total,
    "normal": len(normal_images),
    "tb": len(tb_images),
    "splits": {
        "train": {"normal": len(norm_train), "tb": len(tb_train)},
        "val":   {"normal": len(norm_val),   "tb": len(tb_val)},
        "test":  {"normal": len(norm_test),  "tb": len(tb_test)},
    }
}
with open(BASE_DIR / "dataset_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

log("\n" + "="*60)
log("  DATASET PREPARATION COMPLETE")
log("  Run: python train_densenet.py")
log("="*60 + "\n")
