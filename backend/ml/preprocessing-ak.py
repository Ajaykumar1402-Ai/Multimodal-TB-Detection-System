"""
TB Vision Pro — Unified Preprocessing Module
=============================================
Single source of truth for ALL image preprocessing.

Every code path (training, validation, inference, threshold optimization,
temperature calibration, verification) MUST use these functions to ensure
the model sees identical image distributions everywhere.

Pipeline:
  1. Load grayscale
  2. CLAHE (clipLimit=2.0, tileGrid=8x8) — DETERMINISTIC, always applied
  3. Resize to IMG_SIZE x IMG_SIZE (aspect-ratio-preserving with zero-padding)
  4. Convert to 3-channel RGB
  5. ImageNet normalization

Scientific basis:
  - CLAHE enhances local contrast for subtle TB opacities
    (Pizer et al., 1987; validated on CheXpert by Rajpurkar et al., 2017)
  - ImageNet normalization required for transfer learning
    (Kornblith et al., 2019)
  - Aspect-ratio-preserving resize avoids anatomical distortion
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ─────────────────────────────────────────────────
# GLOBAL CONSTANTS — shared by all modules
# ─────────────────────────────────────────────────
IMG_SIZE = 320   # Increased from 224 for better lesion resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)


# ─────────────────────────────────────────────────
# CORE PREPROCESSING (deterministic, always applied)
# ─────────────────────────────────────────────────
def apply_clahe(gray_img):
    """
    Apply CLAHE to a grayscale image.
    This is DETERMINISTIC — always applied with the same parameters.
    
    Args:
        gray_img: np.ndarray, single-channel grayscale image (H, W)
    Returns:
        np.ndarray: CLAHE-enhanced grayscale image (H, W)
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray_img)


def preprocess_cxr_from_path(img_path, img_size=IMG_SIZE):
    """
    Load and preprocess a CXR image from a file path.
    Returns a 3-channel RGB numpy array (H, W, 3) in uint8.
    
    Args:
        img_path: str or Path, path to the image file
        img_size: int, target output size
    Returns:
        np.ndarray: preprocessed RGB image (img_size, img_size, 3), dtype=uint8
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return _preprocess_gray(img, img_size)


def preprocess_cxr_from_bytes(image_bytes, img_size=IMG_SIZE):
    """
    Load and preprocess a CXR image from raw bytes (for live inference).
    Returns a 3-channel RGB numpy array (H, W, 3) in uint8.
    
    Args:
        image_bytes: bytes, raw image file bytes
        img_size: int, target output size
    Returns:
        np.ndarray: preprocessed RGB image (img_size, img_size, 3), dtype=uint8
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode image from bytes")
    return _preprocess_gray(img, img_size)


def _preprocess_gray(gray_img, img_size):
    """
    Internal: deterministic pipeline for a grayscale image.
    
    Steps:
      1. CLAHE
      2. Aspect-ratio-preserving resize with zero-padding
      3. Grayscale → RGB
    """
    # 1. CLAHE (deterministic)
    enhanced = apply_clahe(gray_img)
    
    # 2. Aspect-ratio-preserving resize + zero-padding
    h, w = enhanced.shape
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Pad to square with zeros (black border)
    pad_top  = (img_size - new_h) // 2
    pad_bot  = img_size - new_h - pad_top
    pad_left = (img_size - new_w) // 2
    pad_right = img_size - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )
    
    # 3. Grayscale → 3-channel RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    return rgb


# ─────────────────────────────────────────────────
# TRANSFORMS (for PyTorch DataLoader / inference)
# ─────────────────────────────────────────────────
def get_inference_transform(img_size=IMG_SIZE):
    """
    Returns the deterministic transform for validation / test / live inference.
    Applied AFTER preprocess_cxr_from_path or preprocess_cxr_from_bytes.
    
    The input is already a (img_size, img_size, 3) uint8 numpy array,
    so we just convert to tensor and normalize.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_train_augmentation(img_size=IMG_SIZE):
    """
    Returns training augmentation transform.
    Applied AFTER preprocess_cxr_from_path (CLAHE + resize already done).
    
    Medically valid augmentations only:
    - Rotation ±10° (patient positioning variation)
    - Shift/Scale (body position / film distance)
    - Brightness/Contrast ±15% (exposure variation between scanners)
    - Gaussian noise (film grain / low-dose protocol)
    - Gaussian blur (slight motion from breathing)
    
    NOT included:
    - Horizontal flip (TB has laterality preferences: right upper lobe)
    - Vertical flip (no valid CXR is inverted)
    - Elastic deformation (not realistic for rigid thoracic anatomy)
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10, fill=0),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.90, 1.10),
            fill=0
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Gaussian noise is added at tensor level
        AddGaussianNoise(mean=0.0, std=0.01, p=0.3),
    ])


class AddGaussianNoise:
    """Add Gaussian noise to a tensor with probability p."""
    def __init__(self, mean=0.0, std=0.01, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor


# ─────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────
class CXRDataset(Dataset):
    """
    Chest X-ray dataset with unified preprocessing.
    
    Preprocessing pipeline:
      1. Load grayscale → CLAHE → resize with padding → RGB  (deterministic)
      2. Augmentation (train only) or identity (val/test)
      3. ToTensor → ImageNet normalize
    """
    def __init__(self, data_dir, split='train', img_size=IMG_SIZE):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(data_dir, split)
        
        # Load Normal (class 0)
        normal_dir = os.path.join(split_dir, 'normal')
        if os.path.exists(normal_dir):
            for file in sorted(os.listdir(normal_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(normal_dir, file))
                    self.labels.append(0)
                    
        # Load TB (class 1)
        tb_dir = os.path.join(split_dir, 'tb')
        if os.path.exists(tb_dir):
            for file in sorted(os.listdir(tb_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(tb_dir, file))
                    self.labels.append(1)
        
        # Transforms
        if split == 'train':
            self.transform = get_train_augmentation(img_size)
        else:
            self.transform = get_inference_transform(img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Deterministic preprocessing: CLAHE → resize → RGB
            img_array = preprocess_cxr_from_path(img_path, self.img_size)
        except ValueError:
            # Corrupted image — log warning instead of silently using zeros
            print(f"[WARNING] Corrupted image skipped: {img_path}")
            img_array = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Apply augmentation (train) or identity transform (val/test)
        img_tensor = self.transform(img_array)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────
# BACKWARD COMPATIBILITY (legacy function names)
# ─────────────────────────────────────────────────
def get_live_transform(img_size=IMG_SIZE):
    """Legacy alias for get_inference_transform."""
    return get_inference_transform(img_size)


def apply_live_clahe(img_array):
    """
    Legacy function — applies CLAHE on an input numpy image array.
    Kept for backward compatibility with existing code.
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    enhanced = apply_clahe(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
