import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CXRDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=224):
        """
        Args:
            data_dir (str): Path to data directory containing train, val, test folders.
            split (str): 'train', 'val', or 'test'
            img_size (int): Target image size
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(data_dir, split)
        
        # Load Normal (class 0)
        normal_dir = os.path.join(split_dir, 'normal')
        if os.path.exists(normal_dir):
            for file in os.listdir(normal_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(normal_dir, file))
                    self.labels.append(0)
                    
        # Load TB (class 1)
        tb_dir = os.path.join(split_dir, 'tb')
        if os.path.exists(tb_dir):
            for file in os.listdir(tb_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(tb_dir, file))
                    self.labels.append(1)
                    
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Augmentations for train
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Transforms for val/test
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def apply_clahe(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Handle empty/corrupted image
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
        img = self.clahe.apply(img)
        # Convert grayscale to 3-channel (since densenet expects 3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Apply CLAHE preprocessing
        img_array = self.apply_clahe(img_path)
        
        if self.split == 'train':
            img_tensor = self.train_transform(img_array)
        else:
            img_tensor = self.val_transform(img_array)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

def get_live_transform(img_size=224):
    """Returns the same transform used in val/test to be used during live inference."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def apply_live_clahe(img_array):
    """
    Applies CLAHE on an input numpy image array.
    Expects a BGR or RGB image, converts to Grayscale, applies CLAHE, and back to RGB.
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb
