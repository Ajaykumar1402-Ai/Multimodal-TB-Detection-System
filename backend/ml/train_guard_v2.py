import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# EMA implementation
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

class GuardDatasetV2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 1. Load CXR (Label 1)
        cxr_dir = os.path.join(root_dir, 'cxr')
        if os.path.exists(cxr_dir):
            for f in os.listdir(cxr_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cxr_dir, f), 1))
                    
        # 2. Load Non-CXR & Hard Negatives (Label 0)
        non_cxr_dir = os.path.join(root_dir, 'non_cxr')
        if os.path.exists(non_cxr_dir):
            for root, _, files in os.walk(non_cxr_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Note: Label 0 = Non-CXR
                        self.samples.append((os.path.join(root, f), 0))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Fallback for corrupted images
            img = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)

def train_guard_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Guard AI V2 on {device}")
    
    # Advanced Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_guard')
    if not os.path.exists(os.path.join(dataset_root, 'cxr')):
        print("Dataset not found. Please run prepare_guard_dataset.py and import data first.")
        return
        
    dataset = GuardDatasetV2(dataset_root, transform=train_transform)
    
    # Class Balancing
    labels = [s[1] for s in dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
    if class_counts[0] == 0 or class_counts[1] == 0:
        print("Error: Missing classes in dataset. Aborting.")
        return
        
    class_weights = [1.0/class_counts[0], 1.0/class_counts[1]]
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0)
    
    # Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(device)
    
    ema = ModelEMA(model)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = GradScaler()
    
    epochs = 20
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cxr_v2_guard_master.pth')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixed Precision
            with autocast():
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema.update()
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            
        avg_loss = total_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")
        
        # Early Stopping & Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save EMA weights
            ema.apply_shadow()
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"  -> Saved new best model (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    train_guard_v2()
