import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np

# Force UTF-8 on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, 'cxr_v2_classifier.pth')

class CombinedDataset(Dataset):
    def __init__(self, cxr_dir, cifar_dataset, transform=None, num_cifar_samples=1000):
        self.transform = transform
        self.samples = []
        
        # 1. Load CXR images (Label 1)
        cxr_classes = ['normal', 'tb']
        cxr_count = 0
        for cls in cxr_classes:
            cls_dir = os.path.join(cxr_dir, cls)
            if os.path.exists(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        fpath = os.path.join(cls_dir, fname)
                        self.samples.append((fpath, 1)) # 1 = CXR
                        cxr_count += 1
        print(f"Loaded {cxr_count} CXR images.")
        
        # 2. Add sub-sampled CIFAR-10 images (Label 0)
        cifar_indices = np.random.choice(len(cifar_dataset), min(num_cifar_samples, len(cifar_dataset)), replace=False)
        for idx in cifar_indices:
            img, _ = cifar_dataset[idx]
            self.samples.append((img, 0)) # 0 = Non-CXR
        print(f"Loaded {min(num_cifar_samples, len(cifar_dataset))} CIFAR-10 non-CXR images.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item, label = self.samples[idx]
        if isinstance(item, str):
            # Load CXR from file
            img = Image.open(item).convert('RGB')
        else:
            # CIFAR image is already PIL Image
            img = item.convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)

def train_classifier():
    print(f"Training CXR vs Non-CXR MobileNetV2 classifier on: {DEVICE}")
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset for negative samples
    print("Downloading/Loading CIFAR-10 for non-CXR samples...")
    try:
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True)
    except Exception as e:
        print(f"CIFAR-10 download failed: {e}. Downloading test split or falling back to synthetic CIFAR...")
        cifar_train = datasets.CIFAR10(root='./data', train=False, download=True)
        
    # Combined dataset
    cxr_dir = './data/train'
    if not os.path.exists(cxr_dir):
        print(f"Error: CXR train directory {cxr_dir} not found.")
        sys.exit(1)
        
    dataset = CombinedDataset(cxr_dir, cifar_train, transform=train_transforms, num_cifar_samples=800)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    # Load Pretrained MobileNetV2
    print("Loading MobileNetV2 backbone...")
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=True)
        
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(DEVICE)
    
    # Freeze backbone features (fine-tune classifier first)
    for param in model.features.parameters():
        param.requires_grad = False
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    epochs = 3
    print("Training starts...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Unfreeze all layers for second epoch onwards to allow full adaptation
        if epoch == 1:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Accuracy: {acc:.2f}%")
        
    # Save model weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"CXR vs Non-CXR Classifier model saved to {SAVE_PATH}")

if __name__ == '__main__':
    train_classifier()
