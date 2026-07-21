import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from PIL import Image

class GuardDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        super().__init__()
        self.is_train = is_train
        
        # Transforms (similar size to densenet)
        img_size = 224
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = self.train_transform if is_train else self.val_transform
        
        self.samples = []
        self.labels = []
        
        # Load Non-CXR (CIFAR-10) -> Label 1
        cifar_dir = os.path.join(data_dir, 'cifar10')
        cifar = datasets.CIFAR10(root=cifar_dir, train=True, download=True)
        # Use a subset of 1500 CIFAR-10 images
        num_cifar = 1500
        for i in range(num_cifar):
            img, _ = cifar[i] # img is PIL
            self.samples.append(img)
            self.labels.append(1)
            
        # Load CXR -> Label 0
        cxr_dir = os.path.join(data_dir, 'train', 'normal')
        cxr_tb_dir = os.path.join(data_dir, 'train', 'tb')
        
        cxr_files = []
        if os.path.exists(cxr_dir):
            cxr_files.extend([os.path.join(cxr_dir, f) for f in os.listdir(cxr_dir) if f.endswith('.png')])
        if os.path.exists(cxr_tb_dir):
            cxr_files.extend([os.path.join(cxr_tb_dir, f) for f in os.listdir(cxr_tb_dir) if f.endswith('.png')])
            
        import random
        random.seed(42)
        random.shuffle(cxr_files)
        cxr_files = cxr_files[:1500] # Use 1500 CXR images to balance
        
        for f in cxr_files:
            img = Image.open(f).convert('RGB')
            self.samples.append(img)
            self.labels.append(0)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(label, dtype=torch.long)


def train_guard():
    print("Initializing MobileNetV2 Guard training...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    full_dataset = GuardDataset(data_dir, is_train=True)
    val_dataset = GuardDataset(data_dir, is_train=False) # Simplified for validation
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, _ = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    _, val_ds = random_split(val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)
    
    # Weight penalty: 3.0x for False Positives (label 1 misclassified as 0 -> weight for class 1 should be higher to penalize missing Non-CXR)
    # Class 0: CXR, Class 1: Non-CXR
    weights = torch.tensor([1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 5
    best_acc = 0.0
    weight_path = os.path.join(models_dir, 'cxr_v2_classifier.pth')
    
    import copy
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
                
            epoch_loss = running_loss / total
            epoch_acc = correct.double() / total
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), weight_path)
                    print(f"Saved new best guard model with Acc: {best_acc:.4f}")

if __name__ == '__main__':
    train_guard()
