"""
TB Vision Pro — DenseNet-121 Training (ML Module)
===================================================
This is the ml/ subdirectory version. It imports from the local
preprocessing module and trains with the unified pipeline.

For the full training with 5-fold CV, use the root train_densenet.py.
This version trains a single run for quick iteration.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
from preprocessing import CXRDataset, IMG_SIZE
import copy

# ── UTF-8 on Windows ──
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass


class AsymmetricBCELoss(nn.Module):
    """
    Asymmetric loss: penalizes false negatives more heavily.
    pos_weight=5.0 means missing a TB case costs 5x more than a false alarm.
    gamma_neg=2.0 applies focal modulation to easy negatives.
    """
    def __init__(self, pos_weight=5.0, gamma_neg=2.0, gamma_pos=0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        
    def forward(self, logits, targets):
        targets = targets.unsqueeze(1).float() if targets.dim() == 1 else targets.float()
        logits = logits.view_as(targets)
        prob = torch.sigmoid(logits)
        
        pos_loss = -targets * torch.log(prob + 1e-8)
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1 - prob) ** self.gamma_pos)
        
        neg_loss = -(1 - targets) * torch.log(1 - prob + 1e-8)
        neg_loss = neg_loss * (prob ** self.gamma_neg)
        
        loss = self.pos_weight * pos_loss + neg_loss
        return loss.mean()


def get_weighted_sampler(dataset):
    class_counts = [0, 0]
    for label in dataset.labels:
        class_counts[label] += 1
    
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[torch.tensor(dataset.labels)]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def train_model():
    print("Initializing DenseNet-121 training (unified preprocessing)...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Use unified CXRDataset (CLAHE + aspect-ratio resize + no horizontal flip)
    train_dataset = CXRDataset(data_dir, split='train', img_size=IMG_SIZE)
    val_dataset = CXRDataset(data_dir, split='val', img_size=IMG_SIZE)
    
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Run collect_data.py first.")
        return
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
        
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use lighter V2 architecture
    class TBDetectorV2(nn.Module):
        def __init__(self, dropout_rate=0.4):
            super().__init__()
            try:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            except AttributeError:
                self.backbone = models.densenet121(pretrained=True)
                
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
            # SE attention
            self.se = nn.Sequential(
                nn.Linear(in_features, in_features // 16),
                nn.ReLU(inplace=True),
                nn.Linear(in_features // 16, in_features),
                nn.Sigmoid()
            )
            
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 1)
            )

        def forward(self, x):
            features = self.backbone(x)
            se_weights = self.se(features)
            features = features * se_weights
            return self.classifier(self.dropout(features))

    model = TBDetectorV2()
    model = model.to(device)
    
    # Asymmetric loss: penalize FN 2x more than FP to reduce false positives
    criterion = AsymmetricBCELoss(pos_weight=2.0, gamma_neg=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    num_epochs = 30
    best_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    from sklearn.metrics import roc_auc_score
    
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
            all_preds = []
            all_labels = []
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_auc = roc_auc_score(all_labels, all_preds)
            
            print(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}')
            
            if phase == 'val':
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    weight_path = os.path.join(models_dir, 'best_tb_densenet121.pth')
                    torch.save(model.state_dict(), weight_path)
                    print(f"Saved new best model with AUC: {best_auc:.4f}")
                    
    print(f'Training complete. Best Val AUC: {best_auc:.4f}')

if __name__ == '__main__':
    train_model()
