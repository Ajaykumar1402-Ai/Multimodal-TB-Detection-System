import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
from preprocessing import CXRDataset
import copy

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, 1], targets: [B]
        targets_f = targets.unsqueeze(1).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_f, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

def get_weighted_sampler(dataset):
    class_counts = [0, 0]
    for label in dataset.labels:
        class_counts[label] += 1
    
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def train_model():
    print("Initializing DenseNet-121 training...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    train_dataset = CXRDataset(data_dir, split='train')
    val_dataset = CXRDataset(data_dir, split='val')
    
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Run collect_data.py first.")
        return
        
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    class TBDetector(nn.Module):
        def __init__(self, dropout_rate=0.4):
            super().__init__()
            try:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            except AttributeError:
                self.backbone = models.densenet121(pretrained=True)
                
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout_rate * 0.75),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.backbone(x)

    model = TBDetector()
    
    # Load previous weights if available
    weight_path = os.path.join(models_dir, 'best_tb_densenet121.pth')
    if os.path.exists(weight_path):
        print("Loading existing weights to fine-tune...")
        try:
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Could not load existing weights: {e}")
            
    model = model.to(device)
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    num_epochs = 15
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
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_auc = roc_auc_score(all_labels, all_preds)
            
            print(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}')
            
            if phase == 'val':
                scheduler.step(epoch_auc)
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), weight_path)
                    print(f"Saved new best model with AUC: {best_auc:.4f}")
                    
    print(f'Training complete. Best Val AUC: {best_auc:.4f}')

if __name__ == '__main__':
    train_model()
