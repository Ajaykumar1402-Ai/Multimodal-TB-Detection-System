import os
import json
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing import CXRDataset
from sklearn.metrics import roc_curve, auc

def optimize_threshold():
    print("Optimizing classification threshold...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    val_dataset = CXRDataset(data_dir, split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class TBDetector(nn.Module):
        def __init__(self, dropout_rate=0.4):
            super().__init__()
            try:
                self.backbone = models.densenet121(weights=None)
            except AttributeError:
                self.backbone = models.densenet121(pretrained=False)
                
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
    
    weight_path = os.path.join(models_dir, 'best_tb_densenet121.pth')
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found. Run train_densenet.py first.")
        return
        
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze() # logits
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_logits.extend(outputs.cpu().numpy().tolist())
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            
    with open(os.path.join(models_dir, 'val_logits.json'), 'w') as f:
        json.dump({'logits': all_logits, 'labels': [int(l) for l in all_labels]}, f)
        
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    print(f"Validation AUC: {roc_auc:.4f}")
    
    optimal_idx = 0
    best_threshold = 0.5
    # WHO Targets: Sensitivity (TPR) >= 0.90, Specificity (1 - FPR) >= 0.70
    for i, threshold in enumerate(thresholds):
        sensitivity = tpr[i]
        specificity = 1 - fpr[i]
        
        if sensitivity >= 0.90 and specificity >= 0.70:
            print(f"Found compliant threshold: {threshold:.4f} (Sens: {sensitivity:.4f}, Spec: {specificity:.4f})")
            best_threshold = float(threshold)
            break
    else:
        print("Could not find a threshold that satisfies both Sensitivity >= 90% and Specificity >= 70%.")
        # Find best tradeoff point (Youden's J statistic)
        J = tpr - fpr
        optimal_idx = J.argmax()
        best_threshold = float(thresholds[optimal_idx])
        print(f"Fallback to best threshold: {best_threshold:.4f} (Sens: {tpr[optimal_idx]:.4f}, Spec: {1-fpr[optimal_idx]:.4f})")
        
    config_path = os.path.join(models_dir, 'densenet_config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    config['threshold'] = best_threshold
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Saved optimized threshold {best_threshold:.4f} to config.")

if __name__ == '__main__':
    optimize_threshold()
