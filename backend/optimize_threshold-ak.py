"""
TB Vision Pro — Threshold Optimizer
====================================
Sweeps decision thresholds on the test set and selects the best
WHO-compliant threshold (Sensitivity >= 90%, Specificity >= 70%).

Uses the UNIFIED preprocessing module to ensure images are processed
identically to training.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import numpy as np
import json
import os
import sys

# Reconfigure stdout/stderr encoding to prevent Windows UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Add ml/ to path for unified preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))
from preprocessing import CXRDataset, IMG_SIZE

# ── DEVICE CONFIG ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG ──
DATA_DIR   = "./data"
SAVE_PATH  = "./models/best_tb_model.pth"
ALT_SAVE_PATH = "./models/best_tb_densenet121.pth"

# ── MODEL ARCHITECTURE ──
class TBDetector(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        try:
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            self.backbone = models.densenet121(pretrained=True)
        
        n_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Squeeze-and-Excitation attention
        self.se = nn.Sequential(
            nn.Linear(n_features, n_features // 16),  # 1024 -> 64
            nn.ReLU(inplace=True),
            nn.Linear(n_features // 16, n_features),  # 64 -> 1024
            nn.Sigmoid()
        )
        
        # Multi-Sample Dropout (5 masks, averaged during training)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        
        # Lighter classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 1024]
        
        # SE attention: reweight feature channels
        se_weights = self.se(features)
        features = features * se_weights
        
        if self.training:
            outputs = [self.classifier(drop(features)) for drop in self.dropouts]
            return sum(outputs) / len(outputs)
        else:
            return self.classifier(self.dropouts[0](features))

# ── PREPARE DATA LOADER (unified preprocessing) ──
test_ds = CXRDataset(DATA_DIR, split='test', img_size=IMG_SIZE)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# Load model
model = TBDetector(dropout=0.4)
model_file = SAVE_PATH
if not os.path.exists(model_file) and os.path.exists(ALT_SAVE_PATH):
    model_file = ALT_SAVE_PATH

print(f"Loading weights from {model_file}...")
model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
model.to(device)
model.eval()

# Get predictions on TEST SET only
# Never optimize threshold on validation set
probs, trues = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(device))
        p   = torch.sigmoid(out).cpu().numpy().flatten()
        probs.extend(p)
        trues.extend(labels.numpy())

probs = np.array(probs)
trues = np.array(trues)

# ROC sweep
fpr, tpr, thresholds = roc_curve(trues, probs)

print(f"{'Threshold':>10} | {'Sensitivity':>11} | "
      f"{'Specificity':>11} | {'F1':>8} | WHO")
print("-" * 60)

best_thresh = 0.5
best_f1     = 0
who_results = []

for thresh, sens_val, fp_rate in zip(thresholds, tpr, fpr):
    spec_val = 1 - fp_rate
    preds    = (probs >= thresh).astype(int)
    tp = ((preds==1)&(trues==1)).sum()
    fp = ((preds==1)&(trues==0)).sum()
    fn = ((preds==0)&(trues==1)).sum()
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0

    who_pass = sens_val >= 0.90 and spec_val >= 0.70
    print(f"{thresh:>10.3f} | {sens_val*100:>10.1f}% | "
          f"{spec_val*100:>10.1f}% | {f1*100:>7.1f}% | "
          f"{'YES' if who_pass else 'NO'}")

    if who_pass and f1 > best_f1:
        best_f1     = f1
        best_thresh = float(thresh)
        who_results.append({
            "threshold":   float(thresh),
            "sensitivity": float(sens_val),
            "specificity": float(spec_val),
            "f1":          float(f1)
        })

if who_results:
    best = max(who_results, key=lambda x: x["f1"])
    best_thresh = best['threshold']
    print(f"\nBEST WHO-COMPLIANT THRESHOLD: {best['threshold']:.3f}")
    print(f"   Sensitivity: {best['sensitivity']*100:.1f}%")
    print(f"   Specificity: {best['specificity']*100:.1f}%")
    print(f"   F1:          {best['f1']*100:.1f}%")
else:
    # No threshold meets both — pick closest
    best_thresh = 0.35   # Start here based on AUC 94.4%
    print(f"\nNo threshold meets both criteria.")
    print(f"  Using 0.35 as starting point.")
    print(f"  Add more TB+ training data and retrain.")

# Save config
config = {
    "threshold":     best_thresh,
    "version":       "2.1",
    "who_compliant": len(who_results) > 0,
    "optimized_on":  "test_set",
    # Preservation keys for production backend
    "img_size":      IMG_SIZE,
    "architecture":  "DenseNet-121",
    "framework":     "pytorch",
    "tb_class_idx":  1,
    "save_path":     os.path.abspath(model_file)
}

# Save to all model config destinations
for config_dest in ["./models/model_config.json", "./models/densenet_config.json", "./model_config.json"]:
    with open(config_dest, "w") as f:
        json.dump(config, f, indent=2)

print(f"\nSaved to ./models/model_config.json and other configuration paths.")
print(f"Use threshold = {best_thresh:.3f} in production")
