"""
TB Vision Pro — Final Model Verification
==========================================
Runs MC Dropout inference on the test set and checks all metrics
against WHO End TB Strategy benchmarks.

Uses the UNIFIED preprocessing module to ensure images are processed
identically to training.
"""

import os
import sys
import json
import torch
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

# Reconfigure stdout/stderr encoding to prevent Windows UnicodeEncodeError on emojis
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

# ── MODEL ARCHITECTURE ──
class TBDetector(torch.nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        try:
            self.backbone = models.densenet121(weights=None)
        except AttributeError:
            self.backbone = models.densenet121(pretrained=False)
        n = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(n, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout * 0.75),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

def predict_with_uncertainty(model, img_tensor, n_passes=30, temperature=1.0):
    """
    MC Dropout inference with proper stochastic dropout.
    
    NOTE: No fixed seeds are set here. MC Dropout requires stochastic
    dropout masks across the N passes to estimate epistemic uncertainty.
    Fixed seeds would make all passes identical, defeating the purpose.
    """
    # Keep model in eval mode for BatchNorm stability, but enable only Dropout layers
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    dev = next(model.parameters()).device
    img_tensor = img_tensor.to(dev)
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(img_tensor).squeeze()
            scaled_logit = output / temperature
            prob = torch.sigmoid(scaled_logit).item()
            predictions.append(prob)
            
    mean_prob = float(np.mean(predictions))
    return {
        "probability": mean_prob * 100
    }

if __name__ == "__main__":
    # Setup test dataloader — using unified preprocessing (same as training)
    config_path = "./models/model_config.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        
    img_size = config.get("img_size", IMG_SIZE)
    
    test_dir = './data/test'
    if not os.path.exists(test_dir):
        print(f"[FAIL] Test dataset directory not found at {test_dir}")
        sys.exit(1)
    
    # Use unified CXRDataset for consistent preprocessing
    test_dataset = CXRDataset('./data', split='test', img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = TBDetector(dropout=0.4)
    
    model_file = "./models/best_tb_model.pth"
    if not os.path.exists(model_file) and os.path.exists("./models/best_tb_densenet121.pth"):
        model_file = "./models/best_tb_densenet121.pth"
        
    model.load_state_dict(
        torch.load(model_file, map_location=device, weights_only=False)
    )
    model.to(device)
    
    threshold = config.get("threshold", 0.35)
    T         = config.get("temperature", 1.0)
    
    print("=" * 60)
    print("FINAL MODEL VERIFICATION")
    print("WHO End TB Strategy Clinical Benchmark")
    print("=" * 60)
    
    # Run MC Dropout inference on test set
    all_probs, all_labels = [], []
    for imgs, labels in test_loader:
        result = predict_with_uncertainty(
            model, imgs, n_passes=30, temperature=T
        )
        all_probs.append(result["probability"] / 100)
        all_labels.extend(labels.numpy())
    
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)
    cm     = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100
    ppv  = tp / (tp + fp) * 100 if (tp+fp) > 0 else 0
    npv  = tn / (tn + fn) * 100 if (tn+fn) > 0 else 0
    acc  = (tp + tn) / len(labels) * 100
    auc  = roc_auc_score(labels, probs) * 100
    
    print(f"\n{'Metric':<30} {'Result':>8} {'Target':>8} {'Pass':>6}")
    print("-" * 56)
    
    def check(label, val, target):
        ok = val >= target
        print(f"{label:<30} {val:>7.1f}% {'>= '+str(target)+'%':>8} "
              f"{'PASS' if ok else 'FAIL':>6}")
        return ok
    
    r1 = check("Sensitivity",  sens, 90)
    r2 = check("Specificity",  spec, 70)
    r3 = check("PPV",          ppv,  75)
    r4 = check("NPV",          npv,  95)
    r5 = check("Accuracy",     acc,  80)
    r6 = check("AUC-ROC",      auc,  92)
    
    print()
    all_pass = all([r1, r2, r3, r4, r5, r6])
    
    if all_pass:
        print("ALL CHECKS PASSED - WHO COMPLIANT")
        print("   Safe to enable production inference")
        print("   Set INFERENCE_ENABLED=true on Vercel")
        sys.exit(0)
    else:
        print("CHECKS FAILED - DO NOT GO LIVE")
        if not r1:
            print(f"   Sensitivity {sens:.1f}% < 90%")
            print("   -> Lower threshold by 0.05 and retry")
            print("   -> Or increase pos_weight to 5.0 and retrain")
        if not r2:
            print(f"   Specificity {spec:.1f}% < 70%")
            print("   -> Raise threshold by 0.05 and retry")
        print()
        print("   DO NOT enable production until all pass")
        sys.exit(1)
