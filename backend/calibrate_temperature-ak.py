"""
TB Vision Pro — Temperature Calibration
========================================
Fits temperature scaling on the validation set to calibrate
predicted probabilities.

Uses the UNIFIED preprocessing module to ensure images are processed
identically to training.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
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

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(
            torch.ones(1) * 1.5
        )

    def forward(self, logits):
        return logits / self.T

# ── MODEL ARCHITECTURE ──
class TBDetector(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        try:
            self.backbone = models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
        except Exception:
            self.backbone = models.densenet121(pretrained=True)
        n = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.75),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

def calibrate(model, val_loader, device):
    model.eval()
    logits_list, labels_list = [], []

    # Collect raw logits from validation set
    print("Collecting raw logits from validation set...")
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.to(device)).squeeze()
            logits_list.append(out.cpu())
            labels_list.append(labels)

    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list).float()

    scaler    = TemperatureScaler()
    optimizer = torch.optim.LBFGS(
        [scaler.T], lr=0.01, max_iter=100
    )
    criterion = nn.BCEWithLogitsLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled = scaler(logits_all)
        loss   = criterion(scaled, labels_all)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    T = float(scaler.T.item())
    T = max(0.5, min(5.0, T))   # Clamp to safe range

    print(f"Optimal temperature T = {T:.4f}")
    print(f"T > 1 means model was overconfident")
    print(f"T < 1 means model was underconfident")

    # Save T to config files
    for config_dest in ["./models/model_config.json", "./models/densenet_config.json", "./model_config.json"]:
        if os.path.exists(config_dest):
            with open(config_dest) as f:
                config = json.load(f)
            config["temperature"] = T
            with open(config_dest, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Updated {config_dest} with temperature = {T:.4f}")

    return T

if __name__ == "__main__":
    # Setup Dataloader — using unified preprocessing (same as training)
    val_ds = CXRDataset(DATA_DIR, split='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load Model
    model = TBDetector(dropout=0.4)
    model_file = SAVE_PATH
    if not os.path.exists(model_file) and os.path.exists(ALT_SAVE_PATH):
        model_file = ALT_SAVE_PATH

    print(f"Loading weights from {model_file}...")
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
    model.to(device)

    calibrate(model, val_loader, device)
