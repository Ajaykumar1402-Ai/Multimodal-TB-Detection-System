"""
TB Vision Pro — DenseNet-121 Master Training Pipeline (Phase 1 Baseline)
======================================================================
"""

import os
import sys
import json
import time
import copy
import random
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Disable warnings for cleaner logs
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))
from preprocessing import CXRDataset, IMG_SIZE, preprocess_cxr_from_path, get_train_augmentation, get_inference_transform
from utils.diagnostics import DiagnosticsManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── SEED FIXING ──
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ── CONFIG ──
DATA_DIR   = "./data"
SAVE_PATH  = "./models/best_tb_densenet121_phase1.pth"
CONFIG_PATH= "./models/model_config.json"
N_FOLDS    = 5
EPOCHS     = 40
BATCH      = 16 if torch.cuda.is_available() else 8
LR         = 3e-4
WEIGHT_DECAY = 1e-3
POS_WEIGHT = 2.0
os.makedirs("./models/training_reports", exist_ok=True)
NUM_WORKERS = 0 if os.name == "nt" else 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MODEL ARCHITECTURE ──
class TBDetectorV2(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        try:
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            self.backbone = models.densenet121(pretrained=True)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.se = nn.Sequential(
            nn.Linear(n_features, n_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(n_features // 16, n_features),
            nn.Sigmoid()
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        features = self.backbone(x)
        se_weights = self.se(features)
        features = features * se_weights
        if self.training:
            outputs = [self.classifier(drop(features)) for drop in self.dropouts]
            return sum(outputs) / len(outputs)
        else:
            return self.classifier(self.dropouts[0](features))

# ── ASYMMETRIC LOSS ──
class AsymmetricBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0, gamma_neg=2.0, gamma_pos=0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        
    def forward(self, logits, targets):
        targets = targets.float().view_as(logits)
        prob = torch.sigmoid(logits)
        pos_loss = -targets * torch.log(prob + 1e-7)
        if self.gamma_pos > 0: pos_loss = pos_loss * ((1.0 - prob + 1e-7) ** self.gamma_pos)
        neg_loss = -(1.0 - targets) * torch.log(1.0 - prob + 1e-7)
        neg_loss = neg_loss * ((prob + 1e-7) ** self.gamma_neg)
        loss = self.pos_weight * pos_loss + neg_loss
        return loss.mean()

# ── EMA ──
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])  # FIXED: use copy_
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])  # FIXED: use copy_
        self.backup = {}

# ── SUBSET DATASET ──
class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, split, img_size):
        self.images = images
        self.labels = labels
        self.img_size = img_size
        self.split = split
        if split == 'train': self.transform = get_train_augmentation(img_size)
        else: self.transform = get_inference_transform(img_size)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path, label = self.images[idx], self.labels[idx]
        try: img_array = preprocess_cxr_from_path(img_path, self.img_size)
        except ValueError: img_array = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img_tensor = self.transform(img_array)
        return img_tensor, torch.tensor(label, dtype=torch.long), img_path

def evaluate(model, loader, device):
    model.eval()
    probs, trues, paths = [], [], []
    with torch.no_grad():
        for imgs, labels, p in loader:
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()): # FIXED: New AMP API
                out = model(imgs.to(device, non_blocking=True))
            p_out = torch.sigmoid(out).cpu().numpy().flatten()
            probs.extend(p_out)
            trues.extend(labels.numpy())
            paths.extend(p)
    return np.array(probs), np.array(trues), paths

def train_one_fold(fold_idx, train_ds, val_ds, n_folds, diag):
    logger.info(f"\n{'='*60}\n  FOLD {fold_idx + 1} / {n_folds}\n{'='*60}")
    
    train_labels = [train_ds.labels[i] for i in range(len(train_ds))]
    counts = np.bincount(train_labels)
    weights = 1.0 / counts
    sample_w = [weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_w, len(sample_w))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = TBDetectorV2(dropout=0.4).to(device)
    criterion = AsymmetricBCELoss(pos_weight=POS_WEIGHT, gamma_neg=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()) # FIXED: New API
    ema = EMA(model, decay=0.999)
    
    best_auc = 0.0
    best_state = None
    no_improve, patience = 0, 12
    
    history = {'train_loss':[], 'val_loss':[], 'val_auc':[], 'lr':[], 'val_sens':[], 'val_spec':[], 'val_f1':[]}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(imgs).squeeze()
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation using EMA
        ema.apply_shadow()
        probs, trues, val_paths = evaluate(model, val_loader, device)
        ema.restore()
        
        # Validation Loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    out = model(imgs.to(device)).squeeze()
                    loss = criterion(out, labels.float().to(device))
                    val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        try: auc = roc_auc_score(trues, probs)
        except Exception: auc = 0.0
        
        preds = (probs >= 0.5).astype(int) # FIXED: Use 0.5 for stable metric reporting
        tp = ((preds == 1) & (trues == 1)).sum()
        fn = ((preds == 0) & (trues == 1)).sum()
        tn = ((preds == 0) & (trues == 0)).sum()
        fp = ((preds == 1) & (trues == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(trues, preds, zero_division=0)
        
        curr_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(auc)
        history['lr'].append(curr_lr)
        history['val_sens'].append(sens)
        history['val_spec'].append(spec)
        history['val_f1'].append(f1)
        
        logger.info(f"Ep {epoch+1:2d}/{EPOCHS} | TLoss:{avg_train_loss:.4f} | VLoss:{avg_val_loss:.4f} | AUC:{auc:.4f} | Sens:{sens*100:.1f}% | Spec:{spec*100:.1f}%")
        
        # FIXED: Save model directly on highest AUC, removing arbitrary 0.85 floor
        if auc > best_auc:
            best_auc = auc
            ema.apply_shadow()
            best_state = copy.deepcopy(model.state_dict())
            ema.restore()
            logger.info(f"  -> New best AUC: {best_auc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break
                
    # Run diagnostics for this fold
    logger.info("Running post-fold diagnostics...")
    model.load_state_dict(best_state)
    model.eval()
    probs, trues, val_paths = evaluate(model, val_loader, device)
    
    diag.save_predictions_csv(fold_idx, val_paths, trues, probs, threshold=0.5)
    diag.plot_roc_curve(fold_idx, trues, probs)
    diag.plot_pr_curve(fold_idx, trues, probs)
    diag.plot_confusion_matrix(fold_idx, trues, probs, threshold=0.5)
    diag.plot_probability_histogram(fold_idx, trues, probs)
    diag.plot_calibration_curve(fold_idx, trues, probs)
    diag.plot_learning_curves(fold_idx, history)
    
    return {'best_auc': best_auc, 'best_state': best_state}

def main():
    logger.info("Starting TB Vision Pro Master Pipeline (Phase 1)")
    
    train_ds = CXRDataset(DATA_DIR, split='train', img_size=IMG_SIZE)
    val_ds = CXRDataset(DATA_DIR, split='val', img_size=IMG_SIZE)
    
    all_images = train_ds.images + val_ds.images
    all_labels = train_ds.labels + val_ds.labels
    
    logger.info(f"Total images for CV: {len(all_images)} (Normal: {all_labels.count(0)}, TB: {all_labels.count(1)})")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    diag = DiagnosticsManager()
    
    fold_results = []
    best_global_auc = 0
    best_global_state = None
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_images, all_labels)):
        fold_train_ds = FoldDataset([all_images[i] for i in train_indices], [all_labels[i] for i in train_indices], 'train', IMG_SIZE)
        fold_val_ds = FoldDataset([all_images[i] for i in val_indices], [all_labels[i] for i in val_indices], 'val', IMG_SIZE)
        
        res = train_one_fold(fold_idx, fold_train_ds, fold_val_ds, N_FOLDS, diag)
        fold_results.append(res)
        
        if res['best_auc'] > best_global_auc:
            best_global_auc = res['best_auc']
            best_global_state = res['best_state']
            
    logger.info("CROSS-VALIDATION COMPLETE")
    aucs = [r['best_auc'] for r in fold_results]
    for i, a in enumerate(aucs):
        logger.info(f"Fold {i+1}: AUC={a:.4f}")
    logger.info(f"Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    
    if best_global_state:
        torch.save(best_global_state, SAVE_PATH)
        torch.save(best_global_state, "./models/best_tb_model.pth")
        logger.info(f"Saved best model with AUC {best_global_auc:.4f} to {SAVE_PATH}")
        
if __name__ == "__main__":
    main()
