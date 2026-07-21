import os
import sys
import copy
import random
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))
from preprocessing import CXRDataset, IMG_SIZE, preprocess_cxr_from_path, get_train_augmentation, get_inference_transform
from utils.diagnostics import DiagnosticsManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- STAGE 1 HYPERPARAMETERS ---
DATA_DIR   = "./data"
N_FOLDS    = 3     # Stage 1 Screening
EPOCHS     = 20    # Stage 1 Screening
BATCH      = 16 if torch.cuda.is_available() else 8
LR         = 3e-4
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 0 if os.name == "nt" else 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR = "./models/exp1_loss_comparison"
os.makedirs(EXP_DIR, exist_ok=True)

# ── MODEL ARCHITECTURE (Identical to Phase 1) ──
class TBDetectorV2(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        try: self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception: self.backbone = models.densenet121(pretrained=True)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.se = nn.Sequential(nn.Linear(n_features, n_features // 16), nn.ReLU(inplace=True), nn.Linear(n_features // 16, n_features), nn.Sigmoid())
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.classifier = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Linear(256, 1))
    def forward(self, x):
        features = self.backbone(x)
        features = features * self.se(features)
        if self.training: return sum([self.classifier(drop(features)) for drop in self.dropouts]) / len(self.dropouts)
        else: return self.classifier(self.dropouts[0](features))

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data
    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup = {}

# ── LOSS FUNCTIONS ──
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float().view_as(logits))
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

class AsymmetricBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0, gamma_neg=2.0, gamma_pos=0.0):
        super().__init__()
        self.pos_weight = pos_weight; self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos
    def forward(self, logits, targets):
        targets = targets.float().view_as(logits)
        prob = torch.sigmoid(logits)
        pos_loss = -targets * torch.log(prob + 1e-7) * ((1.0 - prob + 1e-7) ** self.gamma_pos)
        neg_loss = -(1.0 - targets) * torch.log(1.0 - prob + 1e-7) * ((prob + 1e-7) ** self.gamma_neg)
        return (self.pos_weight * pos_loss + neg_loss).mean()

class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, split, img_size):
        self.images = images; self.labels = labels; self.img_size = img_size
        self.transform = get_train_augmentation(img_size) if split == 'train' else get_inference_transform(img_size)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        try: arr = preprocess_cxr_from_path(self.images[idx], self.img_size)
        except: arr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        return self.transform(arr), torch.tensor(self.labels[idx], dtype=torch.long), self.images[idx]

def evaluate(model, loader, device):
    model.eval()
    probs, trues, paths = [], [], []
    with torch.no_grad():
        for imgs, labels, p in loader:
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(imgs.to(device, non_blocking=True))
            probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
            trues.extend(labels.numpy())
            paths.extend(p)
    return np.array(probs), np.array(trues), paths

def train_model(loss_name, criterion, train_ds, val_ds, diag):
    train_labels = [train_ds.labels[i] for i in range(len(train_ds))]
    counts = np.bincount(train_labels)
    sample_w = [1.0/counts[l] for l in train_labels]
    
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=WeightedRandomSampler(sample_w, len(sample_w)), num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False, num_workers=NUM_WORKERS)
    
    model = TBDetectorV2(dropout=0.4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    ema = EMA(model)
    
    best_auc, best_state, no_improve = 0.0, None, 0
    history = {'train_loss':[], 'val_loss':[], 'val_auc':[], 'lr':[], 'val_sens':[], 'val_spec':[], 'val_bal_acc':[]}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, labels, _ in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(imgs.to(device)).squeeze()
                loss = criterion(out, labels.float().to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        ema.apply_shadow()
        probs, trues, _ = evaluate(model, val_loader, device)
        ema.restore()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    out = model(imgs.to(device)).squeeze()
                    val_loss += criterion(out, labels.float().to(device)).item()
        avg_val_loss = val_loss / len(val_loader)
        
        metrics = diag.evaluate_metrics(trues, probs)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(metrics["ROC-AUC"])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['val_sens'].append(metrics["Sensitivity (Recall)"])
        history['val_spec'].append(metrics["Specificity"])
        history['val_bal_acc'].append(metrics["Balanced Accuracy"])
        
        logger.info(f"{loss_name} Ep {epoch+1:2d}/{EPOCHS} | T:{avg_train_loss:.3f} V:{avg_val_loss:.3f} | AUC:{metrics['ROC-AUC']:.3f} Sens:{metrics['Sensitivity (Recall)']:.2f} Spec:{metrics['Specificity']:.2f}")
        
        if metrics['ROC-AUC'] > best_auc:
            best_auc = metrics['ROC-AUC']
            ema.apply_shadow()
            best_state = copy.deepcopy(model.state_dict())
            ema.restore()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 8: break
            
    model.load_state_dict(best_state)
    probs, trues, paths = evaluate(model, val_loader, device)
    metrics = diag.evaluate_metrics(trues, probs)
    ece, brier = diag.plot_calibration_curve(loss_name, trues, probs)
    metrics['ECE'] = ece
    metrics['Brier Score'] = brier
    
    diag.save_predictions_csv(loss_name, paths, trues, probs)
    diag.plot_roc_curve(loss_name, trues, probs)
    diag.plot_pr_curve(loss_name, trues, probs)
    diag.plot_confusion_matrix(loss_name, trues, probs)
    diag.plot_probability_histogram(loss_name, trues, probs)
    diag.plot_learning_curves(loss_name, history)
    
    return metrics

def main():
    logger.info("Starting Experiment 1: Loss Function Comparison (Stage 1)")
    train_ds = CXRDataset(DATA_DIR, split='train', img_size=IMG_SIZE)
    val_ds = CXRDataset(DATA_DIR, split='val', img_size=IMG_SIZE)
    all_images = train_ds.images + val_ds.images
    all_labels = train_ds.labels + val_ds.labels
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    diag = DiagnosticsManager(EXP_DIR)
    
    losses = {
        "BCE": nn.BCEWithLogitsLoss(),
        "Focal": FocalLoss(),
        "AsymBCE": AsymmetricBCELoss(pos_weight=5.0)
    }
    
    results = {k: [] for k in losses.keys()}
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        logger.info(f"\n--- FOLD {fold_idx+1} ---")
        fold_train = FoldDataset([all_images[i] for i in train_idx], [all_labels[i] for i in train_idx], 'train', IMG_SIZE)
        fold_val = FoldDataset([all_images[i] for i in val_idx], [all_labels[i] for i in val_idx], 'val', IMG_SIZE)
        
        for name, criterion in losses.items():
            logger.info(f"Training {name} Loss...")
            metrics = train_model(f"{name}_fold{fold_idx+1}", criterion, fold_train, fold_val, diag)
            results[name].append(metrics)
            
    summary = []
    for name, fold_metrics in results.items():
        mean_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
        mean_metrics["Loss Function"] = name
        summary.append(mean_metrics)
        
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(EXP_DIR, "loss_comparison_summary.csv"), index=False)
    logger.info("\n=== EXPERIMENT 1 COMPLETE ===")
    print(df.to_string())

if __name__ == "__main__":
    main()
