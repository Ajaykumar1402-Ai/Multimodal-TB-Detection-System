import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
import numpy as np
import json, os, time

# ── GPU CONFIG ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ── CONFIG ──
DATA_DIR   = "./data"          # data/train/ data/val/ data/test/
SAVE_PATH  = "./models/best_tb_densenet121.pth"
CONFIG_PATH= "./models/model_config.json"
EPOCHS     = 60
BATCH      = 32 if torch.cuda.is_available() else 8
LR         = 1e-4
IMG_SIZE   = 224
os.makedirs("./models", exist_ok=True)

# Windows Safe Workers
NUM_WORKERS = 0 if os.name == "nt" else 4

# ── AUGMENTATION ──
train_tf = A.Compose([
    A.Resize(IMG_SIZE + 32, IMG_SIZE + 32),
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.6),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),
    A.GaussNoise(var_limit=(10,50), p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.CLAHE(                    # Critical for low-contrast CXRs
        clip_limit=4.0,
        tile_grid_size=(8,8),
        p=0.5
    ),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.dataset   = datasets.ImageFolder(root)
        self.transform = transform
        self.classes   = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img.convert("RGB"))
        aug = self.transform(image=img)
        return aug["image"], label

# ── MODEL ──
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

# ── FOCAL LOSS ──
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = a_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()

def main():
    train_ds = CXRDataset(f"{DATA_DIR}/train", train_tf)
    val_ds   = CXRDataset(f"{DATA_DIR}/val",   val_tf)
    test_ds  = CXRDataset(f"{DATA_DIR}/test",  val_tf)

    # ── HANDLE CLASS IMBALANCE ──
    labels = [l for _, l in train_ds.dataset.samples]
    counts = np.bincount(labels)
    print(f"Before: TB-={counts[0]}, TB+={counts[1]}")

    # Give TB+ cases 5x more weight than TB- cases
    weights = np.where(
        np.array(labels) == 1,
        5.0,   # TB+ weight
        1.0    # TB- weight
    )
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights) * 3,  # 3x more batches
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = TBDetector(dropout=0.4).to(device)
    pos_weight = torch.tensor([8.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── OPTIMIZER ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── MIXED PRECISION ──
    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    # ── TRAINING LOOP ──
    best_auc   = 0
    best_sens  = 0
    no_improve = 0
    patience   = 12

    print(f"\nStarting training ({EPOCHS} epochs)...")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            # MixUp augmentation for TB+ cases only
            tb_mask = (labels == 1)
            if tb_mask.sum() > 1 and np.random.random() < 0.4:
                tb_indices  = torch.where(tb_mask)[0]
                perm        = torch.randperm(len(tb_indices))
                idx1        = tb_indices
                idx2        = tb_indices[perm]
                lam         = np.random.beta(0.4, 0.4)
                imgs[idx1]  = lam * imgs[idx1] + (1 - lam) * imgs[idx2]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):
                out  = model(imgs).squeeze()
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available()
                ):
                    out = model(imgs.to(device, non_blocking=True))
                p = torch.sigmoid(out).cpu().numpy().flatten()
                probs.extend(p)
                trues.extend(labels.numpy())

        probs = np.array(probs)
        trues = np.array(trues)
        try:
            auc = roc_auc_score(trues, probs)
        except Exception:
            auc = 0.0
        preds = (probs >= 0.5).astype(int)

        tp   = ((preds==1) & (trues==1)).sum()
        fn   = ((preds==0) & (trues==1)).sum()
        tn   = ((preds==0) & (trues==0)).sum()
        fp   = ((preds==1) & (trues==0)).sum()
        sens = tp/(tp+fn) if (tp+fn) > 0 else 0
        spec = tn/(tn+fp) if (tn+fp) > 0 else 0
        sec  = time.time() - t0

        print(f"Ep {epoch+1:3d}/{EPOCHS} | "
              f"Loss:{train_loss/len(train_loader):.4f} | "
              f"AUC:{auc:.4f} | "
              f"Sens:{sens*100:.1f}% | "
              f"Spec:{spec*100:.1f}% | "
              f"{sec:.0f}s")

        # Save when sensitivity >= 85% AND best AUC
        if sens >= 0.85 and auc > best_auc:
            best_auc  = auc
            best_sens = sens
            torch.save(model.state_dict(), SAVE_PATH)
            torch.save(model.state_dict(), "./models/best_tb_model.pth")
            print(f"  [OK] Saved — AUC={auc:.4f} Sens={sens*100:.1f}%")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\nBest AUC: {best_auc:.4f}")
    print(f"Best Sensitivity: {best_sens*100:.1f}%")

if __name__ == "__main__":
    main()
