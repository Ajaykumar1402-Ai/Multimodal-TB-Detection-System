"""
Find a normal X-ray that is correctly classified as Normal, and a TB that is correctly classified as TB.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import glob

def load_model(models_dir):
    class TBDetector(nn.Module):
        def __init__(self, dropout_rate=0.4):
            super().__init__()
            self.backbone = models.densenet121(weights=None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, 512), nn.ReLU(),
                nn.BatchNorm1d(512), nn.Dropout(p=dropout_rate * 0.75), nn.Linear(512, 128),
                nn.ReLU(), nn.Dropout(p=dropout_rate * 0.5), nn.Linear(128, 1)
            )
        def forward(self, x): return self.backbone(x)
    model = TBDetector()
    wp = os.path.join(models_dir, 'best_tb_densenet121.pth')
    ck = torch.load(wp, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck)
    model.eval()
    return model

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); img = clahe.apply(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return tf(Image.fromarray(img_rgb)).unsqueeze(0)

def infer(model, path, threshold, temperature):
    with torch.no_grad():
        logit = model(preprocess(path)).squeeze()
        prob = torch.sigmoid(logit / temperature).item()
    return prob, ("TB POSITIVE" if prob >= threshold else "NORMAL")

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
cfg = json.load(open(os.path.join(models_dir, 'densenet_config.json')))
threshold, temperature = cfg.get('threshold', 0.4035), cfg.get('temperature', 1.4738)
model = load_model(models_dir)

base = r'C:\Users\itsak\OneDrive\tb 1\backend\data\test'
good_normal = None
good_tb     = None

for f in sorted(glob.glob(os.path.join(base, 'normal', '*.png'))):
    prob, pred = infer(model, f, threshold, temperature)
    if pred == 'NORMAL':
        good_normal = (f, prob)
        print(f"GOOD NORMAL: {os.path.basename(f)} -> prob={prob*100:.1f}%")
        break

for f in sorted(glob.glob(os.path.join(base, 'tb', '*.png'))):
    prob, pred = infer(model, f, threshold, temperature)
    if pred == 'TB POSITIVE':
        good_tb = (f, prob)
        print(f"GOOD TB: {os.path.basename(f)} -> prob={prob*100:.1f}%")
        break
