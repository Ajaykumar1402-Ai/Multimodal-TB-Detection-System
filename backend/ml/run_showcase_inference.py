"""
Run inference on one TB-positive and one Normal X-ray and print results.
"""
import os, sys, json, base64
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

def load_model(models_dir):
    class TBDetector(nn.Module):
        def __init__(self, dropout_rate=0.4):
            super().__init__()
            self.backbone = models.densenet121(weights=None)
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
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)

def infer(model, image_path, threshold, temperature):
    tensor = preprocess(image_path)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        scaled = logit / temperature
        prob = torch.sigmoid(scaled).item()
    label = "TB POSITIVE" if prob >= threshold else "NORMAL"
    return prob, label

def main():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    config_path = os.path.join(models_dir, 'densenet_config.json')
    config = json.load(open(config_path))
    threshold = config.get('threshold', 0.4035)
    temperature = config.get('temperature', 1.4738)

    model = load_model(models_dir)
    print(f"Model loaded. Threshold={threshold:.4f}, Temperature={temperature:.4f}")
    print()

    tb_image    = r"C:\Users\itsak\OneDrive\tb 1\backend\data\test\tb\Tuberculosis-1.png"
    norm_image  = r"C:\Users\itsak\OneDrive\tb 1\backend\data\test\normal\Normal-1.png"

    for img_path, ground_truth in [(tb_image, "TB POSITIVE (Ground Truth)"), (norm_image, "NORMAL (Ground Truth)")]:
        prob, prediction = infer(model, img_path, threshold, temperature)
        confidence = prob * 100 if "POSITIVE" in prediction else (1 - prob) * 100
        print(f"Image: {os.path.basename(img_path)}")
        print(f"  Ground Truth : {ground_truth}")
        print(f"  TB Probability: {prob*100:.1f}%")
        print(f"  System Result : {prediction} (Confidence: {confidence:.1f}%)")
        match = "CORRECT" if (("TB" in ground_truth) == ("POSITIVE" in prediction)) else "INCORRECT"
        print(f"  Match         : {match}")
        print()

if __name__ == '__main__':
    main()
