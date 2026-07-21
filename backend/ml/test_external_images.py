import os, sys, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Import V2 Guard AI
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
from validation_engine import ValidationEngine
from guard_explainability import analyze_guard_confidence

def load_guard_model(models_dir):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    weight_path = os.path.join(models_dir, 'cxr_v2_guard_master.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Load DenseNet Model
def load_densenet_model(models_dir):
    class TBDetector(nn.Module):
        def __init__(self, dropout_rate=0.4):
            super().__init__()
            self.backbone = models.densenet121(weights=None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.se = nn.Sequential(
                nn.Linear(in_features, in_features // 16),
                nn.ReLU(inplace=True),
                nn.Linear(in_features // 16, in_features),
                nn.Sigmoid()
            )
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 1)
            )
        def forward(self, x):
            features = self.backbone(x)
            se_weights = self.se(features)
            features = features * se_weights
            return self.classifier(self.dropout(features))

    model = TBDetector()
    weight_path = os.path.join(models_dir, 'best_tb_densenet121.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, device

def infer_tb(model_tuple, image_path, threshold, temperature):
    model, device = model_tuple
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, "INVALID"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        scaled = logit / temperature
        prob = torch.sigmoid(scaled).item()
    label = "TB POSITIVE" if prob >= threshold else "NORMAL"
    return prob, label

def main():
    import urllib.request
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    config_path = os.path.join(models_dir, 'densenet_config.json')
    config = json.load(open(config_path))
    threshold = config.get('threshold', 0.3500)
    temperature = config.get('temperature', 1.4738)

    guard_model_tuple = load_guard_model(models_dir)
    tb_model_tuple = load_densenet_model(models_dir)
    validation_engine = ValidationEngine()
    print("V2 Pipeline Models loaded successfully.")

    test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'external_test')
    os.makedirs(test_dir, exist_ok=True)
    
    tb_path = os.path.join(test_dir, 'tb_international.jpg')
    normal_path = os.path.join(test_dir, 'normal_international.jpg')
    invalid_path = os.path.join(test_dir, 'invalid_cat.jpg')
    
    tb_path = os.path.join(test_dir, 'tb.jpg')
    # Use a normal image from the local training data
    normal_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_guard', 'cxr', 'cxr_0.png')
    invalid_path = os.path.join(test_dir, 'invalid.jpg')
        
    images = [
        {'path': tb_path, 'type': 'Positive (Tuberculosis)', 'source': 'International Sample (TB.jpg)'},
        {'path': normal_path, 'type': 'Negative (Normal)', 'source': 'Local CXR Dataset'},
        {'path': invalid_path, 'type': 'Invalid (Not a CXR)', 'source': 'International Sample (Cat)'}
    ]

    for img_info in images:
        img_path = img_info['path']
        print(f"\n{'='*50}\n--- Testing Image: {os.path.basename(img_path)} ---\nType: {img_info['type']}\nSource: {img_info['source']}")
        
        # 1. Validation Engine
        passed, msg, gray_img = validation_engine.run_pipeline(img_path)
        if not passed:
            print(f"Validation Engine: REJECTED -> {msg}")
            continue
        print(f"Validation Engine: PASSED (Heuristics checks complete)")
        
        # 2. Guard AI
        guard_model, _ = guard_model_tuple
        confidence, prob = analyze_guard_confidence(guard_model, img_path)
        print(f"Guard AI Model: {confidence} (Score: {prob*100:.2f}%)")
        
        if confidence in ["Invalid", "Probably Non-CXR"] or confidence.startswith("Uncertain"):
            print("System Output: Image rejected by Guard Model.")
            continue
            
        # 3. DenseNet TB Inference
        tb_prob, prediction = infer_tb(tb_model_tuple, img_path, threshold, temperature)
        print(f"DenseNet TB Model: {prediction} (Probability: {tb_prob*100:.2f}%)")

if __name__ == '__main__':
    main()
