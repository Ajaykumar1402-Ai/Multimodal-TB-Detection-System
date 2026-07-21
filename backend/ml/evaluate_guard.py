import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from torchvision import models, transforms
from torch.utils.data import DataLoader
from train_guard_v2 import GuardDatasetV2

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Guard AI V2 on {device}")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ideally use a separate test directory, but for demonstration we evaluate on the training set
    dataset_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_guard')
    if not os.path.exists(dataset_root):
        print("Dataset not found.")
        return
        
    dataset = GuardDatasetV2(dataset_root, transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cxr_v2_guard_master.pth')
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    if len(np.unique(all_labels)) < 2:
        print("Cannot compute ROC/AUC with only 1 class present in evaluation data.")
        return
        
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('guard_roc_curve.png')
    print("Saved guard_roc_curve.png")
    
    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('guard_pr_curve.png')
    print("Saved guard_pr_curve.png")
    
    # Confusion Matrix
    pred_labels = (all_preds >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, pred_labels, target_names=["Non-CXR", "CXR"]))

if __name__ == '__main__':
    evaluate()
