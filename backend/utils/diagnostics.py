import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, confusion_matrix, 
                             brier_score_loss, accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score)
import torch
from captum.attr import LayerGradCam
from torchvision.transforms import functional as F

class DiagnosticsManager:
    def __init__(self, output_dir="./models/training_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def evaluate_metrics(self, trues, probs, threshold=0.5):
        preds = (np.array(probs) >= threshold).astype(int)
        trues = np.array(trues)
        
        # Base metrics
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, zero_division=0)
        mcc = matthews_corrcoef(trues, preds)
        bal_acc = balanced_accuracy_score(trues, preds)
        
        # ROC and PR AUC
        try:
            fpr, tpr, _ = roc_curve(trues, probs)
            roc_auc = auc(fpr, tpr)
        except Exception: roc_auc = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(trues, probs)
            pr_auc = auc(recall_curve, precision_curve)
        except Exception: pr_auc = 0.0
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(trues, preds, labels=[0,1]).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        return {
            "Accuracy": acc,
            "Sensitivity (Recall)": sens,
            "Specificity": spec,
            "Precision (PPV)": ppv,
            "NPV": npv,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "MCC": mcc,
            "Balanced Accuracy": bal_acc
        }

    def save_predictions_csv(self, prefix, filenames, trues, probs, threshold=0.5):
        preds = (np.array(probs) >= threshold).astype(int)
        correct = (preds == np.array(trues))
        
        if len(filenames) != len(trues):
            filenames = [f"image_{i}.png" for i in range(len(trues))]
            
        df = pd.DataFrame({
            "Filename": filenames,
            "Ground Truth": trues,
            "Predicted Probability": probs,
            "Prediction": preds,
            "Confidence": np.where(preds == 1, probs, 1.0 - np.array(probs)),
            "Correct / Incorrect": ["Correct" if c else "Incorrect" for c in correct]
        })
        csv_path = os.path.join(self.output_dir, f"{prefix}_predictions.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def plot_roc_curve(self, prefix, trues, probs):
        fpr, tpr, _ = roc_curve(trues, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {prefix}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_roc.png"))
        plt.close()

    def plot_pr_curve(self, prefix, trues, probs):
        precision, recall, _ = precision_recall_curve(trues, probs)
        pr_auc = auc(recall, precision)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title(f'Precision-Recall Curve - {prefix}')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_pr.png"))
        plt.close()

    def plot_confusion_matrix(self, prefix, trues, probs, threshold=0.5):
        preds = (np.array(probs) >= threshold).astype(int)
        cm = confusion_matrix(trues, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Thresh={threshold}) - {prefix}')
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_cm.png"))
        plt.close()

    def plot_probability_histogram(self, prefix, trues, probs):
        df = pd.DataFrame({"prob": probs, "label": trues})
        plt.figure()
        sns.histplot(data=df, x="prob", hue="label", bins=50, kde=True)
        plt.title(f'Probability Histogram - {prefix}')
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_prob_hist.png"))
        plt.close()

    def plot_calibration_curve(self, prefix, trues, probs, n_bins=10):
        bin_limits = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_limits[:-1]
        bin_uppers = bin_limits[1:]
        
        accs, confs = [], []
        ece = 0.0
        
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > lower) & (probs <= upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                acc = trues[in_bin].mean()
                conf = probs[in_bin].mean()
                accs.append(acc)
                confs.append(conf)
                ece += np.abs(acc - conf) * prop_in_bin
                
        plt.figure()
        plt.plot(confs, accs, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability (Confidence)')
        plt.ylabel('Fraction of Positives (Accuracy)')
        plt.title(f'Reliability Diagram - {prefix} (ECE: {ece:.4f})')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_calibration.png"))
        plt.close()
        
        brier = brier_score_loss(trues, probs)
        return ece, brier

    def plot_learning_curves(self, prefix, history):
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(epochs, history['val_auc'], label='Val AUC', color='green')
        plt.title('ROC-AUC')
        plt.legend()
        
        plt.subplot(2, 3, 3)
        plt.plot(epochs, history['lr'], label='LR', color='red')
        plt.title('Learning Rate')
        plt.legend()
        
        plt.subplot(2, 3, 4)
        plt.plot(epochs, history['val_sens'], label='Sensitivity')
        plt.title('Sensitivity')
        plt.legend()
        
        plt.subplot(2, 3, 5)
        plt.plot(epochs, history['val_spec'], label='Specificity')
        plt.title('Specificity')
        plt.legend()
        
        plt.subplot(2, 3, 6)
        plt.plot(epochs, history['val_bal_acc'], label='Balanced Acc')
        plt.title('Balanced Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_learning_curves.png"))
        plt.close()
