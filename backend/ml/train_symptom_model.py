"""
=============================================================
STEP 2: Real Clinical Symptom Model Training
=============================================================
Trains an XGBoost + MLP ensemble classifier on the merged
real clinical symptom dataset produced by prepare_real_symptoms.py

Input:  backend/data/clinical/real_symptoms_combined.csv
Output: backend/models/symptom_xgb.pkl
        backend/models/symptom_mlp.pkl
        backend/models/symptom_scaler.pkl
        backend/models/symptom_model_meta.json
=============================================================
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# Try XGBoost — fall back to LogisticRegression if not installed
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    print("[OK] XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] XGBoost not installed. Using LogisticRegression.")
    print("       Install: pip install xgboost")

# ── Paths ─────────────────────────────────────────────────────
CLINICAL_DIR = Path(__file__).resolve().parent.parent / "data" / "clinical"
MODEL_DIR    = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH    = CLINICAL_DIR / "real_symptoms_combined.csv"

FEATURES = [
    "fever",
    "cough_duration_weeks",
    "weight_loss",
    "night_sweats",
    "sputum_test",
    "genexpert_test"
]

RANDOM_SEED = 42


def load_data():
    """Load and validate the merged symptom CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset not found: {DATA_PATH}\n"
            "        Run prepare_real_symptoms.py first."
        )
    df = pd.read_csv(DATA_PATH)

    missing_cols = [f for f in FEATURES + ["label"] if f not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing columns: {missing_cols}")

    df = df.dropna(subset=["label"])
    df[FEATURES + ["label"]] = df[FEATURES + ["label"]].fillna(0).astype(int)

    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y, df


def train():
    print("\n" + "=" * 60)
    print("  TB-Vision: Clinical Symptom Model Training")
    print("=" * 60)

    X, y, df = load_data()

    print(f"\n  Dataset         : {len(df)} total records")
    print(f"  TB Positive     : {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  Normal          : {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Features        : {FEATURES}")

    # ── Stratified Split: 65% train / 15% val / 20% test ──────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )

    print(f"\n  Split   : Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # ── Feature Scaling ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ──────────────────────────────────────────────────────────
    # MODEL 1: XGBoost Classifier (or LogisticRegression fallback)
    # ──────────────────────────────────────────────────────────
    print("\n  [1/3] Training primary classifier...")
    if XGB_AVAILABLE:
        # Compute class imbalance weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos = max(1.0, n_neg / n_pos)

        primary_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            verbosity=0
        )
        primary_model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False
        )
        p1 = primary_model.predict_proba(X_test_s)[:, 1]
        print(f"  XGBoost AUC: {roc_auc_score(y_test, p1):.4f}")
        model_name = "XGBoost"

    else:
        primary_model = CalibratedClassifierCV(
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
            cv=5
        )
        primary_model.fit(X_train_s, y_train)
        p1 = primary_model.predict_proba(X_test_s)[:, 1]
        print(f"  LogisticRegression AUC: {roc_auc_score(y_test, p1):.4f}")
        model_name = "LogisticRegression"

    # ──────────────────────────────────────────────────────────
    # MODEL 2: MLP Neural Network
    # ──────────────────────────────────────────────────────────
    print("  [2/3] Training MLP neural network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,          # L2 regularization
        max_iter=1000,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20
    )
    mlp.fit(X_train_s, y_train)
    p2 = mlp.predict_proba(X_test_s)[:, 1]
    print(f"  MLP AUC: {roc_auc_score(y_test, p2):.4f}")

    # ──────────────────────────────────────────────────────────
    # ENSEMBLE: 60% primary + 40% MLP
    # ──────────────────────────────────────────────────────────
    print("  [3/3] Computing ensemble...")
    ensemble_prob = (p1 * 0.60) + (p2 * 0.40)
    y_pred = (ensemble_prob >= 0.50).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, ensemble_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n  {'='*50}")
    print(f"  ENSEMBLE RESULTS ({model_name} 60% + MLP 40%)")
    print(f"  {'='*50}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%   <- Must be >= 90%")
    print(f"  F1-Score   : {f1*100:.2f}%")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  {'='*50}")
    print(f"\n  Per-Class Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))

    # WHO targets check
    if rec >= 0.90:
        print("  [PASS] Recall >= 90% — WHO Target Met!")
    else:
        print(f"  [WARN] Recall {rec*100:.1f}% below WHO 90% target")
        print("         Consider: more data, lower threshold, or class weights")

    # ── 5-Fold Cross-Validation ────────────────────────────────
    print("\n  Running 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_auc = cross_val_score(primary_model, X_train_s, y_train, cv=cv, scoring="roc_auc")
    cv_rec = cross_val_score(primary_model, X_train_s, y_train, cv=cv, scoring="recall")
    print(f"  CV AUC    : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
    print(f"  CV Recall : {cv_rec.mean():.4f} +/- {cv_rec.std():.4f}")

    # ── SHAP Feature Importance ────────────────────────────────
    feature_importance = {}
    if XGB_AVAILABLE:
        try:
            import shap
            explainer = shap.TreeExplainer(primary_model)
            shap_vals = explainer.shap_values(X_test_s)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            feature_importance = {f: round(float(v), 4) for f, v in zip(FEATURES, mean_abs)}
            print(f"\n  SHAP Feature Importance (higher = more influential):")
            for k, v in sorted(feature_importance.items(), key=lambda x: -x[1]):
                bar = "█" * int(v * 40 / max(feature_importance.values(), default=1))
                print(f"    {k:25s} {bar} {v:.4f}")
        except ImportError:
            print("  [WARN] SHAP not installed (pip install shap). Using XGBoost native importance.")
            raw_imp = primary_model.feature_importances_
            feature_importance = {f: round(float(v), 4) for f, v in zip(FEATURES, raw_imp)}
        except Exception as e:
            print(f"  [WARN] SHAP error: {e}")

    # ── Save All Models ────────────────────────────────────────
    print("\n  Saving models...")
    with open(MODEL_DIR / "symptom_xgb.pkl", "wb") as f:
        pickle.dump(primary_model, f)
    print(f"  [OK] symptom_xgb.pkl  ({model_name})")

    with open(MODEL_DIR / "symptom_mlp.pkl", "wb") as f:
        pickle.dump(mlp, f)
    print(f"  [OK] symptom_mlp.pkl  (MLP)")

    with open(MODEL_DIR / "symptom_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [OK] symptom_scaler.pkl")

    # ── Save Metadata JSON ─────────────────────────────────────
    meta = {
        "model_type": f"{model_name}+MLP Ensemble",
        "primary_model": model_name,
        "ensemble_weights": {"primary": 0.60, "mlp": 0.40},
        "features": FEATURES,
        "n_features": len(FEATURES),
        "threshold": 0.50,
        "dataset_size": int(len(df)),
        "data_source": "Mendeley TB Clinical + Semarang + Kaggle (CC BY 4.0)",
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "metrics": {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc, 4),
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std": round(float(cv_auc.std()), 4),
            "cv_recall_mean": round(float(cv_rec.mean()), 4)
        },
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp)
        },
        "feature_importance": feature_importance,
        "who_targets": {
            "recall_90pct_met": bool(rec >= 0.90),
            "auc_target_met": bool(auc >= 0.90)
        }
    }

    meta_path = MODEL_DIR / "symptom_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] symptom_model_meta.json")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Models saved in: {MODEL_DIR}")
    print(f"  AUC Score      : {auc:.4f}")
    print(f"  Recall         : {rec*100:.2f}%")
    print(f"\n  Next step: Restart backend server")
    print(f"             The symptom model activates automatically")
    print("=" * 60)

    return meta


if __name__ == "__main__":
    train()
