"""
=============================================================
STEP 3: Symptom Predictor Service
=============================================================
Loads the trained XGBoost+MLP ensemble at startup and provides
real-time clinical symptom scoring at inference time.

Automatically falls back to WHO W4SS formula if model files
are not found — the system NEVER goes down.

Replaces: hardcoded process_clinical_data() WHO formula
Used by : backend/app/services/ml_pipeline-ak.py
=============================================================
"""

import pickle
import json
import numpy as np
from pathlib import Path

# ── Model Paths ───────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent   # → backend/
MODEL_DIR = BASE_DIR / "models"

# ── Lazy-loaded model state ───────────────────────────────────
_xgb    = None
_mlp    = None
_scaler = None
_meta   = None
_loaded = False

# The exact 6 features this model was trained on (order matters)
FEATURES = [
    "fever",
    "cough_duration_weeks",
    "weight_loss",
    "night_sweats",
    "sputum_test",
    "genexpert_test"
]

# WHO W4SS weights (clinical guideline fallback values)
WHO_WEIGHTS = {
    "cough_duration_weeks": 0.35,   # >= 2 weeks
    "cough_short":          0.10,   # < 2 weeks but present
    "fever":                0.20,
    "weight_loss":          0.25,
    "night_sweats":         0.20,
    "sputum_test":          0.45,
    "any_symptom_bonus":    0.10
}


def _load():
    """Lazy-load all model artifacts once on first call."""
    global _xgb, _mlp, _scaler, _meta, _loaded
    if _loaded:
        return
    _loaded = True

    try:
        xgb_path    = MODEL_DIR / "symptom_xgb.pkl"
        mlp_path    = MODEL_DIR / "symptom_mlp.pkl"
        scaler_path = MODEL_DIR / "symptom_scaler.pkl"
        meta_path   = MODEL_DIR / "symptom_model_meta.json"

        with open(xgb_path, "rb") as f:
            _xgb = pickle.load(f)
        with open(mlp_path, "rb") as f:
            _mlp = pickle.load(f)
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)

        if meta_path.exists():
            with open(meta_path) as f:
                _meta = json.load(f)
            model_type = _meta.get("model_type", "Unknown")
            auc = _meta.get("metrics", {}).get("roc_auc", "?")
            print(f"[Symptom-Model] Loaded: {model_type} | AUC={auc}")
        else:
            print("[Symptom-Model] Model loaded (no metadata file)")

    except FileNotFoundError as e:
        print(f"[Symptom-Model] WARN: {e}")
        print("[Symptom-Model] WHO W4SS fallback is active")
    except Exception as e:
        print(f"[Symptom-Model] ERROR during load: {e}")
        print("[Symptom-Model] WHO W4SS fallback is active")


def _who_w4ss_fallback(features: dict) -> dict:
    """
    WHO Four-Symptom Screen (W4SS) formula.
    Used when ML model is not available.
    Clinical reference: WHO Tuberculosis Screening Guidelines 2024
    """
    cough_val = float(features.get("cough_duration_weeks", 0) or 0)
    score = 0.0
    symptoms_present = 0

    # Cough scoring
    if cough_val >= 2:
        score += WHO_WEIGHTS["cough_duration_weeks"]
        symptoms_present += 1
    elif cough_val > 0:
        score += WHO_WEIGHTS["cough_short"]

    # Other symptoms
    for key in ["fever", "weight_loss", "night_sweats"]:
        if int(features.get(key, 0) or 0) == 1:
            score += WHO_WEIGHTS.get(key, 0)
            symptoms_present += 1

    # Sputum test
    if int(features.get("sputum_test", 0) or 0) == 1:
        score += WHO_WEIGHTS["sputum_test"]

    # Bonus if any symptom present
    if symptoms_present > 0:
        score += WHO_WEIGHTS["any_symptom_bonus"]

    return {
        "clinical_prob": round(min(0.97, score), 4),
        "source": "WHO W4SS Formula (fallback — train the model to activate ML)",
        "model_active": False,
        "explanation": {}
    }


def predict_clinical_score(features: dict) -> dict:
    """
    Main entry point for clinical symptom scoring.

    Args:
        features: dict containing any of:
            - fever (0/1)
            - cough_duration_weeks (int, weeks)
            - weight_loss (0/1)
            - night_sweats (0/1)
            - sputum_test (0/1)
            - genexpert_test (0/1)
            - no_symptoms (0/1) — patient explicitly reports no symptoms

    Returns:
        dict with keys:
            - clinical_prob: float 0.0-1.0
            - source: str describing which model/formula was used
            - model_active: bool
            - explanation: dict of SHAP values (if available)
    """
    _load()

    # ── Case 1: No symptoms declared ──────────────────────────
    if int(features.get("no_symptoms", 0) or 0) == 1:
        return {
            "clinical_prob": 0.05,
            "source": "No Symptoms Override",
            "model_active": False,
            "explanation": {}
        }

    # ── Case 2: GeneXpert positive = lab-confirmed TB ─────────
    if int(features.get("genexpert_test", 0) or 0) == 1:
        return {
            "clinical_prob": 0.97,
            "source": "GeneXpert Positive (Bacteriological Confirmation — Lab Verified)",
            "model_active": False,
            "explanation": {
                "genexpert_test": 0.97,
                "note": "Positive GeneXpert is considered definitive TB confirmation"
            }
        }

    # ── Case 3: Real ML model (if trained and loaded) ─────────
    if _xgb and _mlp and _scaler:
        try:
            cough_val = float(features.get("cough_duration_weeks", 0) or 0)
            cough_bin = 1 if cough_val >= 2 else 0

            x = np.array([[
                int(features.get("fever", 0) or 0),
                int(cough_bin),
                int(features.get("weight_loss", 0) or 0),
                int(features.get("night_sweats", 0) or 0),
                int(features.get("sputum_test", 0) or 0),
                0   # GeneXpert already handled above
            ]], dtype=np.float32)

            x_scaled = _scaler.transform(x)

            p_xgb = float(_xgb.predict_proba(x_scaled)[0][1])
            p_mlp = float(_mlp.predict_proba(x_scaled)[0][1])
            final_prob = round((p_xgb * 0.60) + (p_mlp * 0.40), 4)

            # ── SHAP explanations (if library installed) ───────
            explanation = {}
            try:
                import shap
                exp = shap.TreeExplainer(_xgb)
                sv  = exp.shap_values(x_scaled)
                explanation = {
                    feat: round(float(val), 4)
                    for feat, val in zip(FEATURES, sv[0])
                }
            except Exception:
                pass   # SHAP is optional

            return {
                "clinical_prob": final_prob,
                "xgb_prob": round(p_xgb, 4),
                "mlp_prob": round(p_mlp, 4),
                "source": "Real Clinical Model (XGBoost+MLP Ensemble — Mendeley Data)",
                "model_active": True,
                "explanation": explanation
            }

        except Exception as e:
            print(f"[Symptom-Model] Runtime inference error: {e}")
            print("[Symptom-Model] Falling back to WHO W4SS formula")

    # ── Case 4: WHO W4SS Fallback ──────────────────────────────
    return _who_w4ss_fallback(features)


def get_model_status() -> dict:
    """Return current model load status for health checks."""
    _load()
    return {
        "model_loaded": _xgb is not None and _mlp is not None,
        "scaler_loaded": _scaler is not None,
        "metadata": _meta,
        "fallback_active": _xgb is None
    }
