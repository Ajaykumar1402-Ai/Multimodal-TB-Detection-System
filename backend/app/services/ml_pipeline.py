import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialize/Mock the ML models
# MobileNetV2 pretrained on ImageNet as a base, in a real scenario we'd load fine-tuned weights
try:
    cnn_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
except Exception as e:
    cnn_model = None
    print(f"Warning: Could not load MobileNetV2. Error: {e}")

def process_xray_image(image_bytes: bytes) -> float:
    # Simulates TB inference on an X-ray
    try:
        if cnn_model is None:
            return 0.5 # fallback mock
            
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        preds = cnn_model.predict(img_array)
        # Using a simulated probability based on the model output for demo purposes
        # Since this is imagenet, we just hash the top prediction probability as a TB score mock
        prob = float(np.max(preds[0]))
        
        # Add some reasonable randomness for demo variation between 0.1 and 0.95
        np.random.seed(int(prob * 10000) % 1000)
        tb_prob = np.random.uniform(0.1, 0.95)
        return round(tb_prob, 4)
    except Exception:
        return 0.5

def process_clinical_data(features: dict) -> float:
    # Logistic Regression mock logic for TB
    # Age, cough (weeks), fever (1/0), weight_loss (1/0), night_sweats (1/0)
    score = 0.0
    if features.get("cough_duration_weeks", 0) > 2:
        score += 0.3
    if features.get("fever", 0) == 1:
        score += 0.2
    if features.get("weight_loss", 0) == 1:
        score += 0.25
    if features.get("night_sweats", 0) == 1:
        score += 0.15
        
    return min(0.99, score + 0.05)

def multimodal_fusion(img_prob: float, clinical_prob: float) -> dict:
    # Combine weighting: 60% Image, 40% Clinical
    final_prob = (img_prob * 0.6) + (clinical_prob * 0.4)
    
    if final_prob > 0.7:
        risk = "High"
        rec = "Immediate medical evaluation required. Isolate patient and schedule sputum test."
    elif final_prob > 0.4:
        risk = "Medium"
        rec = "Further clinical testing recommended. Monitor symptoms."
    else:
        risk = "Low"
        rec = "TB unlikely. Consider other respiratory conditions if symptoms persist."
        
    return {
        "final_prob": round(final_prob, 4),
        "risk_level": risk,
        "recommendations": rec
    }
