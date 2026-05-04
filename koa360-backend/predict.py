import sys
import json
import joblib
import pandas as pd
import numpy as np
import os
import shap

# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

model = joblib.load(MODEL_PATH)

input_data = json.loads(sys.stdin.read())
features = input_data["features"]

FEATURE_COLUMNS = [
    "age","gender","height","weight","occupation","living_environment",
    "knee_pain","knee_pain_in_past_week","stifness_after_resting",
    "knee_injuries","swelling","difficulty_in_performing",
    "family_history","obesity","diabetes","hypertension",
    "vitaminD_deficiency","rheumatoid_arthritis","fbs","wbc",
    "platelets","cs","cholesterol","crp","esr","rf","fbc",
    "physical_activity_level_Low","physical_activity_level_Moderate","BMI"
]

X = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

prediction = model.predict(X)[0]

confidence = None
if hasattr(model, "predict_proba"):
    confidence = float(np.max(model.predict_proba(X)[0]))

# SHAP values for Explainable AI
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle list return (classification) vs array return
    if isinstance(shap_values, list):
        vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        # shap_values could be a 3D array or 2D array depending on shap version and model type
        if len(shap_values.shape) == 3: # Binary classification might return (samples, features, classes)
            vals = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
        else:
            vals = shap_values[0]

    explanations = []
    for i, col in enumerate(FEATURE_COLUMNS):
        explanations.append({"feature": col, "value": float(vals[i])})

    explanations.sort(key=lambda x: abs(x["value"]), reverse=True)
    top_5_explanations = explanations[:5]
except Exception as e:
    top_5_explanations = []

print(json.dumps({
    "model": "GB (Best Model)",
    "prediction": str(prediction),
    "confidence": confidence,
    "explanations": top_5_explanations
}))
