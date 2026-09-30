"""Single-profile inference with explicit schema checks."""
import joblib
import pandas as pd
from .data import FEATURES, clean_features


def predict_customer(payload, model_path='artifacts/model.joblib'):
    if not isinstance(payload, dict):
        raise ValueError('Input must be one JSON object')
    unexpected = sorted(set(payload) - set(FEATURES) - {'customerID'})
    if unexpected:
        raise ValueError(f'Unexpected fields: {unexpected}')
    frame = pd.DataFrame([payload])
    clean_features(frame)  # Clear input errors before loading the model.
    model = joblib.load(model_path)  # Load trusted local artifacts only.
    probability = float(model.predict_proba(frame)[0, 1])
    return {'prediction': int(probability >= 0.5),
            'churn': 'Yes' if probability >= 0.5 else 'No',
            'churn_probability': probability, 'threshold': 0.5}
