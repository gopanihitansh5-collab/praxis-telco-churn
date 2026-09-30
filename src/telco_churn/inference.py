"""Typed single-profile prediction, shared by CLI and HTTP service."""

from pathlib import Path
from typing import Any
from typing_extensions import TypedDict
import pandas as pd
from .artifact import load_model
from .data import FEATURES, clean_features


class Prediction(TypedDict):
    prediction: int
    churn: str
    churn_probability: float
    threshold: float


class Predictor:
    """Load once and serve many profiles without learning from request data."""

    def __init__(self, model_path: str | Path = "artifacts/model.joblib") -> None:
        self.model = load_model(model_path)

    def predict(self, payload: dict[str, Any]) -> Prediction:
        if not isinstance(payload, dict):
            raise ValueError("Input must be one JSON object")
        unexpected = sorted(set(payload) - set(FEATURES) - {"customerID"})
        if unexpected:
            raise ValueError(f"Unexpected fields: {unexpected}")
        if "customerID" in payload and not isinstance(
            payload["customerID"], (str, type(None))
        ):
            raise ValueError("customerID must be a string or null")
        from .schema import Customer

        Customer.model_validate(payload)
        frame = pd.DataFrame([payload])
        clean_features(frame)
        probability = float(self.model.predict_proba(frame)[0, 1])
        return {
            "prediction": int(probability >= 0.5),
            "churn": "Yes" if probability >= 0.5 else "No",
            "churn_probability": probability,
            "threshold": 0.5,
        }


def predict_customer(
    payload: dict[str, Any], model_path: str | Path = "artifacts/model.joblib"
) -> Prediction:
    return Predictor(model_path).predict(payload)
