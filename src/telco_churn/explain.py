"""Tree SHAP contributions aggregated from encoded columns to raw feature names."""

import numpy as np
import pandas as pd
from .data import NUMERIC, CATEGORICAL


class Explainer:
    def __init__(self, pipeline):
        import shap
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(pipeline["model"], RandomForestClassifier):
            raise ValueError("Tree explanations require a Random Forest artifact")

        self.pipeline = pipeline
        self.explainer = shap.TreeExplainer(pipeline["model"])
        encoder = pipeline["preprocess"].named_transformers_["categorical"]["encode"]
        self.features = list(NUMERIC) + [
            name for name, cats in zip(CATEGORICAL, encoder.categories_) for _ in cats
        ]

    def explain(self, frame: pd.DataFrame, top_k: int = 5) -> dict:
        clean = self.pipeline["clean"].transform(frame)
        encoded = self.pipeline["preprocess"].transform(clean)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        values = np.asarray(self.explainer.shap_values(encoded))
        contributions = values[0, :, 1]
        totals = {name: 0.0 for name in NUMERIC + CATEGORICAL}
        for feature, contribution in zip(self.features, contributions):
            totals[feature] += float(contribution)
        baseline = float(np.asarray(self.explainer.expected_value)[1])
        prediction = float(self.pipeline.predict_proba(frame)[0, 1])
        if not np.isclose(baseline + sum(totals.values()), prediction, atol=1e-6):
            raise ValueError("SHAP additivity check failed")
        drivers = sorted(totals.items(), key=lambda pair: abs(pair[1]), reverse=True)[
            :top_k
        ]
        return {
            "baseline_probability": baseline,
            "all_contributions_sum": sum(totals.values()),
            "top_drivers": [
                {"feature": key, "contribution": value} for key, value in drivers
            ],
            "interpretation": "Model associations relative to its training-path baseline; not causal effects.",
        }
