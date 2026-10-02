"""Calibrated companion artifact and out-of-fold scores; the reviewed default is unchanged."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from .artifact import write_manifest
from .data import SEED, load_split
from .train import make_pipeline

# Reusing the already-selected configuration keeps this a calibration step, not a
# second model search. Re-searching here would make the comparison selection-optimistic.
SELECTED_PARAMS = {"max_depth": 6, "max_features": 0.7, "min_samples_leaf": 2}
OUTER_FOLDS = 5
INNER_FOLDS = 3


def read_selected_params(metrics_path: str | Path) -> dict:
    """Prefer the recorded training run's parameters over the hardcoded fallback."""
    path = Path(metrics_path)
    if not path.exists():
        return dict(SELECTED_PARAMS)
    recorded = json.loads(path.read_text()).get("forest_best_params", {})
    parsed = {
        key.removeprefix("model__"): value
        for key, value in recorded.items()
        if key.startswith("model__")
    }
    return parsed or dict(SELECTED_PARAMS)


def build_estimator(params: dict) -> object:
    return make_pipeline(
        RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
            **params,
        )
    )


def build_calibrated(params: dict, folds: int = INNER_FOLDS) -> CalibratedClassifierCV:
    inner = StratifiedKFold(folds, shuffle=True, random_state=SEED + 1)
    return CalibratedClassifierCV(
        build_estimator(params), method="sigmoid", cv=inner, ensemble=True, n_jobs=1
    )


def summarize(y, probability) -> dict:
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
    }


def out_of_fold_scores(X, y, params: dict, folds: int = OUTER_FOLDS) -> dict:
    """Outer-fold scores for both variants; calibration is refit inside every fold."""
    outer = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    raw = cross_val_predict(
        build_estimator(params), X, y, cv=outer, method="predict_proba", n_jobs=2
    )[:, 1]
    calibrated = cross_val_predict(
        build_calibrated(params), X, y, cv=outer, method="predict_proba", n_jobs=2
    )[:, 1]
    return {"raw": raw, "calibrated": calibrated}


def run(data_path: str | Path, output_dir: str | Path, reports_dir: str | Path) -> dict:
    X, _, y, _ = load_split(data_path)
    output, reports = Path(output_dir), Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    params = read_selected_params(output / "metrics.json")
    scores = out_of_fold_scores(X, y, params)

    # The policy layer must consume out-of-fold scores. In-sample probabilities from a
    # fitted forest are near-separable and would inflate any expected-value estimate.
    frame = pd.DataFrame(
        {
            "probability": scores["calibrated"],
            "probability_uncalibrated": scores["raw"],
            "MonthlyCharges": pd.to_numeric(X["MonthlyCharges"].to_numpy()),
            "churn": y.to_numpy(),
        }
    )
    frame.to_csv(output / "oof_probabilities.csv", index=False)

    calibrated = build_calibrated(params)
    calibrated.fit(X, y)
    joblib.dump(calibrated, output / "model_calibrated.joblib", compress=3)
    write_manifest(output / "model_calibrated.joblib")

    report = {
        "purpose": "Probability quality for the decision layer, not a new model selection.",
        "selected_params": params,
        "outer_folds": OUTER_FOLDS,
        "inner_calibration_folds": INNER_FOLDS,
        "rows": int(len(y)),
        "holdout_used": False,
        "default_model_changed": False,
        "uncalibrated": summarize(y, scores["raw"]),
        "calibrated": summarize(y, scores["calibrated"]),
        "mean_predicted_calibrated": float(np.mean(scores["calibrated"])),
        "observed_churn_rate": float(np.mean(y)),
        "caveats": [
            "Training-partition outer folds only; the 1,409-row holdout is untouched.",
            "artifacts/model.joblib remains the reviewed default. The calibrated file is a"
            " separate opt-in artifact selected explicitly by path.",
            "Sigmoid calibration with ensemble=True also averages several fitted models, so"
            " differences are not attributable to the sigmoid map alone.",
            "Better in-sample Brier and log loss do not establish future calibration on new"
            " customers, and ranking metrics are largely unchanged by a monotone map.",
        ],
    }
    (reports / "calibration_served.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset not found: {args.data}. Run python download_data.py first."
        )
    print(json.dumps(run(args.data, args.output, args.reports), indent=2))


if __name__ == "__main__":
    main()
