"""Nested, training-only F1 threshold diagnostic; does not change the served artifact."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score
from .data import SEED, load_split
from .train import make_pipeline


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, probabilities >= threshold) for threshold in candidates]
    # Stable tie-breaking: smallest threshold among equally good F1 values.
    return float(candidates[int(np.argmax(scores))])


def experiment(data_path: str = "data/telco.csv") -> dict:
    X, _, y, _ = load_split(data_path)
    outer = StratifiedKFold(3, shuffle=True, random_state=SEED)
    probability = np.zeros(len(y))
    decisions = np.zeros(len(y), dtype=int)
    thresholds = []
    for train_idx, valid_idx in outer.split(X, y):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        search = GridSearchCV(
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=1,
                )
            ),
            {
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [2, 8],
                "model__max_features": ["sqrt", 0.7],
            },
            scoring="average_precision",
            cv=inner,
            n_jobs=2,
        )
        search.fit(train_X, train_y)
        # Refit the whole search in each inner OOF training fold: no label leaks into threshold scores.
        oof = cross_val_predict(
            search, train_X, train_y, cv=inner, method="predict_proba", n_jobs=1
        )[:, 1]
        threshold = choose_threshold(train_y.to_numpy(), oof)
        p = search.best_estimator_.predict_proba(X.iloc[valid_idx])[:, 1]
        probability[valid_idx] = p
        decisions[valid_idx] = p >= threshold
        thresholds.append(threshold)

    def metrics(pred):
        return {
            "f1": float(f1_score(y, pred)),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
        }

    return {
        "rows": len(y),
        "holdout_used": False,
        "default_model_changed": False,
        "outer_fold_thresholds": thresholds,
        "fixed_0_5": metrics(probability >= 0.5),
        "training_selected_f1": metrics(decisions),
        "caveat": "Nested 3-fold training-only diagnostic. Threshold chosen by inner OOF F1, not business cost. Outer estimates are evaluation, not selection. No single deployment threshold is inferred.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--output", default="reports/threshold_experiment.json")
    args = parser.parse_args()
    report = experiment(args.data)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
