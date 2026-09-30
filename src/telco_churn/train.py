"""Train and evaluate leakage-safe baselines on an untouched holdout."""

import argparse
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from .artifact import write_manifest
from .diagnostics import create_diagnostics
from .data import NUMERIC, CATEGORICAL, FEATURES, SEED, clean_features, load_split


def make_pipeline(model):
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_features, validate=False)),
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC),
                        ("categorical", categorical, CATEGORICAL),
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def metrics(y, prob):
    pred = prob >= 0.5
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc_ap": average_precision_score(y, prob),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train(data_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split(data_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc_ap": "average_precision", "f1": "f1"}
    baseline = make_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    )
    scores = cross_validate(
        baseline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=2
    )
    baseline.fit(X_train, y_train)
    forest = make_pipeline(
        RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=SEED, n_jobs=1
        )
    )
    search = GridSearchCV(
        forest,
        {
            "model__max_depth": [6, None],
            "model__min_samples_leaf": [2, 8],
            "model__max_features": ["sqrt", 0.7],
        },
        scoring=scoring,
        refit="pr_auc_ap",
        cv=cv,
        n_jobs=2,
    )
    search.fit(X_train, y_train)
    results = []
    for name, pipeline, cv_values in [
        (
            "logistic_regression",
            baseline,
            {m: float(scores[f"test_{m}"].mean()) for m in scoring},
        ),
        (
            "random_forest_tuned",
            search.best_estimator_,
            {
                m: float(search.cv_results_[f"mean_test_{m}"][search.best_index_])
                for m in scoring
            },
        ),
    ]:
        results.append(
            {
                "model": name,
                "cv": cv_values,
                "holdout": metrics(y_test, pipeline.predict_proba(X_test)[:, 1]),
            }
        )
    # Model choice uses training CV only, never the holdout comparison.
    selected = max(results, key=lambda r: r["cv"]["pr_auc_ap"])["model"]
    pipeline = baseline if selected == "logistic_regression" else search.best_estimator_
    joblib.dump(pipeline, output / "model.joblib", compress=3)
    write_manifest(output / "model.joblib")
    from .drift import make_reference

    (output / "drift_reference.json").write_text(
        json.dumps(make_reference(X_train), indent=2) + "\n"
    )
    report = {
        "seed": SEED,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "selected_model": selected,
        "selection_metric": "training CV average precision",
        "decision_threshold": 0.5,
        "forest_best_params": search.best_params_,
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "required_features": FEATURES,
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(search.cv_results_).to_csv(output / "cv_results.csv", index=False)
    create_diagnostics(pipeline, X_test, y_test, output.parent / "reports")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Opt-in MLflow URI; use a local file URI for private runs",
    )
    args = parser.parse_args()
    report = train(args.data, args.output)
    if args.tracking_uri:
        from .tracking import track_report

        report["mlflow_run_id"] = track_report(report, args.output, args.tracking_uri)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
