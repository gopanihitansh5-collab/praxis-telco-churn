"""Tenure-cohort shift study; tenure proxies arrival cohort and is not a timestamp."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .data import SEED, load_split
from .drift import compare, make_reference
from .train import make_pipeline

COHORT_SPLIT_MONTHS = 12
METRICS = ("roc_auc", "average_precision", "f1", "precision", "recall")
CONTROL_REPEATS = 3
WITHIN_FOLDS = 3
MIN_COHORT_ROWS = 30


def default_estimator(seed: int = SEED) -> RandomForestClassifier:
    """One fixed configuration for every fit, so no cohort informs model selection."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def tenure_months(X: pd.DataFrame) -> pd.Series:
    """Blank strings become NaN here exactly as data.clean_features treats them."""
    return pd.to_numeric(X["tenure"], errors="coerce")


def split_by_tenure(
    X: pd.DataFrame, y: pd.Series, boundary: int = COHORT_SPLIT_MONTHS
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Established (tenure >= boundary) against recent (tenure < boundary).

    Missing tenure places a row in neither cohort: assigning it to either side
    would silently invent a cohort membership. Those rows leave the experiment
    entirely and the report states how many were dropped.
    """
    tenure = tenure_months(X)
    established, recent = tenure >= boundary, tenure < boundary
    return X[established], y[established], X[recent], y[recent]


def _scores(y: pd.Series, probability: np.ndarray) -> dict:
    pred = probability >= 0.5
    return {
        "n": int(len(y)),
        "churn_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
    }


def _transfer(
    factory: Callable[[int], object],
    seed: int,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> dict:
    pipeline = make_pipeline(factory(seed))
    pipeline.fit(train_X, train_y)
    result = _scores(test_y, pipeline.predict_proba(test_X)[:, 1])
    result["train_n"] = int(len(train_y))
    result["train_churn_rate"] = float(np.mean(train_y))
    return result


def _within_cohort(
    factory: Callable[[int], object], seed: int, X: pd.DataFrame, y: pd.Series
) -> dict:
    cv = StratifiedKFold(WITHIN_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(
        make_pipeline(factory(seed)), X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    result = _scores(y, oof)
    result["folds"] = WITHIN_FOLDS
    return result


def _random_control(
    factory: Callable[[int], object],
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    n_train: int,
    n_test: int,
) -> dict:
    """Random stratified split of the pooled cohorts at the cohort split's sizes."""
    runs = []
    for offset in range(CONTROL_REPEATS):
        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            train_size=n_train,
            test_size=n_test,
            stratify=y,
            random_state=seed + offset,
        )
        runs.append(_transfer(factory, seed, train_X, train_y, test_X, test_y))
    return {
        "n": int(n_test),
        "train_n": int(n_train),
        "repeats": CONTROL_REPEATS,
        "churn_rate": float(np.mean([r["churn_rate"] for r in runs])),
        **{m: float(np.mean([r[m] for r in runs])) for m in METRICS},
        "per_repeat_roc_auc": [r["roc_auc"] for r in runs],
    }


def _covariate_shift(established_X: pd.DataFrame, recent_X: pd.DataFrame) -> dict:
    """Reuses the existing PSI prototype rather than reimplementing binning."""
    if min(len(established_X), len(recent_X)) < 100:
        return {
            "available": False,
            "reason": "the PSI prototype requires at least 100 rows per cohort",
        }
    result = compare(make_reference(established_X), recent_X)
    features = result["features"]
    others = {k: v for k, v in features.items() if k != "tenure"}
    return {
        "available": True,
        "reference_cohort": "established",
        "compared_cohort": "recent",
        "features": features,
        "flagged_features": sorted(k for k, v in features.items() if v["review_flag"]),
        "max_psi_excluding_tenure": max(v["psi"] for v in others.values()),
        "tenure_psi_is_definitional": True,
        "caveat": result["caveat"],
    }


def _gaps(transfer: dict, control: dict, within: dict) -> dict:
    return {
        metric: {
            "transfer": transfer[metric],
            "random_split_control": control[metric],
            "within_cohort": within[metric],
            "transfer_minus_control": transfer[metric] - control[metric],
            "transfer_minus_within_cohort": transfer[metric] - within[metric],
        }
        for metric in ("roc_auc", "average_precision")
    }


def _describe(X: pd.DataFrame, y: pd.Series, boundary: int) -> dict:
    tenure = tenure_months(X)
    return {
        "n": int(len(y)),
        "churn_rate": float(np.mean(y)),
        "tenure_min": float(tenure.min()),
        "tenure_max": float(tenure.max()),
        "boundary_months": boundary,
    }


def evaluate_shift(
    X: pd.DataFrame,
    y: pd.Series,
    boundary: int = COHORT_SPLIT_MONTHS,
    estimator_factory: Callable[[int], object] = default_estimator,
    seed: int = SEED,
) -> dict:
    """Cross-cohort transfer with a same-size random control and within-cohort refs.

    `estimator_factory(seed)` returns an unfitted estimator, which is always wrapped
    in `train.make_pipeline` so imputation, scaling and encoding stay inside the fit.
    """
    established_X, established_y, recent_X, recent_y = split_by_tenure(X, y, boundary)
    missing_tenure = int(tenure_months(X).isna().sum())
    for name, cohort_y in [("established", established_y), ("recent", recent_y)]:
        if len(cohort_y) < MIN_COHORT_ROWS:
            raise ValueError(
                f"{name} cohort has {len(cohort_y)} rows; "
                f"boundary {boundary} needs at least {MIN_COHORT_ROWS}"
            )
        if cohort_y.nunique() < 2:
            raise ValueError(f"{name} cohort is single-class at boundary {boundary}")
    pool_X = pd.concat([established_X, recent_X])
    pool_y = pd.concat([established_y, recent_y])

    transfer = {
        "established_to_recent": _transfer(
            estimator_factory, seed, established_X, established_y, recent_X, recent_y
        ),
        "recent_to_established": _transfer(
            estimator_factory, seed, recent_X, recent_y, established_X, established_y
        ),
    }
    within = {
        "established": _within_cohort(
            estimator_factory, seed, established_X, established_y
        ),
        "recent": _within_cohort(estimator_factory, seed, recent_X, recent_y),
    }
    control = {
        "established_to_recent": _random_control(
            estimator_factory,
            seed,
            pool_X,
            pool_y,
            len(established_y),
            len(recent_y),
        ),
        "recent_to_established": _random_control(
            estimator_factory,
            seed,
            pool_X,
            pool_y,
            len(recent_y),
            len(established_y),
        ),
    }
    return {
        "experiment": (
            "train on one tenure cohort, score the other; compared against a random "
            "stratified split of the same pooled rows at the same train/test sizes"
        ),
        "boundary_months": boundary,
        "seed": seed,
        "rows": int(len(y)),
        "rows_evaluated": int(len(pool_y)),
        "excluded_missing_tenure": missing_tenure,
        "holdout_used": False,
        "default_model_changed": False,
        "estimator": str(estimator_factory(seed)),
        "tuning": "none; one fixed configuration for every fit, no per-cohort search",
        "cohorts": {
            "established": _describe(established_X, established_y, boundary),
            "recent": _describe(recent_X, recent_y, boundary),
        },
        "covariate_shift": _covariate_shift(established_X, recent_X),
        "cross_cohort_transfer": transfer,
        "within_cohort_reference": within,
        "random_split_control": control,
        "gap_summary": {
            "established_to_recent": _gaps(
                transfer["established_to_recent"],
                control["established_to_recent"],
                within["recent"],
            ),
            "recent_to_established": _gaps(
                transfer["recent_to_established"],
                control["recent_to_established"],
                within["established"],
            ),
        },
        "caveats": [
            "Tenure is a PROXY for arrival cohort, not a timestamp. This dataset has "
            "no dates, so nothing here is temporal validation and no claim about "
            "future periods follows from it.",
            "CONFOUND: short-tenure customers churn far more and tenure is itself one "
            "of the strongest predictors, so a transfer drop mixes covariate shift "
            "with the intrinsic difficulty of the new-customer segment.",
            "transfer_minus_within_cohort holds the scored cohort fixed and isolates "
            "the effect of the training source, but the within-cohort model trains on "
            "fewer rows, so part of that difference is sample size.",
            "transfer_minus_control holds train/test sizes and the pooled population "
            "fixed and keeps both effects together; it is the comparison that answers "
            "'does cohort structure cost anything a random split would not show'.",
            "Compare against random_split_control, never against the headline holdout "
            "ROC-AUC: that figure comes from a different sample size, a different "
            "population and a tuned model.",
            "within_cohort_reference is 3-fold out-of-fold prediction inside a single "
            "cohort; it is an in-distribution reference, not a deployable score.",
            "Tenure PSI between the cohorts is definitional because the split is on "
            "tenure. Only max_psi_excluding_tenure is evidence of genuine covariate "
            "shift, and 0.2 remains a heuristic flag.",
            "No hyperparameter search anywhere, so every score here is below what the "
            "tuned bundled forest reports for that reason alone.",
            "Rows with missing tenure belong to no cohort and were removed from the "
            "transfer arms, the within-cohort references and the control pool alike.",
            "One boundary, one seed family, no confidence intervals. This is a "
            "diagnostic, not a monitoring policy or a retraining trigger.",
            "Original training partition only; the 1,409-row holdout is untouched and "
            "the bundled default model is unchanged.",
        ],
    }


def create_chart(report: dict, output_dir: str | Path = "reports") -> Path:
    directions = ["established_to_recent", "recent_to_established"]
    labels = ["Train established\nscore recent", "Train recent\nscore established"]
    series = [
        ("transfer", "Cross-cohort transfer", "#315d88"),
        ("random_split_control", "Random split, same sizes", "#cb7546"),
        ("within_cohort", "Within-cohort reference", "#8c8c8c"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(directions))
    for ax, metric, title in zip(
        axes, ("roc_auc", "average_precision"), ("ROC-AUC", "Average precision")
    ):
        for offset, (key, label, color) in enumerate(series):
            ax.bar(
                positions + (offset - 1) * 0.27,
                [report["gap_summary"][d][metric][key] for d in directions],
                0.25,
                label=label,
                color=color,
            )
        ax.set(ylabel=title, title=f"{title} by evaluation condition")
        ax.set_xticks(positions, labels, fontsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cohort_shift.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--boundary", type=int, default=COHORT_SPLIT_MONTHS)
    parser.add_argument("--output", default="reports/cohort_shift.json")
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(
            f"{args.data} not found. Fetch the real dataset with download_data.py; "
            "this study never fabricates or substitutes data."
        )
    X_train, _, y_train, _ = load_split(args.data)
    report = evaluate_shift(X_train, y_train, args.boundary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    create_chart(report, output.parent)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
