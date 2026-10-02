import json
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from telco_churn.cohort import (
    COHORT_SPLIT_MONTHS,
    create_chart,
    evaluate_shift,
    main,
    split_by_tenure,
)
from telco_churn.data import CATEGORICAL

CATEGORY_VALUES = {name: [f"{name}_a", f"{name}_b"] for name in CATEGORICAL}


def cheap_estimator(seed: int) -> LogisticRegression:
    """Deliberately tiny: these tests never fit the 150-tree tuned grid."""
    return LogisticRegression(max_iter=500, random_state=seed)


def frame(tenure, rng=None, charges=None):
    rng = rng or np.random.default_rng(0)
    n = len(tenure)
    data = {
        "SeniorCitizen": rng.integers(0, 2, n),
        "tenure": tenure,
        "MonthlyCharges": rng.normal(70.0, 15.0, n) if charges is None else charges,
        "TotalCharges": np.clip(rng.normal(2000.0, 400.0, n), 1.0, None),
    }
    for name, values in CATEGORY_VALUES.items():
        data[name] = rng.choice(values, n)
    return pd.DataFrame(data)


def synthetic(n=400, seed=0, shift=False, missing=0):
    """Cohorts differ only in tenure unless `shift`, which moves MonthlyCharges."""
    rng = np.random.default_rng(seed)
    half = n // 2
    tenure = np.r_[
        rng.integers(1, COHORT_SPLIT_MONTHS, half),
        rng.integers(COHORT_SPLIT_MONTHS, 61, n - half),
    ].astype(float)
    recent = tenure < COHORT_SPLIT_MONTHS
    charges = np.clip(rng.normal(70.0, 15.0, n), 5.0, None)
    if shift:
        charges = charges + np.where(recent, 60.0, 0.0)
    X = frame(tenure, rng=rng, charges=charges)
    standardized = (charges - charges.mean()) / charges.std()
    y = pd.Series(
        (rng.random(n) < 1.0 / (1.0 + np.exp(-1.8 * standardized))).astype(int)
    )
    if missing:
        X.loc[X.index[:missing], "tenure"] = np.nan
    return X, y


@pytest.fixture(scope="module")
def unshifted_report():
    X, y = synthetic(n=800, seed=1)
    return evaluate_shift(X, y, estimator_factory=cheap_estimator, seed=7)


def test_split_partitions_at_boundary():
    X = frame([0.0, 5.0, 11.0, 12.0, 13.0, 60.0, np.nan])
    y = pd.Series([0, 1, 0, 1, 0, 1, 0])
    established_X, established_y, recent_X, recent_y = split_by_tenure(X, y)
    assert list(established_X["tenure"]) == [12.0, 13.0, 60.0]
    assert list(recent_X["tenure"]) == [0.0, 5.0, 11.0]
    assert not set(established_X.index) & set(recent_X.index)
    # Exhaustive over known tenure only; the NaN row joins neither cohort.
    assert len(established_X) + len(recent_X) + 1 == len(X)
    assert list(established_y.index) == list(established_X.index)
    assert list(recent_y.index) == list(recent_X.index)
    wider = split_by_tenure(X, y, 24)
    assert list(wider[0]["tenure"]) == [60.0]


def test_missing_tenure_is_reported_not_silently_dropped():
    X, y = synthetic(n=300, seed=2, missing=11)
    report = evaluate_shift(X, y, estimator_factory=cheap_estimator, seed=7)
    assert report["excluded_missing_tenure"] == 11
    assert report["rows"] == 300
    assert report["rows_evaluated"] == 289
    assert any("missing tenure" in c for c in report["caveats"])


def test_report_keys_and_honesty_flags(unshifted_report):
    report = unshifted_report
    assert report["holdout_used"] is False
    assert report["default_model_changed"] is False
    assert report["tuning"].startswith("none")
    for direction in ("established_to_recent", "recent_to_established"):
        for section in ("cross_cohort_transfer", "random_split_control", "gap_summary"):
            assert direction in report[section]
        transfer = report["cross_cohort_transfer"][direction]
        for metric in ("roc_auc", "average_precision", "f1", "precision", "recall"):
            assert metric in transfer
        assert transfer["n"] > 0
        assert 0.0 <= transfer["churn_rate"] <= 1.0
    for cohort in ("established", "recent"):
        assert report["within_cohort_reference"][cohort]["n"] > 0
    assert report["covariate_shift"]["available"] is True
    assert any("PROXY" in c for c in report["caveats"])
    assert any("CONFOUND" in c for c in report["caveats"])


@pytest.mark.parametrize("seed", [1, 21])
def test_no_manufactured_shift_when_cohorts_match(seed):
    """The harness must not report transfer loss that the data does not contain.

    The bound is 0.08, not 0.01: the cohorts still differ in tenure range by
    construction, so training inside one band extrapolates slightly onto the
    other. A manufactured shift would show an order of magnitude more.
    """
    X, y = synthetic(n=800, seed=seed)
    report = evaluate_shift(X, y, estimator_factory=cheap_estimator, seed=7)
    for direction in ("established_to_recent", "recent_to_established"):
        gaps = report["gap_summary"][direction]
        assert abs(gaps["roc_auc"]["transfer_minus_control"]) < 0.08
        assert abs(gaps["average_precision"]["transfer_minus_control"]) < 0.10
    shift = report["covariate_shift"]
    assert shift["max_psi_excluding_tenure"] < 0.2
    assert shift["flagged_features"] in ([], ["tenure"])


def test_genuine_covariate_shift_is_detected():
    X, y = synthetic(n=800, seed=3, shift=True)
    report = evaluate_shift(X, y, estimator_factory=cheap_estimator, seed=7)
    shift = report["covariate_shift"]
    assert shift["max_psi_excluding_tenure"] > 0.2
    assert "MonthlyCharges" in shift["flagged_features"]
    assert shift["features"]["MonthlyCharges"]["review_flag"] is True


def test_deterministic_under_fixed_seed():
    X, y = synthetic(n=300, seed=4)
    first = evaluate_shift(X, y, estimator_factory=cheap_estimator, seed=7)
    second = evaluate_shift(X, y, estimator_factory=cheap_estimator, seed=7)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_rejects_degenerate_cohort():
    X, y = synthetic(n=300, seed=5)
    with pytest.raises(ValueError, match="at least"):
        evaluate_shift(X, y, boundary=200, estimator_factory=cheap_estimator)


def test_chart_is_written(unshifted_report, tmp_path):
    path = create_chart(unshifted_report, tmp_path)
    assert path.exists() and path.name == "cohort_shift.png"
    assert path.stat().st_size > 0


def test_cli_refuses_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["cohort", "--data", str(tmp_path / "absent.csv")])
    with pytest.raises(SystemExit, match="not found"):
        main()
