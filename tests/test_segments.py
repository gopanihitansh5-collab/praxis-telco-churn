import numpy as np
import pandas as pd
import pytest
from telco_churn.segments import (
    SEGMENTS,
    TENURE_EDGES,
    TENURE_LABELS,
    UNKNOWN_LEVEL,
    evaluate_segments,
    markdown_table,
    rank_weakest,
    segment_metrics,
    tenure_band,
)


def make_frame(rows):
    """Synthetic holdout-shaped frame; no CSV and no fitted model involved."""
    return pd.DataFrame(
        [
            {
                "Contract": contract,
                "InternetService": internet,
                "SeniorCitizen": senior,
                "PaymentMethod": payment,
                "tenure": tenure,
            }
            for contract, internet, senior, payment, tenure in rows
        ]
    )


def test_tenure_band_edges_and_missing():
    values = pd.Series([0, 6, 7, 12, 13, 24, 25, 48, 49, 200, np.nan, None])
    bands = tenure_band(values).tolist()
    assert bands == [
        "0-6",
        "0-6",
        "7-12",
        "7-12",
        "13-24",
        "13-24",
        "25-48",
        "25-48",
        "49+",
        "49+",
        UNKNOWN_LEVEL,
        UNKNOWN_LEVEL,
    ]
    assert TENURE_EDGES == [6, 12, 24, 48]
    assert len(TENURE_LABELS) == len(TENURE_EDGES) + 1
    assert tenure_band(pd.Series([], dtype=float)).tolist() == []


def test_tenure_band_non_numeric_is_unknown():
    assert tenure_band(pd.Series([""])).tolist() == [UNKNOWN_LEVEL]


def test_single_class_segment_returns_none_with_reason():
    metrics = segment_metrics(np.zeros(40), np.full(40, 0.9))
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["single_class"] is True
    assert "undefined" in metrics["reason"]
    assert metrics["churn_rate"] == 0.0
    # Recall and lift are undefined without positives; they must not read as 0.0.
    assert metrics["recall"] is None
    assert metrics["lift"] is None
    assert metrics["precision"] == 0.0


def test_single_class_all_positive_does_not_raise():
    metrics = segment_metrics(np.ones(40), np.full(40, 0.9))
    assert metrics["average_precision"] is None
    assert metrics["recall"] == 1.0
    assert metrics["reason"] is not None


def test_nothing_flagged_leaves_precision_undefined():
    metrics = segment_metrics([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert metrics["precision"] is None
    assert metrics["lift"] is None
    assert metrics["recall"] == 0.0
    assert metrics["average_precision"] is not None
    # The reason names F1 undefined, so F1 must not read as a measured 0.0.
    assert metrics["f1"] is None
    assert "F1" in metrics["reason"]


def test_f1_is_zero_when_flagged_rows_are_all_wrong():
    # Flagged but never right is a measurement, not an undefined metric.
    metrics = segment_metrics([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["reason"] is None


def test_empty_segment_is_reported_not_crashed():
    metrics = segment_metrics([], [])
    assert metrics["n"] == 0
    assert metrics["churn_rate"] is None
    assert metrics["f1"] is None
    assert metrics["reason"] == "empty segment: no metrics defined"


def test_lift_hand_worked():
    # 10 rows, 4 churners. Five rows flagged, three of them churners.
    y = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    prob = [0.9, 0.8, 0.7, 0.6, 0.51, 0.2, 0.2, 0.2, 0.2, 0.2]
    metrics = segment_metrics(y, prob)
    assert metrics["churn_rate"] == pytest.approx(0.4)
    assert metrics["precision"] == pytest.approx(0.6)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["lift"] == pytest.approx(1.5)


def test_uninformative_model_lift_is_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    metrics = segment_metrics(y, np.full(400, 0.6))
    # A constant score above the threshold flags everyone, so precision is the
    # base rate and lift collapses to exactly 1.
    assert metrics["lift"] == pytest.approx(1.0)
    assert metrics["recall"] == 1.0


def test_insufficient_rows_flagged_not_dropped():
    rows = [("Two year", "DSL", 0, "Mailed check", 60)] * 5
    rows += [("Month-to-month", "Fiber optic", 1, "Electronic check", 3)] * 40
    X = make_frame(rows)
    y = [0, 1, 0, 1, 0] + [1, 0] * 20
    prob = [0.2, 0.8, 0.3, 0.7, 0.1] + [0.8, 0.2] * 20
    report = evaluate_segments(X, y, prob, min_rows=30)
    tiny = report["segmentations"]["Contract"]["Two year"]
    assert tiny["n"] == 5
    assert tiny["insufficient_rows"] is True
    assert (
        report["segmentations"]["Contract"]["Month-to-month"]["insufficient_rows"]
        is False
    )
    # Flagged segments stay addressable in the report.
    assert set(report["segmentations"]["Contract"]) == {"Two year", "Month-to-month"}
    assert report["min_rows"] == 30
    assert report["holdout_used"] is True


def test_every_level_carries_its_n():
    rows = [("Month-to-month", "DSL", 0, "Mailed check", i % 70) for i in range(60)]
    X = make_frame(rows)
    y = [i % 3 == 0 for i in range(60)]
    report = evaluate_segments(X, y, np.linspace(0, 1, 60))
    assert set(SEGMENTS) == set(report["segmentations"])
    for levels in report["segmentations"].values():
        assert levels
        for metrics in levels.values():
            assert isinstance(metrics["n"], int)
        assert sum(m["n"] for m in levels.values()) == 60


def test_perfect_model_scores_one_everywhere():
    rows = []
    for contract in ("Month-to-month", "One year", "Two year"):
        rows += [(contract, "DSL", 0, "Mailed check", 5)] * 20
        rows += [(contract, "Fiber optic", 1, "Credit card", 60)] * 20
    X = make_frame(rows)
    y = np.array([i % 2 for i in range(len(rows))])
    report = evaluate_segments(X, y, y.astype(float), min_rows=30)
    for levels in report["segmentations"].values():
        for metrics in levels.values():
            if metrics["insufficient_rows"] or metrics["single_class"]:
                continue
            assert metrics["average_precision"] == pytest.approx(1.0)
            assert metrics["recall"] == pytest.approx(1.0)
            assert metrics["precision"] == pytest.approx(1.0)
            assert metrics["roc_auc"] == pytest.approx(1.0)


def build_mixed_report():
    rows = [("Month-to-month", "Fiber optic", 0, "Electronic check", 2)] * 40
    rows += [("One year", "DSL", 0, "Mailed check", 30)] * 40
    rows += [("Two year", "No", 1, "Credit card", 70)] * 40
    rows += [("Two year", "No", 1, "Bank transfer", 70)] * 4
    X = make_frame(rows)
    # Strong ranking in the first block, inverted in the second, constant in the
    # third (single class), and a tiny fourth block.
    y = [1, 0] * 20 + [1, 0] * 20 + [0] * 40 + [1, 0] * 2
    prob = (
        list(np.linspace(0.95, 0.55, 40))
        + [0.4, 0.9] * 20
        + [0.3] * 40
        + [0.8, 0.2, 0.7, 0.1]
    )
    return evaluate_segments(X, y, prob, min_rows=30)


def test_rank_weakest_ordering_and_exclusions():
    report = build_mixed_report()
    ranked = rank_weakest(report)
    values = [row["average_precision"] for row in ranked]
    assert values == sorted(values)
    flagged = {
        (name, level)
        for name, levels in report["segmentations"].items()
        for level, metrics in levels.items()
        if metrics["insufficient_rows"] or metrics["single_class"]
    }
    assert ("PaymentMethod", "Bank transfer") in flagged  # 4 rows
    assert ("PaymentMethod", "Credit card") in flagged  # no churn events
    assert flagged.isdisjoint({(r["segmentation"], r["level"]) for r in ranked})
    assert all(row["n"] >= report["min_rows"] for row in ranked)
    assert ranked == report["weakest_segments"]
    # The inverted-ranking segment must rank below the strong one.
    by_level = {row["level"]: row["average_precision"] for row in ranked}
    assert by_level["One year"] < by_level["Month-to-month"]


def test_rank_weakest_honours_metric_argument():
    report = build_mixed_report()
    ranked = rank_weakest(report, metric="roc_auc")
    assert ranked
    assert all("roc_auc" in row for row in ranked)
    assert [row["roc_auc"] for row in ranked] == sorted(
        row["roc_auc"] for row in ranked
    )


def test_determinism_on_repeated_calls():
    report_a = build_mixed_report()
    report_b = build_mixed_report()
    assert report_a == report_b
    assert markdown_table(report_a) == markdown_table(report_b)
    assert rank_weakest(report_a) == rank_weakest(report_b)


def test_markdown_table_is_weakest_first_with_sample_sizes():
    report = build_mixed_report()
    table = markdown_table(report)
    body = [line for line in table.splitlines() if line.startswith("| Contract |")]
    assert body
    weakest = report["weakest_segments"][0]
    first = [
        line
        for line in table.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ][1]
    assert weakest["level"] in first
    assert f"| {weakest['n']} |" in first
    assert "n/a" in table  # Undefined metrics are shown, never zero-filled.
    assert "not a fairness audit" in table


def test_markdown_and_json_agree_on_undefined_f1():
    rows = [("One year", "DSL", 0, "Mailed check", 30)] * 40
    X = make_frame(rows)
    report = evaluate_segments(X, [1, 0] * 20, np.full(40, 0.2), min_rows=30)
    metrics = report["segmentations"]["Contract"]["One year"]
    assert metrics["flagged"] == 0
    assert metrics["f1"] is None
    table = markdown_table(report)
    row = next(line for line in table.splitlines() if line.startswith("| Contract |"))
    # Precision, F1 and lift are undefined here; recall is a measured 0.000.
    assert row.count("n/a") == 3
    assert "| n/a | 0.000 | n/a | n/a |" in row
    assert metrics["reason"] in table


def test_index_misalignment_does_not_shuffle_labels():
    rows = [("Month-to-month", "DSL", 0, "Mailed check", 5)] * 40
    X = make_frame(rows)
    X.index = range(100, 140)
    y = pd.Series([1, 0] * 20, index=range(500, 540))
    report = evaluate_segments(X, y, np.linspace(0, 1, 40))
    assert report["segmentations"]["Contract"]["Month-to-month"]["n"] == 40
    assert report["overall"]["churn_count"] == 20


def test_length_mismatch_raises():
    X = make_frame([("Month-to-month", "DSL", 0, "Mailed check", 5)] * 3)
    with pytest.raises(ValueError):
        evaluate_segments(X, [1, 0], [0.5, 0.5])


def test_missing_segmentation_column_raises():
    X = make_frame([("Month-to-month", "DSL", 0, "Mailed check", 5)] * 3)
    with pytest.raises(ValueError, match="Missing segmentation columns"):
        evaluate_segments(X.drop(columns=["Contract"]), [1, 0, 1], [0.5, 0.5, 0.5])


def test_missing_categorical_level_is_named_not_dropped():
    rows = [("Month-to-month", "DSL", 0, "Mailed check", 5)] * 3
    X = make_frame(rows)
    X.loc[0, "Contract"] = np.nan
    report = evaluate_segments(X, [1, 0, 1], [0.6, 0.2, 0.8], min_rows=1)
    assert UNKNOWN_LEVEL in report["segmentations"]["Contract"]
    assert report["segmentations"]["Contract"][UNKNOWN_LEVEL]["n"] == 1


def test_report_states_edges_and_caveats():
    report = build_mixed_report()
    assert report["tenure_band_edges_months"]["inclusive_upper_bounds"] == TENURE_EDGES
    assert report["tenure_band_edges_months"]["labels"] == TENURE_LABELS
    assert report["default_model_changed"] is False
    joined = " ".join(report["caveats"]).lower()
    assert "not a fairness audit" in joined
    assert "retuning on this holdout" in joined
    assert "high-variance" in joined
