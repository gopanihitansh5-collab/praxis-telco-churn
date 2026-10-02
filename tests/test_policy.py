import json
import numpy as np
import pytest
from telco_churn.policy import (
    DEFAULTS,
    THRESHOLDS,
    Economics,
    break_even_probability,
    customer_expected_value,
    optimal_policy,
    policy_curve,
    retained_values,
    sensitivity_grid,
)

WORKED = Economics(
    monthly_margin_rate=0.5,
    horizon_months=10,
    contact_cost=3.0,
    offer_cost=10.0,
    acceptance_rate=0.5,
    uplift=0.4,
)
FLAT = Economics(
    monthly_margin_rate=0.5,
    horizon_months=10,
    contact_cost=30.0,
    offer_cost=40.0,
    acceptance_rate=0.5,
    uplift=0.5,
)


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.01, 0.99, n)
    charges = rng.uniform(20.0, 120.0, n)
    labels = (rng.uniform(size=n) < prob).astype(int)
    return prob, charges, labels


def test_break_even_matches_hand_computation():
    # M = 0.5 * 20 * 10 = 100; cost = 3 + 0.5 * 10 = 8; p* = 8 / (0.4 * 100) = 0.2.
    assert retained_values([20.0], WORKED).tolist() == [100.0]
    assert break_even_probability(WORKED, 100.0) == pytest.approx(0.2)
    assert customer_expected_value([0.2], [20.0], WORKED)[0] == pytest.approx(0.0)
    assert customer_expected_value([0.4], [20.0], WORKED)[0] == pytest.approx(8.0)


def test_break_even_infinite_without_uplift_or_value():
    zero_uplift = Economics(0.3, 12, 2.0, 60.0, 0.5, 0.0)
    assert break_even_probability(zero_uplift, 500.0) == float("inf")
    assert break_even_probability(DEFAULTS, 0.0) == float("inf")
    prob, charges, labels = synthetic()
    report = optimal_policy(prob, charges, labels, zero_uplift)
    assert report["expected_value"] == 0.0
    assert report["recommended_action"] == "contact_nobody"


def test_optimal_threshold_rises_with_contact_cost():
    prob, charges, labels = synthetic()
    thresholds = [
        optimal_policy(
            prob, charges, labels, Economics(0.3, 12, cost, 60.0, 0.5, 0.25)
        )["optimal"]["threshold"]
        for cost in (0.0, 2.0, 10.0, 50.0, 200.0)
    ]
    assert thresholds == sorted(thresholds)


def test_optimal_threshold_falls_with_uplift():
    prob, charges, labels = synthetic()
    thresholds = [
        optimal_policy(
            prob, charges, labels, Economics(0.3, 12, 2.0, 60.0, 0.5, uplift)
        )["optimal"]["threshold"]
        for uplift in (0.05, 0.25, 0.5, 0.9, 1.0)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_negative_value_customers_are_not_contacted():
    # Flat charges make EV monotone in probability, so the threshold rule is exact.
    prob = np.round(np.linspace(0.05, 0.95, 19), 2)
    charges = np.full(19, 100.0)
    labels = (prob > 0.5).astype(int)
    expected = customer_expected_value(prob, charges, FLAT)
    report = optimal_policy(prob, charges, labels, FLAT)
    threshold = report["optimal"]["threshold"]
    assert break_even_probability(FLAT, 500.0) == pytest.approx(0.2)
    assert (expected < 0).any()
    assert threshold > 0.15
    assert report["optimal"]["negative_ev_contacts"] == 0
    assert report["optimal"]["contacted"] == int((expected >= 0).sum())
    assert (expected[prob >= threshold] < 0).sum() == 0


def test_per_customer_rule_dominates_threshold_rule():
    prob, charges, labels = synthetic()
    report = optimal_policy(prob, charges, labels, DEFAULTS)
    rule = report["baselines"]["per_customer_ev_rule"]
    assert rule["negative_ev_contacts"] == 0
    assert rule["expected_value"] >= report["optimal"]["expected_value"] - 1e-9


@pytest.mark.parametrize(
    "field,value",
    [
        ("monthly_margin_rate", -0.01),
        ("monthly_margin_rate", 1.01),
        ("monthly_margin_rate", float("nan")),
        ("monthly_margin_rate", "0.3"),
        ("acceptance_rate", 1.5),
        ("acceptance_rate", float("inf")),
        ("uplift", -1.0),
        ("uplift", True),
        ("contact_cost", -1.0),
        ("contact_cost", float("nan")),
        ("offer_cost", -0.5),
        ("offer_cost", None),
        ("horizon_months", 0),
        ("horizon_months", -3),
        ("horizon_months", 12.5),
        ("horizon_months", True),
    ],
)
def test_economics_rejects_bad_inputs(field, value):
    fields = {
        "monthly_margin_rate": 0.3,
        "horizon_months": 12,
        "contact_cost": 2.0,
        "offer_cost": 60.0,
        "acceptance_rate": 0.5,
        "uplift": 0.25,
    }
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        Economics(**fields)


def test_economics_accepts_boundary_values():
    assert Economics(0.0, 1, 0.0, 0.0, 0.0, 0.0).uplift == 0.0
    assert Economics(1.0, 120, 0.0, 0.0, 1.0, 1.0).horizon_months == 120


def test_contact_nobody_is_zero_and_never_beats_the_optimum():
    for economics in (DEFAULTS, Economics(0.3, 12, 500.0, 900.0, 0.9, 0.05)):
        prob, charges, labels = synthetic()
        report = optimal_policy(prob, charges, labels, economics)
        nobody = report["baselines"]["contact_nobody"]
        assert nobody["expected_value"] == 0.0
        assert nobody["realised_expected_value"] == 0.0
        assert nobody["contacted"] == 0
        assert report["expected_value"] >= nobody["expected_value"]


def test_curve_shape_and_honesty_keys():
    prob, charges, labels = synthetic()
    curve = policy_curve(prob, charges, labels, DEFAULTS)
    assert len(curve) == len(THRESHOLDS) == 99
    assert [row["threshold"] for row in curve] == sorted(
        row["threshold"] for row in curve
    )
    assert all(0.0 <= row["contact_rate"] <= 1.0 for row in curve)
    report = optimal_policy(prob, charges, labels, DEFAULTS, budget=25)
    assert report["holdout_used"] is False
    assert report["default_model_changed"] is False
    assert report["baselines"]["budget_top_n"]["contacted"] == 25
    assert report["baselines"]["contact_everybody"]["contacted"] == len(prob)
    assert report["baselines"]["served_threshold"]["threshold"] == 0.5
    assert isinstance(report["caveats"], list) and report["caveats"]


def test_sensitivity_grid_shape_and_dead_zone():
    prob, charges, labels = synthetic()
    uplifts, offer_costs = (0.1, 0.3, 0.5), (0.0, 50.0, 1e6, 1e7)
    grid = sensitivity_grid(prob, charges, labels, DEFAULTS, uplifts, offer_costs)
    assert len(grid) == len(uplifts) * len(offer_costs)
    assert {(cell["uplift"], cell["offer_cost"]) for cell in grid} == {
        (u, d) for u in uplifts for d in offer_costs
    }
    assert any(cell["value_positive"] for cell in grid)
    assert all(
        cell["value_positive"] is False for cell in grid if cell["offer_cost"] >= 1e6
    )


def test_identical_inputs_give_identical_output():
    prob, charges, labels = synthetic()
    first = optimal_policy(prob, charges, labels, DEFAULTS, budget=30)
    second = optimal_policy(prob, charges, labels, DEFAULTS, budget=30)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert sensitivity_grid(prob, charges, labels, DEFAULTS) == sensitivity_grid(
        prob, charges, labels, DEFAULTS
    )


def test_empty_sample_does_not_divide_by_zero():
    report = optimal_policy([], [], [], DEFAULTS, budget=10)
    assert report["rows"] == 0
    assert report["expected_value"] == 0.0
    assert report["optimal"]["contact_rate"] == 0.0
    assert report["mean_retained_value"] == 0.0
    assert report["break_even_probability_at_mean_retained_value"] == float("inf")
    assert policy_curve([], [], [], DEFAULTS)[0]["f1"] == 0.0


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_labels_are_handled(label):
    prob, charges, _ = synthetic(50)
    labels = np.full(50, label)
    report = optimal_policy(prob, charges, labels, DEFAULTS)
    row = report["optimal"]
    assert 0.0 <= row["precision"] <= 1.0
    assert 0.0 <= row["recall"] <= 1.0
    if label == 0:
        assert row["realised_expected_value"] <= 0.0


def test_missing_monthly_charges_carry_no_value():
    prob = np.array([0.9, 0.9, 0.9])
    charges = np.array([100.0, np.nan, 100.0])
    labels = np.array([1, 1, 1])
    expected = customer_expected_value(prob, charges, DEFAULTS)
    assert retained_values(charges, DEFAULTS)[1] == 0.0
    assert expected[1] < 0.0
    report = optimal_policy(prob, charges, labels, DEFAULTS)
    assert report["missing_monthly_charges"] == 1
    # Identical probabilities: no threshold can exclude the valueless customer, but the
    # per-customer rule does. This is the heterogeneity limitation stated in the caveats.
    assert report["optimal"]["negative_ev_contacts"] == 1
    rule = report["baselines"]["per_customer_ev_rule"]
    assert rule["contacted"] == 2
    assert rule["negative_ev_contacts"] == 0


def test_mismatched_lengths_and_bad_labels_raise():
    with pytest.raises(ValueError, match="same shape"):
        customer_expected_value([0.5, 0.5], [50.0], DEFAULTS)
    with pytest.raises(ValueError, match="same length"):
        optimal_policy([0.5, 0.5], [50.0, 50.0], [1], DEFAULTS)
    with pytest.raises(ValueError, match="0/1"):
        policy_curve([0.5], [50.0], [2], DEFAULTS)


def test_yes_no_labels_are_accepted():
    prob, charges, labels = synthetic(40)
    words = np.where(labels == 1, "Yes", "No")
    assert policy_curve(prob, charges, labels, DEFAULTS) == policy_curve(
        prob, charges, words, DEFAULTS
    )
