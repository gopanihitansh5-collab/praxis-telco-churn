import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from telco_churn.policy import DEFAULTS, Economics
from telco_churn.score_batch import apply_budget, score, summarize


class StubModel:
    """Returns a fixed probability per row so ranking behaviour is testable exactly."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        p = self.probabilities[: len(frame)]
        return np.column_stack([1.0 - p, p])


@pytest.fixture
def customer():
    return json.loads(Path("sample_customer.json").read_text())


def frame_of(customer, charges):
    rows = []
    for i, value in enumerate(charges):
        row = dict(customer)
        row["customerID"] = f"id-{i}"
        row["MonthlyCharges"] = value
        row["TotalCharges"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_ranks_by_expected_value_not_probability(customer):
    # Lower risk but far higher margin must outrank higher risk on a small account.
    frame = frame_of(customer, [20.0, 110.0])
    ranked = score(frame, StubModel([0.60, 0.45]), DEFAULTS)
    assert list(ranked.customerID) == ["id-1", "id-0"]
    assert ranked.churn_probability.iloc[0] < ranked.churn_probability.iloc[1]


def test_negative_expected_value_is_not_contacted(customer):
    frame = frame_of(customer, [20.0])
    ranked = score(frame, StubModel([0.05]), DEFAULTS)
    assert ranked.expected_value.iloc[0] < 0
    assert ranked.action.iloc[0] == "no_contact"
    assert ranked.priority.iloc[0] == ""


def test_break_even_probability_is_per_customer(customer):
    frame = frame_of(customer, [30.0, 120.0])
    ranked = score(frame, StubModel([0.5, 0.5]), DEFAULTS)
    cheap = ranked[ranked.monthly_charges == 30.0].break_even_probability.iloc[0]
    rich = ranked[ranked.monthly_charges == 120.0].break_even_probability.iloc[0]
    assert cheap > rich


def test_priority_bands_only_cover_contacts(customer):
    frame = frame_of(customer, [110.0, 100.0, 90.0, 80.0, 15.0])
    ranked = score(frame, StubModel([0.9, 0.8, 0.7, 0.6, 0.02]), DEFAULTS)
    contacted = ranked[ranked.action == "contact"]
    assert (contacted.priority != "").all()
    assert (ranked[ranked.action == "no_contact"].priority == "").all()


def test_budget_suppresses_without_promoting(customer):
    frame = frame_of(customer, [110.0, 100.0, 90.0, 15.0])
    ranked = score(frame, StubModel([0.9, 0.85, 0.8, 0.01]), DEFAULTS)
    limited = apply_budget(ranked, 2)
    assert list(limited.action) == ["contact", "contact", "over_budget", "no_contact"]
    # A budget must never turn a value-destroying customer into a contact.
    assert "contact" not in list(limited.action[limited.expected_value <= 0])
    assert (limited.action == "contact").sum() <= 2


def test_budget_zero_and_none(customer):
    frame = frame_of(customer, [110.0, 100.0])
    ranked = score(frame, StubModel([0.9, 0.8]), DEFAULTS)
    assert (apply_budget(ranked, 0).action == "contact").sum() == 0
    assert apply_budget(ranked, None).equals(ranked)
    with pytest.raises(ValueError, match="nonnegative"):
        apply_budget(ranked, -1)


def test_missing_feature_raises(customer):
    frame = frame_of(customer, [50.0]).drop(columns=["tenure"])
    with pytest.raises(ValueError, match="Missing required fields"):
        score(frame, StubModel([0.5]), DEFAULTS)


def test_invalid_input_is_rejected_by_cleaning(customer):
    frame = frame_of(customer, [50.0])
    frame.loc[0, "tenure"] = -1
    with pytest.raises(ValueError):
        score(frame, StubModel([0.5]), DEFAULTS)


def test_zero_uplift_contacts_nobody(customer):
    economics = Economics(
        monthly_margin_rate=0.3,
        horizon_months=12,
        contact_cost=2.0,
        offer_cost=60.0,
        acceptance_rate=0.5,
        uplift=0.0,
    )
    frame = frame_of(customer, [110.0, 100.0])
    ranked = score(frame, StubModel([0.99, 0.95]), economics)
    assert (ranked.action == "no_contact").all()
    assert np.isinf(ranked.break_even_probability).all()


def test_summary_flags_in_sample(customer):
    frame = frame_of(customer, [110.0])
    ranked = score(frame, StubModel([0.9]), DEFAULTS)
    flagged = summarize(ranked, DEFAULTS, None, in_sample=True)
    clean = summarize(ranked, DEFAULTS, None, in_sample=False)
    assert flagged["scored_in_sample"] is True
    assert any("in-sample" in c for c in flagged["caveats"])
    assert clean["scored_in_sample"] is False
    assert not any("in-sample" in c for c in clean["caveats"])


def test_summary_totals_match_contacts(customer):
    frame = frame_of(customer, [110.0, 100.0, 15.0])
    ranked = score(frame, StubModel([0.9, 0.8, 0.01]), DEFAULTS)
    report = summarize(ranked, DEFAULTS, None)
    contacted = ranked[ranked.action == "contact"]
    assert report["contacts_recommended"] == len(contacted)
    assert report["total_expected_value"] == pytest.approx(
        contacted.expected_value.sum()
    )
    assert report["cost_per_contact"] == pytest.approx(32.0)


def test_deterministic(customer):
    frame = frame_of(customer, [110.0, 90.0, 40.0])
    a = score(frame, StubModel([0.7, 0.5, 0.3]), DEFAULTS)
    b = score(frame, StubModel([0.7, 0.5, 0.3]), DEFAULTS)
    assert a.equals(b)
