import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from telco_churn.paths import project_root
from telco_churn.policy import DEFAULTS, Economics
from telco_churn.score_batch import apply_budget, main, relative_path, score, summarize


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


def run_main(tmp_path, *extra):
    """Drive the CLI end to end; the summary is the only artefact the assertions read."""
    summary = tmp_path / "summary.json"
    argv = [
        "score_batch",
        "--output",
        str(tmp_path / "worklist.csv"),
        "--summary",
        str(summary),
        *extra,
    ]
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(sys, "argv", argv)
        main()
    finally:
        monkey.undo()
    return json.loads(summary.read_text())


def external_csv(tmp_path, customer):
    source = tmp_path / "customers.csv"
    frame_of(customer, [110.0, 90.0, 20.0]).to_csv(source, index=False)
    return source


def test_external_input_is_not_reported_in_sample(tmp_path, customer):
    # An external file's overlap with the training rows is unknowable, so neither the
    # flag nor the caveats may assert that its rows were trained on.
    report = run_main(tmp_path, "--input", str(external_csv(tmp_path, customer)))
    assert report["rows_scored"] == 3
    assert report["scored_in_sample"] is False
    assert not any("in-sample" in caveat for caveat in report["caveats"])
    assert not any("--partition holdout" in caveat for caveat in report["caveats"])


def test_partition_with_external_input_is_rejected_accurately(tmp_path, customer):
    with pytest.raises(SystemExit) as error:
        run_main(
            tmp_path,
            "--input",
            str(external_csv(tmp_path, customer)),
            "--partition",
            "train",
        )
    message = str(error.value)
    assert "--partition" in message
    # The old message advised the very flag this guard rejects.
    assert "use --partition holdout" not in message.lower()


def test_recorded_paths_are_relative(tmp_path, customer):
    report = run_main(tmp_path, "--input", str(external_csv(tmp_path, customer)))
    for key in ("input", "model"):
        assert not Path(report[key]).is_absolute()
        assert "\\" not in report[key]
        assert Path.home().name not in report[key]
    assert report["model"] == "artifacts/model_calibrated.joblib"
    # The input lies outside the project root, so only its basename can be recorded.
    assert report["input"] == "customers.csv"


def test_relative_path_keeps_project_files_inside_the_root():
    assert relative_path(project_root() / "data" / "telco.csv") == "data/telco.csv"
    assert relative_path(Path.home() / "elsewhere" / "telco.csv") == "telco.csv"
