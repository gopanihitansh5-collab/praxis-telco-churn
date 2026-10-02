"""Expected-value retention policy; its costs are illustrative inputs, not measured results."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

SERVED_THRESHOLD = 0.5
THRESHOLDS = np.round(np.arange(1, 100) / 100.0, 2)
REQUIRED_COLUMNS = ("probability", "MonthlyCharges", "churn")
MISSING_INPUT = (
    "--probabilities is required and must point to out-of-fold model scores.\n"
    "No such file is bundled and this module never trains or invents data.\n"
    "Produce the scores with a training-only study, for example\n"
    "  python -m telco_churn.calibration_experiment\n"
    "  python -m telco_churn.threshold\n"
    "then export their out-of-fold scores as CSV or JSON records with columns\n"
    "probability, MonthlyCharges, churn (0/1 or No/Yes)."
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Economics:
    """Retention cost/value assumptions. Every field is an input, never an estimate."""

    monthly_margin_rate: float
    horizon_months: int
    contact_cost: float
    offer_cost: float
    acceptance_rate: float
    uplift: float

    def __post_init__(self) -> None:
        for name in ("monthly_margin_rate", "acceptance_rate", "uplift"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in ("contact_cost", "offer_cost"):
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if isinstance(self.horizon_months, bool) or not isinstance(
            self.horizon_months, Integral
        ):
            raise ValueError("horizon_months must be a positive int")
        if int(self.horizon_months) < 1:
            raise ValueError("horizon_months must be a positive int")


# Illustrative planning benchmarks, not values measured from this dataset. They exist so
# the sensitivity grid has a centre; no figure derived from them is a forecast.
DEFAULTS = Economics(
    # Telecom gross margin is commonly quoted near 30% of ARPU. A save protects margin,
    # not revenue, so charges are scaled down before being called value.
    monthly_margin_rate=0.30,
    # One year: long enough for a save to matter, short enough to avoid discounting and
    # repeat-churn assumptions this dataset cannot support.
    horizon_months=12,
    # Outbound contact: a couple of dollars of dialler and agent time per customer reached.
    contact_cost=2.0,
    # A retention offer of roughly one month of charges; 60 is near this dataset's mean
    # MonthlyCharges. Modelled as a flat voucher, so it does not scale with the customer.
    offer_cost=60.0,
    # Share of contacted customers who take the offer. A round planning placeholder with
    # no published basis; it is one of the inputs the grid is meant to stress.
    acceptance_rate=0.50,
    # Published retention-campaign save rates commonly fall in the 20-35% band.
    uplift=0.25,
)
SENSITIVITY_UPLIFTS = (0.05, 0.15, 0.25, 0.35, 0.50)
SENSITIVITY_OFFER_COSTS = (0.0, 30.0, 60.0, 120.0, 240.0)
CAVEATS = [
    "monthly_margin_rate, contact_cost, offer_cost, acceptance_rate and uplift are "
    "illustrative planning assumptions, not values measured from this dataset.",
    "No currency figure here is a forecast or a claimed ROI. The sensitivity grid, "
    "not any single expected value, is the result of this module.",
    "Planned expected value is computed from model scores only. The calibration study "
    "shows those scores are not calibrated probabilities, which biases every money figure.",
    "realised_expected_value reuses the labels of the same sample and is a retrospective "
    "diagnostic, not an out-of-sample estimate of programme value.",
    "Cells with value_positive false are regions where the programme earns nothing: at low "
    "uplift or high offer cost no threshold beats contacting nobody.",
    "A probability threshold can still contact a low-margin customer whose own expected "
    "value is negative; per_customer_ev_rule is the only reported policy that never does.",
    "Uplift is assumed identical for every customer. Real uplift varies and cannot be "
    "estimated without a randomised holdback, which this dataset does not contain.",
    "Customers with missing MonthlyCharges carry zero retained value by construction and "
    "are therefore never contacted.",
]


def contact_cost_per_customer(economics: Economics) -> float:
    """Cost incurred per contact: c + alpha * d, paid whether or not the customer stays."""
    return economics.contact_cost + economics.acceptance_rate * economics.offer_cost


def retained_values(monthly_charges: object, economics: Economics) -> np.ndarray:
    """Per-customer M_i = margin_rate * MonthlyCharges_i * horizon_months."""
    charges = np.asarray(monthly_charges, dtype=float)
    # Unknown or nonfinite charges have no defensible value; zero keeps EV strictly negative.
    charges = np.where(np.isfinite(charges), charges, 0.0)
    return charges * economics.monthly_margin_rate * float(economics.horizon_months)


def break_even_probability(economics: Economics, retained_value: float) -> float:
    """Closed form p* = (c + alpha * d) / (u * M); inf when no probability can clear it."""
    denominator = economics.uplift * _finite(retained_value, "retained_value")
    if denominator <= 0.0:
        return float("inf")
    return contact_cost_per_customer(economics) / denominator


def customer_expected_value(
    prob: object, monthly_charges: object, economics: Economics
) -> np.ndarray:
    """EV_i = p_i * u * M_i - c - alpha * d, the value of contacting customer i."""
    probabilities = np.asarray(prob, dtype=float)
    values = retained_values(monthly_charges, economics)
    if probabilities.shape != values.shape:
        raise ValueError("prob and monthly_charges must have the same shape")
    return probabilities * economics.uplift * values - contact_cost_per_customer(
        economics
    )


def _binary_labels(y_true: object) -> np.ndarray:
    series = pd.Series(np.asarray(y_true).ravel())
    if series.dtype == object:
        series = series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    labels = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if len(labels) and not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only 0/1 or No/Yes labels")
    return labels.astype(int)


def _label_metrics(labels: np.ndarray, contacted: np.ndarray) -> dict:
    if not len(labels):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    predicted = contacted.astype(int)
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def _summarize(
    contacted: np.ndarray,
    expected: np.ndarray,
    realised: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
) -> dict:
    count = int(contacted.sum())
    total = len(contacted)
    return {
        "threshold": threshold,
        "contacted": count,
        "contact_rate": float(count / total) if total else 0.0,
        "expected_value": float(expected[contacted].sum()),
        "realised_expected_value": float(realised[contacted].sum()),
        "negative_ev_contacts": int((expected[contacted] < 0).sum()),
        **_label_metrics(labels, contacted),
    }


def _prepare(
    prob: object, monthly_charges: object, y_true: object, economics: Economics
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(prob, dtype=float).ravel()
    labels = _binary_labels(y_true)
    values = retained_values(
        np.asarray(monthly_charges, dtype=float).ravel(), economics
    )
    if not probabilities.shape == labels.shape == values.shape:
        raise ValueError("prob, monthly_charges and y_true must have the same length")
    cost = contact_cost_per_customer(economics)
    expected = probabilities * economics.uplift * values - cost
    realised = labels * economics.uplift * values - cost
    return probabilities, labels, expected, realised


def _best_threshold_index(probabilities: np.ndarray, expected: np.ndarray) -> int:
    totals = [float(expected[probabilities >= t].sum()) for t in THRESHOLDS]
    # First argmax keeps the smallest optimal threshold, which preserves monotonicity
    # of the optimum in contact cost and uplift under ties.
    return int(np.argmax(totals))


def policy_curve(
    prob: object, monthly_charges: object, y_true: object, economics: Economics
) -> list[dict]:
    """Threshold sweep 0.01..0.99 with planned EV, realised EV and label diagnostics."""
    probabilities, labels, expected, realised = _prepare(
        prob, monthly_charges, y_true, economics
    )
    return [
        _summarize(
            probabilities >= threshold, expected, realised, labels, float(threshold)
        )
        for threshold in THRESHOLDS
    ]


def optimal_policy(
    prob: object,
    monthly_charges: object,
    y_true: object,
    economics: Economics = DEFAULTS,
    budget: int | None = None,
) -> dict:
    """Best swept threshold by planned EV, against no-contact, all-contact, 0.5 and budget."""
    probabilities, labels, expected, realised = _prepare(
        prob, monthly_charges, y_true, economics
    )
    curve = policy_curve(prob, monthly_charges, y_true, economics)
    best = curve[_best_threshold_index(probabilities, expected)]
    values = retained_values(
        np.asarray(monthly_charges, dtype=float).ravel(), economics
    )
    mean_value = float(values.mean()) if len(values) else 0.0

    def mask_for(count: int) -> np.ndarray:
        order = np.argsort(-probabilities, kind="stable")
        mask = np.zeros(len(probabilities), dtype=bool)
        mask[order[: max(count, 0)]] = True
        return mask

    served = next(row for row in curve if row["threshold"] == SERVED_THRESHOLD)
    baselines = {
        "contact_nobody": _summarize(
            np.zeros(len(probabilities), dtype=bool), expected, realised, labels, None
        ),
        "contact_everybody": _summarize(
            np.ones(len(probabilities), dtype=bool), expected, realised, labels, 0.0
        ),
        "served_threshold": served,
        "per_customer_ev_rule": _summarize(
            expected > 0, expected, realised, labels, None
        ),
    }
    if budget is not None:
        baselines["budget_top_n"] = {
            **_summarize(mask_for(int(budget)), expected, realised, labels, None),
            "budget": int(budget),
        }
    # contact_nobody is exactly 0.0 by definition, so the recommendation never loses money.
    recommended = max(best["expected_value"], 0.0)
    return {
        "rows": int(len(probabilities)),
        "holdout_used": False,
        "default_model_changed": False,
        "economics": asdict(economics),
        "cost_per_contact": contact_cost_per_customer(economics),
        "mean_retained_value": mean_value,
        "break_even_probability_at_mean_retained_value": break_even_probability(
            economics, mean_value
        ),
        "missing_monthly_charges": int(
            (~np.isfinite(np.asarray(monthly_charges, dtype=float).ravel())).sum()
        ),
        "optimal": best,
        "expected_value": recommended,
        "recommended_action": (
            "contact_selected" if best["expected_value"] > 0.0 else "contact_nobody"
        ),
        "expected_value_vs_served_threshold": recommended - served["expected_value"],
        "baselines": baselines,
        "curve": curve,
        "caveats": CAVEATS,
    }


def sensitivity_grid(
    prob: object,
    monthly_charges: object,
    y_true: object,
    economics: Economics = DEFAULTS,
    uplifts: tuple = SENSITIVITY_UPLIFTS,
    offer_costs: tuple = SENSITIVITY_OFFER_COSTS,
) -> list[dict]:
    """Uplift x offer_cost grid showing where no threshold beats contacting nobody."""
    cells = []
    for uplift in uplifts:
        for offer_cost in offer_costs:
            scenario = replace(
                economics, uplift=float(uplift), offer_cost=float(offer_cost)
            )
            probabilities, _, expected, _ = _prepare(
                prob, monthly_charges, y_true, scenario
            )
            index = _best_threshold_index(probabilities, expected)
            contacted = probabilities >= THRESHOLDS[index]
            count = int(contacted.sum())
            total = float(expected[contacted].sum())
            cells.append(
                {
                    "uplift": scenario.uplift,
                    "offer_cost": scenario.offer_cost,
                    "threshold": float(THRESHOLDS[index]),
                    "contacted": count,
                    "contact_rate": float(count / len(probabilities))
                    if len(probabilities)
                    else 0.0,
                    "expected_value": total,
                    "value_positive": bool(total > 0.0),
                }
            )
    return cells


def plot_policy_curve(curve: list[dict], optimum: float, output_path: Path) -> None:
    thresholds = [row["threshold"] for row in curve]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        thresholds,
        [row["expected_value"] for row in curve],
        color="#315d88",
        label="Planned value from scores",
    )
    ax.plot(
        thresholds,
        [row["realised_expected_value"] for row in curve],
        color="#cb7546",
        label="Realised on this sample",
    )
    ax.axhline(0, linestyle="--", color="gray", label="Contact nobody")
    ax.axvline(
        optimum, linestyle="-.", color="#315d88", label=f"EV optimum {optimum:.2f}"
    )
    ax.axvline(
        SERVED_THRESHOLD, linestyle=":", color="#cb7546", label="Served threshold 0.50"
    )
    ax.set(
        xlim=(0, 1),
        xlabel="Contact threshold",
        ylabel="Expected value under stated assumptions",
        title="Retention value by threshold: assumptions are inputs, not measurements",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_probabilities(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"{path} not found.\n{MISSING_INPUT}")
    frame = (
        pd.read_csv(file)
        if file.suffix.lower() == ".csv"
        else pd.DataFrame(json.loads(file.read_text()))
    )
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing required columns {missing}.\n{MISSING_INPUT}"
        )
    if frame.empty:
        raise SystemExit(f"{path} contains no rows.\n{MISSING_INPUT}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expected-value retention policy under stated cost assumptions."
    )
    parser.add_argument(
        "--probabilities",
        help="CSV/JSON of out-of-fold scores: probability, MonthlyCharges, churn",
    )
    parser.add_argument("--monthly-margin-rate", type=float, default=0.30)
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--contact-cost", type=float, default=2.0)
    parser.add_argument("--offer-cost", type=float, default=60.0)
    parser.add_argument("--acceptance-rate", type=float, default=0.50)
    parser.add_argument("--uplift", type=float, default=0.25)
    parser.add_argument("--budget", type=int, help="Maximum customers contactable")
    parser.add_argument("--output", default="reports/policy.json")
    parser.add_argument("--figure", default="reports/policy_curve.png")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    if not args.probabilities:
        raise SystemExit(MISSING_INPUT)
    economics = Economics(
        monthly_margin_rate=args.monthly_margin_rate,
        horizon_months=args.horizon_months,
        contact_cost=args.contact_cost,
        offer_cost=args.offer_cost,
        acceptance_rate=args.acceptance_rate,
        uplift=args.uplift,
    )
    frame = load_probabilities(args.probabilities)
    prob, charges, churn = (
        frame["probability"],
        frame["MonthlyCharges"],
        frame["churn"],
    )
    report = optimal_policy(prob, charges, churn, economics, budget=args.budget)
    report["sensitivity"] = sensitivity_grid(prob, charges, churn, economics)
    report["assumption_source"] = (
        "Benchmark planning assumptions, not measured from this dataset. See DEFAULTS."
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.no_figure:
        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        plot_policy_curve(report["curve"], report["optimal"]["threshold"], figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
