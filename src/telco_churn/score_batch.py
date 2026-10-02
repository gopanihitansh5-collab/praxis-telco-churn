"""Ranked retention worklist from a raw customer CSV; priorities, not approved actions."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .artifact import load_model
from .data import FEATURES, clean_features
from .policy import (
    DEFAULTS,
    Economics,
    break_even_probability,
    contact_cost_per_customer,
    customer_expected_value,
    retained_values,
)
from .paths import resolve_dataset, resolve_model

COLUMNS = [
    "rank",
    "customerID",
    "churn_probability",
    "monthly_charges",
    "retained_value",
    "expected_value",
    "break_even_probability",
    "action",
    "priority",
]


def score(frame: pd.DataFrame, model, economics: Economics) -> pd.DataFrame:
    """Rank by expected value, not by probability; the two orders are not the same."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    clean_features(frame)
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").to_numpy()
    value = retained_values(charges, economics)
    expected = customer_expected_value(probability, charges, economics)
    out = pd.DataFrame(
        {
            "customerID": (
                frame["customerID"].astype(str)
                if "customerID" in frame.columns
                else pd.Series([""] * len(frame), index=frame.index)
            ),
            "churn_probability": probability,
            "monthly_charges": charges,
            "retained_value": value,
            "expected_value": expected,
            "break_even_probability": [
                break_even_probability(economics, v) for v in value
            ],
        }
    )
    out["action"] = np.where(out.expected_value > 0, "contact", "no_contact")
    out = out.sort_values("expected_value", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    # Priority bands only within the contact set: banding the whole file would imply
    # the lowest band is actionable when its expected value is negative.
    out["priority"] = ""
    contacted = out.action == "contact"
    if contacted.any():
        bands = pd.qcut(
            out.loc[contacted, "expected_value"].rank(method="first", ascending=False),
            q=min(4, int(contacted.sum())),
            labels=False,
            duplicates="drop",
        )
        out.loc[contacted, "priority"] = [f"P{int(b) + 1}" for b in bands]
    return out[COLUMNS]


def select_partition(source: str | Path, frame: pd.DataFrame, partition: str) -> dict:
    """Restrict to the untrained rows by default; the same split and seed as training."""
    from .data import load_split

    X_train, X_test, _, _ = load_split(source)
    index = {"holdout": X_test.index, "train": X_train.index}.get(partition)
    if index is None:
        return {"frame": frame, "in_sample": True}
    return {"frame": frame.loc[index], "in_sample": partition == "train"}


def apply_budget(ranked: pd.DataFrame, budget: int | None) -> pd.DataFrame:
    """A budget only ever removes contacts; it never promotes a negative-value customer."""
    if budget is None:
        return ranked
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    out = ranked.copy()
    beyond = (out.index >= budget) & (out.action == "contact")
    out.loc[beyond, "action"] = "over_budget"
    out.loc[beyond, "priority"] = ""
    return out


def summarize(
    ranked: pd.DataFrame,
    economics: Economics,
    budget: int | None,
    in_sample: bool = False,
) -> dict:
    contacted = ranked[ranked.action == "contact"]
    return {
        "rows_scored": int(len(ranked)),
        "scored_in_sample": in_sample,
        "contacts_recommended": int(len(contacted)),
        "contact_rate": float(len(contacted) / len(ranked)) if len(ranked) else 0.0,
        "over_budget_suppressed": int((ranked.action == "over_budget").sum()),
        "budget": budget,
        "total_expected_value": float(contacted.expected_value.sum()),
        "mean_expected_value_per_contact": (
            float(contacted.expected_value.mean()) if len(contacted) else 0.0
        ),
        "cost_per_contact": contact_cost_per_customer(economics),
        "economics": {
            "monthly_margin_rate": economics.monthly_margin_rate,
            "horizon_months": economics.horizon_months,
            "contact_cost": economics.contact_cost,
            "offer_cost": economics.offer_cost,
            "acceptance_rate": economics.acceptance_rate,
            "uplift": economics.uplift,
        },
        "caveats": [
            "A worklist of priorities for human review, not approved retention actions.",
            *(
                [
                    "These rows include customers the model was fitted on, so the"
                    " probabilities are in-sample and every expected value here is"
                    " optimistic. Use --partition holdout for an out-of-sample worklist."
                ]
                if in_sample
                else []
            ),
            "Expected value depends on planning assumptions that this dataset does not"
            " measure, above all the uplift. See reports/policy.json for the sensitivity"
            " surface and the range of assumptions under which the programme earns nothing.",
            "Ranking is by expected value, so a lower-risk high-margin customer can"
            " outrank a higher-risk low-margin one. That is the intended behaviour.",
            "Scores come from a model fitted on one static historical extract. They are"
            " associations, carry no causal claim, and justify no price or eligibility"
            " decision about an individual.",
            "Customers on one- and two-year contracts are scored, but the model flags"
            " almost none of them at any usual cutoff; see reports/segments.json before"
            " reading a low rank there as low risk.",
            "A cheaper channel than a call would need its own cost inputs; only one"
            " contact channel is modelled here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None, help="Raw customer CSV (default: dataset)"
    )
    parser.add_argument("--model", default=None, help="Model artifact path")
    parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help="Use the default artifact instead of the calibrated one",
    )
    parser.add_argument("--output", default="reports/worklist.csv")
    parser.add_argument("--summary", default="reports/worklist_summary.json")
    parser.add_argument("--budget", type=int, default=None, help="Max contacts")
    parser.add_argument(
        "--top", type=int, default=None, help="Write only the top N rows"
    )
    parser.add_argument(
        "--partition",
        choices=("holdout", "train", "all"),
        default="holdout",
        help="Which rows of the dataset to score (default holdout: out-of-sample)",
    )
    for field, value in [
        ("monthly-margin-rate", DEFAULTS.monthly_margin_rate),
        ("contact-cost", DEFAULTS.contact_cost),
        ("offer-cost", DEFAULTS.offer_cost),
        ("acceptance-rate", DEFAULTS.acceptance_rate),
        ("uplift", DEFAULTS.uplift),
    ]:
        parser.add_argument(f"--{field}", type=float, default=value)
    parser.add_argument("--horizon-months", type=int, default=DEFAULTS.horizon_months)
    args = parser.parse_args()

    economics = Economics(
        monthly_margin_rate=args.monthly_margin_rate,
        horizon_months=args.horizon_months,
        contact_cost=args.contact_cost,
        offer_cost=args.offer_cost,
        acceptance_rate=args.acceptance_rate,
        uplift=args.uplift,
    )
    source = resolve_dataset(args.input)
    frame = pd.read_csv(source)
    # Churn is the label, never an input. Dropping it here keeps a labelled training
    # export from silently becoming a feature.
    labelled = "Churn" in frame.columns
    frame = frame.drop(columns=[c for c in ("Churn",) if c in frame.columns])
    in_sample = True
    if args.input is None and labelled:
        # The bundled CSV contains the rows the model was fitted on. Scoring those is
        # in-sample and inflates every expected value, so the holdout is the default.
        selection = select_partition(source, frame, args.partition)
        frame, in_sample = selection["frame"], selection["in_sample"]
    elif args.partition != "holdout":
        raise SystemExit("--partition applies to the bundled dataset, not --input.")
    model_path = resolve_model(args.model, calibrated=not args.uncalibrated)
    if not Path(model_path).exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Run python -m telco_churn.calibrate to build the calibrated artifact,"
            " or pass --uncalibrated to use the reviewed default."
        )
    ranked = apply_budget(score(frame, load_model(model_path), economics), args.budget)
    report = summarize(ranked, economics, args.budget, in_sample)
    report["input"] = str(source)
    report["model"] = str(model_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    (ranked.head(args.top) if args.top else ranked).to_csv(output, index=False)
    Path(args.summary).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
