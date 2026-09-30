"""Offline numeric PSI prototype. Uses training-only bins, never triggers retraining."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import NUMERIC, clean_features, load_split


def proportions(values: pd.Series, edges: list[float]) -> np.ndarray:
    finite = values.dropna().to_numpy()
    cuts = np.array([-np.inf, *edges, np.inf])
    counts = np.histogram(finite, cuts)[0].astype(float)
    # Missingness is its own bin. Smoothing prevents undefined log ratios.
    counts = np.r_[counts, values.isna().sum()]
    return (counts + 0.5) / (len(values) + 0.5 * len(counts))


def make_reference(frame: pd.DataFrame) -> dict:
    clean = clean_features(frame)
    reference = {"schema_version": 1, "rows": len(clean), "features": {}}
    for feature in NUMERIC:
        values = clean[feature].dropna()
        edges = (
            sorted(set(float(x) for x in values.quantile(np.linspace(0.1, 0.9, 9))))
            if len(values)
            else []
        )
        reference["features"][feature] = {
            "edges": edges,
            "proportions": proportions(clean[feature], edges).tolist(),
        }
    return reference


def compare(reference: dict, frame: pd.DataFrame) -> dict:
    if reference.get("schema_version") != 1:
        raise ValueError("Unsupported drift reference")
    if len(frame) < 100:
        raise ValueError("PSI needs at least 100 profiles for this diagnostic")
    clean = clean_features(frame)
    results = {}
    for feature in NUMERIC:
        spec = reference["features"][feature]
        observed = proportions(clean[feature], spec["edges"])
        expected = np.array(spec["proportions"])
        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        results[feature] = {"psi": psi, "review_flag": psi >= 0.2}
    return {
        "rows": len(frame),
        "features": results,
        "caveat": "0.2 is a heuristic review flag, not a validated alert or proof of concept drift. Numeric-only prototype; no labels or automated retraining.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--reference", default="artifacts/drift_reference.json")
    parser.add_argument("--current", help="CSV containing raw feature columns")
    args = parser.parse_args()
    if args.current:
        report = compare(
            json.loads(Path(args.reference).read_text()), pd.read_csv(args.current)
        )
        print(json.dumps(report, indent=2))
    else:
        train, _, _, _ = load_split(args.data)
        Path(args.reference).write_text(
            json.dumps(make_reference(train), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
