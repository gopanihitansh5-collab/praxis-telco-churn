"""Holdout segment breakdown. Describes measured gaps, not fairness or causation."""

import argparse
import json
from numbers import Real
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from .artifact import load_model
from .data import load_split

# Segmentation name -> raw column it is derived from.
SEGMENTS: dict[str, str] = {
    "Contract": "Contract",
    "InternetService": "InternetService",
    "SeniorCitizen": "SeniorCitizen",
    "PaymentMethod": "PaymentMethod",
    "tenure_band": "tenure",
}
TENURE_EDGES = [6, 12, 24, 48]  # Inclusive upper bounds, in months.
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]
UNKNOWN_LEVEL = "unknown"
CAVEATS = [
    "These are post-selection holdout diagnostics, not a basis for selecting "
    "features or retuning on this holdout.",
    "Segment metrics at small n are high-variance; a difference between two "
    "segments is not evidence of a real performance gap, and not evidence of "
    "unfairness, without interval estimates.",
    "This is a performance breakdown, not a fairness audit and not a legal "
    "compliance assessment.",
    "No protected-attribute conclusion follows from SeniorCitizen or gender; "
    "SeniorCitizen appears here only as a reporting cut.",
    "Lift is measured at the reporting threshold on one fixed holdout sample "
    "and is not a validated campaign return.",
    "Segments flagged insufficient_rows or single_class are reported, not "
    "dropped, and their metrics must not be quoted as performance.",
]


def tenure_band(series: pd.Series) -> pd.Series:
    """Band tenure on TENURE_EDGES. Missing tenure becomes its own level."""
    values = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(values, [-np.inf, *TENURE_EDGES, np.inf], labels=TENURE_LABELS)
    # Missing tenure keeps its rows visible instead of silently vanishing.
    return bands.astype(object).where(values.notna(), UNKNOWN_LEVEL).astype(str)


def _as_level(value: object) -> str:
    if pd.isna(value):
        return UNKNOWN_LEVEL
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value).strip() or UNKNOWN_LEVEL


def segment_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Build the string segmentation levels, positionally indexed."""
    missing = sorted(set(SEGMENTS.values()) - set(X.columns))
    if missing:
        raise ValueError(f"Missing segmentation columns: {missing}")
    out = pd.DataFrame(index=pd.RangeIndex(len(X)))
    for name, source in SEGMENTS.items():
        column = X[source].reset_index(drop=True)
        out[name] = (
            tenure_band(column)
            if name == "tenure_band"
            else column.map(_as_level).astype(str)
        )
    return out


def _levels(values: pd.Series, name: str) -> list[str]:
    present = set(values)
    if name == "tenure_band":
        return [band for band in (*TENURE_LABELS, UNKNOWN_LEVEL) if band in present]
    return sorted(present)


def segment_metrics(
    y_true: np.ndarray | pd.Series,
    prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Metrics for one group of rows.

    `lift` is precision at `threshold` divided by the segment's own churn rate:
    how many times more likely a flagged customer is to churn than a customer
    drawn at random from the same segment. 1.0 means the model adds nothing
    here. Undefined metrics are None with a `reason`; they are never 0.0.
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(prob, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError("y_true and prob must have the same length")
    flagged = p >= threshold
    n = int(y.size)
    churn_count = int(y.sum())
    flagged_count = int(flagged.sum())
    true_positive = int(np.sum(flagged & (y == 1)))
    single_class = n > 0 and churn_count in (0, n)
    precision = true_positive / flagged_count if flagged_count else None
    recall = true_positive / churn_count if churn_count else None
    # F1 = 2TP / (2TP + FP + FN); undefined only with no positives and none flagged.
    f1_denominator = flagged_count + churn_count
    churn_rate = churn_count / n if n else None
    reasons = []
    if n == 0:
        reasons.append("empty segment: no metrics defined")
    elif churn_count == 0:
        reasons.append(
            "no churn events in segment: ROC-AUC and average precision undefined"
        )
    elif churn_count == n:
        reasons.append(
            "no retained customers in segment: ROC-AUC and average precision undefined"
        )
    if n and precision is None:
        reasons.append(
            "no customers scored at or above the threshold: precision, F1 and "
            "lift undefined"
        )
    ranked = n > 0 and not single_class
    return {
        "n": n,
        "churn_count": churn_count,
        "churn_rate": churn_rate,
        "flagged": flagged_count,
        "roc_auc": float(roc_auc_score(y, p)) if ranked else None,
        "average_precision": float(average_precision_score(y, p)) if ranked else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else None,
        "lift": (
            precision / churn_rate if precision is not None and churn_rate else None
        ),
        "single_class": bool(single_class),
        "reason": "; ".join(reasons) or None,
    }


def evaluate_segments(
    X: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
    min_rows: int = 30,
) -> dict:
    """Segment-level breakdown of one fixed model on one fixed evaluation sample."""
    frame = segment_frame(X)
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(prob, dtype=float).ravel()
    if not len(frame) == y.size == p.size:
        raise ValueError("X, y_true and prob must have the same length")
    segmentations = {}
    for name in SEGMENTS:
        levels = {}
        for level in _levels(frame[name], name):
            mask = (frame[name] == level).to_numpy()
            metrics = segment_metrics(y[mask], p[mask], threshold)
            # Small segments stay in the report, flagged, because they are where
            # a blended average most easily hides a useless model.
            metrics["insufficient_rows"] = metrics["n"] < min_rows
            levels[level] = metrics
        segmentations[name] = levels
    report = {
        "rows": int(y.size),
        "threshold": float(threshold),
        "min_rows": int(min_rows),
        "holdout_used": True,
        "default_model_changed": False,
        "tenure_band_edges_months": {
            "inclusive_upper_bounds": TENURE_EDGES,
            "labels": TENURE_LABELS,
            "missing_tenure_level": UNKNOWN_LEVEL,
        },
        "overall": segment_metrics(y, p, threshold),
        "segmentations": segmentations,
        "caveats": CAVEATS,
    }
    report["weakest_segments"] = rank_weakest(report)
    return report


def rank_weakest(report: dict, metric: str = "average_precision") -> list[dict]:
    """Worst-first ranking, excluding insufficient-n and single-class segments."""
    rows = [
        {
            "segmentation": name,
            "level": level,
            "n": metrics["n"],
            "churn_rate": metrics["churn_rate"],
            metric: metrics[metric],
        }
        for name, levels in report["segmentations"].items()
        for level, metrics in levels.items()
        if not metrics.get("insufficient_rows")
        and not metrics.get("single_class")
        and metrics.get(metric) is not None
    ]
    # Name tie-break keeps repeated calls byte-identical.
    return sorted(
        rows, key=lambda row: (row[metric], row["segmentation"], row["level"])
    )


def _cell(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _flags(metrics: dict) -> str:
    flags = []
    if metrics.get("insufficient_rows"):
        flags.append("insufficient n")
    if metrics.get("single_class"):
        flags.append("single class")
    return ", ".join(flags) or "-"


def markdown_table(report: dict) -> str:
    """GitHub-flavoured table, weakest measurable segment first, n in every row."""
    rows = [
        (name, level, metrics)
        for name, levels in report["segmentations"].items()
        for level, metrics in levels.items()
    ]
    rows.sort(
        key=lambda row: (
            row[2]["average_precision"] is None,
            row[2]["average_precision"] or 0.0,
            row[0],
            row[1],
        )
    )
    overall = report["overall"]
    lines = [
        "# Holdout performance by segment",
        "",
        f"One fixed model on {report['rows']} holdout rows at threshold "
        f"{report['threshold']:.2f}. Segments under {report['min_rows']} rows are "
        "flagged `insufficient n` and their metrics are unreliable. Tenure bands "
        f"use inclusive upper bounds {TENURE_EDGES} months "
        f"({', '.join(TENURE_LABELS)}); missing tenure becomes `{UNKNOWN_LEVEL}`.",
        "",
        "Blended holdout average precision is "
        f"**{_cell(overall['average_precision'])}** and ROC-AUC is "
        f"**{_cell(overall['roc_auc'])}** over all {overall['n']} rows. "
        "Read every row below against its own n.",
        "",
        "| Segmentation | Level | n | Churn rate | ROC-AUC | AP | Precision | "
        "Recall | F1 | Lift | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, level, metrics in rows:
        lines.append(
            f"| {name} | {level} | {metrics['n']} | "
            f"{_cell(metrics['churn_rate'])} | {_cell(metrics['roc_auc'])} | "
            f"{_cell(metrics['average_precision'])} | {_cell(metrics['precision'])} | "
            f"{_cell(metrics['recall'])} | {_cell(metrics['f1'])} | "
            f"{_cell(metrics['lift'], 2)} | {_flags(metrics)} |"
        )
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in CAVEATS]
    undefined = [
        f"- `{name}` = `{level}`: {metrics['reason']}"
        for name, level, metrics in rows
        if metrics["reason"]
    ]
    if undefined:
        lines += ["", "## Undefined metrics", ""] + undefined
    return "\n".join(lines) + "\n"


def plot_segments(report: dict, output_dir: str | Path = "reports") -> Path:
    """Per-segment average precision against the blended holdout value."""
    rows = [
        (f"{name}: {level}", metrics)
        for name, levels in report["segmentations"].items()
        for level, metrics in levels.items()
        if metrics["average_precision"] is not None
    ]
    rows.sort(key=lambda row: row[1]["average_precision"], reverse=True)
    colors = [
        "#cb7546" if metrics["insufficient_rows"] else "#315d88" for _, metrics in rows
    ]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 2.2))
    ax.barh(
        [label for label, _ in rows],
        [metrics["average_precision"] for _, metrics in rows],
        color=colors,
    )
    overall = report["overall"]["average_precision"]
    if overall is not None:
        ax.axvline(
            overall,
            linestyle="--",
            color="gray",
            label=f"Blended holdout AP {overall:.3f}",
        )
        ax.legend(loc="lower right")
    for position, (_, metrics) in enumerate(rows):
        ax.text(
            metrics["average_precision"] + 0.01,
            position,
            f"n={metrics['n']}",
            va="center",
            fontsize=8,
        )
    ax.set(
        xlim=(0, 1.12),
        xlabel="Average precision (orange: fewer rows than min_rows)",
        title="Holdout average precision by segment",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    path = Path(output_dir) / "segments.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-rows", type=int, default=30)
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(
            f"{args.data} not found. Supply the raw Telco CSV with --data; "
            "this report is never generated from substitute data."
        )
    _, X_test, _, y_test = load_split(args.data)
    model = load_model(args.model)
    prob = model.predict_proba(X_test)[:, 1]
    report = evaluate_segments(X_test, y_test, prob, args.threshold, args.min_rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "segments.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "segments.md").write_text(markdown_table(report))
    plot_segments(report, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
