# Operations and extension guide

## Core and dependency boundaries

Run commands from the repository root. `requirements.txt` is the runtime/training
path, `requirements-dev.txt` adds notebook/test/lint tools, and
`requirements-optional.txt` adds SHAP and MLflow. Versions of direct dependencies
are pinned; transitive packages are resolved by pip, not a hash-locked supply chain.
`pip check` is part of CI. Supported/tested targets: Python 3.10 and 3.11.

## HTTP contracts

Start with `uvicorn telco_churn.api:app`. `/health` is ready only after startup loads
and verifies the local artifact. A missing, mismatched or corrupt artifact prevents
startup rather than serving stale or fabricated predictions.

`POST /predict`: raw `sample_customer.json`. Same four result fields as the CLI.
`POST /predict/batch`: `{"customers": [<raw customer>, ...]}`. Returns
`{"predictions": [<result>, ...]}` in input order. One to 100 items, 256 KiB body cap.
All profiles validate before any predictions. Errors: 415 wrong content type,
413 oversized body, 422 malformed/duplicate/nonstandard JSON or contract violation.
Pydantic error responses omit raw inputs. Customer IDs do not affect predictions.
Nulls/blanks are learned-imputed, and unknown category strings remain allowed.

Local API has no authentication/rate limiter, TLS, persistence, concurrent worker
budget or measured SLA. It must not be exposed directly to the internet. Synchronous
model computation is appropriate for this bounded demo, not a throughput guarantee.

## Explanations

```bash
python -m pip install -r requirements-optional.txt
CHURN_EXPLAIN=1 uvicorn telco_churn.api:app
```

Responses add `explanation`: `baseline_probability`, `all_contributions_sum`,
`top_drivers` (five feature/contribution pairs) and an interpretation caveat.
Tree SHAP uses the fitted forest's tree-path-dependent background. One-hot columns
are summed to original names. Contributions across **all** raw features plus the
baseline reconstruct class-1 probability; returned top five alone need not sum to
it. Positive contributions increase this fitted model's score. They are not causal
reasons or intervention recommendations. Tree-path-dependent SHAP can be affected
by correlated inputs. It is explicitly unsupported for a Logistic Regression
artifact. Enabled on the bundled Random Forest only, checked by additivity tests.

## Container

```bash
docker build -t telco-churn .
docker run --rm -p 127.0.0.1:8000:8000 telco-churn
```

Python 3.11 slim image, installed package, only local model/manifest, non-root user.
No dataset, notebook or private input goes into the image. Docker is unavailable in
the local build environment; the CI container job builds and checks health and a
real prediction. Consult that run before claiming a validated image. Image base
is a tag, not a digest; scanning and digest pinning remain deployment work.
The default image excludes SHAP/MLflow to avoid an unnecessarily large serving image.

## Experiment tracking

```bash
python -m telco_churn.train --tracking-uri "file://$(pwd)/mlruns"
```

Install optional requirements first. MLflow logs each model's CV/holdout scalar
metrics, seed, hyperparameters, source-data hash, selected model and output artifacts.
It does not log individual customer rows or IDs. Opt-in URI is explicit: no default
external telemetry or remote experiment server. `mlruns/` is ignored by Git. The
local tracker is tested with a temporary file store. Tracking failure surfaces as
an error; trained artifacts may already exist. No automatic registry promotion.

## Input drift prototype

```bash
python -m telco_churn.drift                         # training-only reference
python -m telco_churn.drift --current path/to/current_raw_profiles.csv
```

Numeric PSI only: training quantile bins, separate missing bin, smoothed fractions;
at least 100 current profiles. `artifacts/drift_reference.json` is bundled and comes
from the original training split. 0.2 is a heuristic review flag, not a validated
alert. Sample size affects PSI; inspect seasonality and collection changes. This
has no categorical monitoring, labels, schedule, notifications or retraining action.
Distribution shift is not the same as concept drift or degraded model performance.

## Model changes and rollback

Model file and adjacent `.manifest.json` are a pair. Restore both from the same
reviewed revision, restart, call `/health`, then compare the known sample response.
Version and SHA256 guards reject unsupported/corrupt pairs before deserialization.
Never accept arbitrary uploaded model files. SHA256 provides integrity, not trust.
Keep the immutable code/dependency/data/metrics revision with each approved artifact.
No hot reload or silent fallback.

## ML depth without holdout reuse

```bash
python -m telco_churn.calibration_experiment
python -m telco_churn.threshold
```

Calibration uses nested training-only folds. It improves measured Brier/log loss but
reduces F1 at 0.5, so it is not silently promoted. The threshold diagnostic adds a
third validation level: outer evaluation; inner out-of-fold threshold scores; tuning
within each inner training subset. It compares fixed 0.5 to training-selected F1
thresholds without touching the 1,409-row holdout. Expensive by design and optional.
Results do not provide one business-approved deployment threshold. New feature
engineering would require predeclared training-only validation and fresh final data;
it is not justified merely because an inspected holdout plot looks promising.

## Notebook

```bash
python -m ipykernel install --user --name telco-churn --display-name 'Telco Churn'
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p = Path('notebooks/eda.ipynb')
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=120, kernel_name='telco-churn',
               resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(nb, p)
PY
```
