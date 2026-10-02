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
No dataset, notebook or private input goes into the image. The CI container job builds the image and smoke-tests health plus a real prediction.
Docker was unavailable for local testing; deployment and load testing remain out of scope. Image base
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

## Input drift monitoring

```bash
python -m telco_churn.drift                         # training-only reference
python -m telco_churn.drift --current path/to/current_raw_profiles.csv
```

Numeric PSI: training quantile bins, separate missing bin, smoothed fractions;
at least 100 current profiles. `artifacts/drift_reference.json` is bundled and comes
from the original training split. 0.2 is a heuristic review flag, not a validated
alert. Sample size affects PSI; inspect seasonality and collection changes. Scope is
numeric inputs on demand: no categorical monitoring, labels, schedule, notifications or
retraining action.
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
engineering requires a training-only validation plan declared in advance and fresh final
data; an inspected holdout plot is not a basis for it.

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

## Setup and path resolution

```bash
make setup          # venv, pinned deps, editable install
make verify         # setup + test + lint + a real prediction
make all            # verify + dataset + every report
.\setup.ps1 -Reports   # the same on Windows PowerShell
```

`make help` lists every target. Python 3.10 or 3.11 is required, not preferred:
`artifacts/model.manifest.json` records scikit-learn 1.7.2 and `artifact.load_model`
fails closed on a version mismatch, so a newer interpreter with newer wheels cannot
load the committed model. `setup.ps1` locates a supported interpreter and explains the
pin rather than failing obscurely.

`src/telco_churn/paths.py` resolves the dataset, model and report directories by walking
up to the package root, so every command works from any working directory. Resolution
order for the dataset is the explicit `--data` argument, then `$CHURN_DATA`, then
`data/telco.csv`, then the current directory, then the local kagglehub cache. A missing
dataset raises with the command that fixes it and the list of locations searched.

## Dataset

```bash
make data                                  # Kaggle, verifies SHA-256
python download_data.py --csv ~/telco.csv  # adopt a manual download, no network
python download_data.py --force            # replace an existing copy
```

The CSV is deliberately not committed. It belongs to the Kaggle uploader and is not
redistributed here; `data/README.md` records the source, version and expected hash.
The download verifies against the SHA-256 recorded in `artifacts/metrics.json` and warns
loudly on a mismatch rather than silently producing different metrics.

## Decision layer and worklist

```bash
python -m telco_churn.calibrate            # calibrated artifact + out-of-fold scores
python -m telco_churn.policy --probabilities artifacts/oof_probabilities.csv
python -m telco_churn.score_batch          # ranked worklist, holdout by default
python -m telco_churn.score_batch --budget 200 --top 200
python -m telco_churn.score_batch --uplift 0.15 --offer-cost 30
```

`calibrate` must run before `policy`: the policy layer consumes out-of-fold
probabilities, because in-sample scores from a fitted forest are near-separable and
would make any expected-value estimate meaningless. Every economics field is a CLI
override; `reports/policy.json` records the assumptions used alongside the results.

`score_batch` writes `reports/worklist.csv` and a summary. It scores the holdout by
default. `--partition all` scores the full file including rows the model was fitted on,
which sets `scored_in_sample` and adds a caveat that the figures are optimistic.
`--budget` suppresses contacts beyond the cap and never promotes a negative-value
customer. `--uncalibrated` uses the reviewed default artifact instead.

Neither command changes the served model. `artifacts/model_calibrated.joblib` is a
separate opt-in file with its own manifest, selected explicitly by path, so the reported
holdout boundary and model selection still stand.

## Segment and cohort reports

```bash
python -m telco_churn.segments   # reports/segments.{json,md,png}
python -m telco_churn.cohort     # reports/cohort_shift.{json,png}
```

`segments` is a holdout diagnostic and says so in its own report; it must not be used to
select features or retune. `cohort` uses the training partition only. Both refuse to run
on substitute data and exit with a clear message when the CSV is absent.

Operationally the segment report is the one to read before trusting a score: the model
flags almost nobody on one- and two-year contracts, so a low score there carries no
information rather than meaning low risk.
