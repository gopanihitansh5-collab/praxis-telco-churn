# Telco customer churn

Leakage-safe classification, reproducible model comparison, and a raw-JSON inference
interface. Start with the **CLI** for the assessment. The optional serving and
MLOps extensions demonstrate operational thinking without changing the reviewed model.

## Quick start

Python 3.10 or 3.11, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python predict.py --input sample_customer.json
```

The saved model and checksum/version manifest are included. No dataset download or
retraining is needed for inference. Input can also be piped to `python predict.py`.

```json
{"prediction": 1, "churn": "Yes", "churn_probability": 0.7975335395589493, "threshold": 0.5}
```

`sample_customer.json` is the complete assessment request. All 19 feature fields are
required; `customerID` is optional and never modeled. Numeric strings and nulls are
accepted; blanks are imputed. Invalid scalar types, negative/nonfinite numbers,
fractional tenure, invalid SeniorCitizen, duplicate JSON keys and extra fields fail
with a clear error. Unknown category strings are accepted by the fitted encoder.
A probability is a **model score, not a validated calibrated business risk**.

## Results and decisions

| Model | Training CV ROC-AUC | Training CV AP | Training CV F1 | Holdout ROC-AUC | Holdout AP | Holdout F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8459 | 0.6600 | 0.6279 | 0.8413 | 0.6326 | 0.6136 |
| Tuned Random Forest | 0.8480 | 0.6643 | 0.6324 | 0.8449 | 0.6523 | 0.6332 |

- Seed 42; stratified 80/20 split: 5,634 training and 1,409 holdout rows.
- Training churn prevalence 26.54%. Balanced class weights, no pre-CV oversampling.
- Stateless cleaning, numeric median/scaling, categorical mode/one-hot encoding and
  estimator are one sklearn Pipeline. Learned preprocessing stays inside each fold.
- Five-fold stratified CV; eight Random Forest configurations. Model selection uses
  **training CV average precision**, never holdout scores. Tuned CV scores are
  selection-optimistic. The modest gain does not establish statistical superiority.
- Selected forest: 150 trees, depth 6, leaf size 2, feature fraction 0.7. Holdout
  precision 0.5351, recall 0.7754, TN/FP/FN/TP = 783/252/84/290 at fixed threshold 0.5.
- Accuracy alone rewards an always-no-churn model (~73.46% on training). ROC-AUC
  measures ranking, AP emphasizes precision/recall under imbalance, F1 reflects the
  threshold trade-off. Here PR-AUC means sklearn **average precision**, not trapezoidal area.

Full measurements: `artifacts/metrics.json`, `artifacts/cv_results.csv`.
Executed EDA: `notebooks/eda.ipynb`, with training-only distributions, missingness,
class balance and churn associations. Dataset source:
[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
CSV is not redistributed; source/version/checksum are in `data/README.md`.

## Retrain and verify

```bash
python -m pip install -r requirements-dev.txt
python download_data.py                    # or save source CSV as data/telco.csv
python -m telco_churn.train --data data/telco.csv --output artifacts
pytest -q
ruff check src tests predict.py download_data.py
ruff format --check src tests predict.py download_data.py
```

Python 3.10/3.11 CI runs contracts, model integrity, serialization, leakage-boundary
and CLI tests. Optional dependency tests skip unless installed. The saved estimator
is fitted on training rows only, preserving the reported holdout boundary.

## Optional service and lifecycle extensions

```bash
uvicorn telco_churn.api:app --host 127.0.0.1 --port 8000
curl -H 'Content-Type: application/json' --data @sample_customer.json http://127.0.0.1:8000/predict
```

`/health`, `/predict`, `/predict/batch` (1-100 profiles), `/docs`. Pydantic request
validation, clean 422 errors, bounded bodies and load-once startup checks. Invalid
batches fail atomically, not halfway through. No profile/identifier logging.

| Extension | Value | Scope / limitation |
|---|---|---|
| Docker | Repeatable local serving image, non-root user | Build/smoke test in CI; not deployed |
| MLflow | Aggregate metrics, parameters, data hash and artifacts | Opt-in local tracking; no customer rows |
| SHAP | Top raw-feature score contributions | Optional forest-only; associations, not causes |
| Batch API | Bounded ordered bulk inference | At most 100 profiles; no async jobs |
| PSI | Training-only reference vs numeric input distributions | Offline prototype; not concept-drift detection |
| Calibration / threshold studies | Honest training-only decision analysis | Separate from default artifact and holdout |

Install `requirements-optional.txt` to enable MLflow and SHAP.
[Operations guide](docs/operations.md) includes every command, request/response contract,
rollback and production gaps. [Experiments](docs/experiments.md) explains diagnostics,
calibration and uncertainty. [Rubric audit](docs/rubric.md) maps each requirement to evidence.

## With two additional days

1. **Decision policy:** obtain retention cost/value and temporal validation data;
   choose calibrated thresholds using training-only validation, then test once on
   fresh data. F1-optimal is not profit-optimal.
2. **Deployment safety:** authentication/rate limiting, dependency/image scanning,
   load testing, model-registry promotion and measured latency/error budgets.
3. **Feedback lifecycle:** categorical and numeric drift, delayed-label performance,
   subgroup calibration, ownership and retraining/rollback criteria. No blind retraining.

This is a production-shaped reference implementation, not a deployed production
system. The public sample cannot demonstrate future business value. Joblib uses
pickle: **only load trusted local models**. A checksum detects accidental corruption;
it does not authenticate a publisher. The extended work is separate from the brief's
2.5-hour core; no claim is made that all extensions were built inside that time.
