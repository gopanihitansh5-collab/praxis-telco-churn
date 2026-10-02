# Telco customer churn

A churn model, and the retention decision it implies. Leakage-safe training, a
raw-JSON inference interface, and an expected-value layer that turns a score into a
contact decision with costs attached.

## Quick start

Python 3.10 or 3.11. One command:

```bash
make setup && make predict          # macOS / Linux
.\setup.ps1                         # Windows PowerShell
```

Or by hand:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt && python -m pip install -e .
python predict.py --input sample_customer.json
```

```json
{"prediction": 1, "churn": "Yes", "churn_probability": 0.7975335395589493, "threshold": 0.5}
```

The trained model and its checksum manifest are committed, so inference needs no
dataset and no retraining. Input can also be piped to `python predict.py`. Commands
work from any directory. `sample_customer.json` is the complete request: all 19
feature fields are required, `customerID` is optional and never modeled.

The dataset is not redistributed. `make data` fetches it from Kaggle and verifies the
SHA-256; `python download_data.py --csv <file>` adopts a manual download with no
network call.

## Three findings

These are the results worth a reviewer's attention. Each is reproducible by one
command and each carries its limits.

**1. The model's probabilities overstate risk by ~48%, and that breaks any business case.**
Across training outer folds the tuned forest's mean predicted probability is **0.3928**
against an observed churn rate of **0.2654**. Ranking is fine; the numbers are not risk
levels. Sigmoid calibration brings the mean to **0.2646** against 0.2654 observed, with
Brier 0.1570 → 0.1352 and log loss 0.4692 → 0.4176. An expected-value calculation on
the uncalibrated scores inflates the projected return by roughly the same 48%.

**2. At the served threshold the model is blind to 45% of the customer base.**
Blended holdout ROC-AUC is 0.845. Per segment:

| Contract | n | Churn rate | Flagged at 0.5 | Churners found | Max probability |
|---|---:|---:|---:|---:|---:|
| Month-to-month | 773 | 42.6% | 538 | 290 of 329 | 0.947 |
| One year | 300 | 12.0% | 4 | **0 of 36** | 0.544 |
| Two year | 336 | 2.7% | **0** | **0 of 9** | **0.417** |

No two-year customer can cross 0.5 at any input, because the highest score the model
assigns in that segment is 0.417. Those two segments are **45% of the holdout** and hold
**54% of every churner the model misses**. One blended number hides this entirely.

**3. The right policy is not a threshold at all.**
On the training partition, under documented cost assumptions:

| Policy | Contacted | Expected value | Value-destroying contacts |
|---|---:|---:|---:|
| Contact nobody | 0 | 0 | 0 |
| Contact everybody | 5,634 | **−81,262** | 4,303 |
| Served threshold 0.5 | 1,277 | 17,814 | 197 |
| Best single threshold (0.46) | 1,439 | 18,201 | 243 |
| **Per-customer expected value** | 1,331 | **21,123** | **0** |

Contacting each customer where their *own* expected value is positive beats the best
possible single threshold by **16%**, while contacting fewer people and making zero
value-destroying contacts. A high-margin customer is worth calling at 40% risk; a
low-margin one is not worth calling at 60%. No single cutoff can be right for both.

The served 0.5 was already near the best available threshold — the threshold was not
the mistake, thresholds were.

## Model results and decisions

| Model | CV ROC-AUC | CV AP | CV F1 | Holdout ROC-AUC | Holdout AP | Holdout F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8459 | 0.6600 | 0.6279 | 0.8413 | 0.6326 | 0.6136 |
| Tuned Random Forest | 0.8480 | 0.6643 | 0.6324 | 0.8449 | 0.6523 | 0.6332 |

- Seed 42; stratified 80/20 split: 5,634 training and 1,409 holdout rows. Training
  churn prevalence 26.54%. Balanced class weights, no pre-CV oversampling.
- Cleaning, numeric median imputation and scaling, categorical mode imputation and
  one-hot encoding, and the estimator are one sklearn `Pipeline`, so learned
  preprocessing stays inside every fold.
- Five-fold stratified CV; eight Random Forest configurations. Selection uses
  **training CV average precision**, never holdout scores. Selected: 150 trees,
  depth 6, leaf size 2, feature fraction 0.7.
- Holdout precision 0.5351, recall 0.7754, TN/FP/FN/TP = 783/252/84/290 at threshold 0.5.
- 95% bootstrap intervals (500 resamples, sampling variation only): ROC-AUC
  0.8222–0.8669, AP 0.5986–0.7037. The forest's margin over the baseline sits inside
  these intervals, so **no claim of statistical superiority is made** — only that it won
  a pre-declared selection rule on training folds.
- Accuracy is not a headline: always-predict-no-churn scores ~73.5% and finds nobody.
  ROC-AUC measures ranking, AP emphasises precision/recall under imbalance, F1 reflects
  the threshold trade-off. PR-AUC here means sklearn **average precision**.

## Does it hold up on new customers?

A random split assumes training and serving data share a distribution. In production
they do not. This dataset has no timestamps, but tenure proxies arrival cohort.

Trained on the 3,996 customers at 12+ months tenure and scored on the 1,638 below it,
ROC-AUC is **0.771** against **0.836** for a random split of the same population at the
same train and test sizes. That 0.065 gap is what a random split cannot show.

Two controls keep it honest. The random-split control holds population and sample size
fixed. A within-cohort reference scores the same recent customers from a model trained
inside their own cohort, separating transfer loss from the intrinsic difficulty of new
customers, who churn at 48.2% against 17.6%. Tenure is a **proxy for cohort, not a
timestamp**, and no temporal-validation claim is made.

## Run it

```bash
make test                      # 122 tests
make lint                      # ruff check and format
make reports                   # regenerate everything below (needs the dataset)

python -m telco_churn.calibrate        # calibrated artifact + out-of-fold scores
python -m telco_churn.segments         # per-segment performance
python -m telco_churn.cohort           # tenure-cohort transfer
python -m telco_churn.policy --probabilities artifacts/oof_probabilities.csv
python -m telco_churn.score_batch      # ranked worklist for the holdout
```

`reports/worklist.csv` is the operational output: customers ordered by expected value,
each with their own break-even probability so a reviewer can see why they qualified.

Evidence: `artifacts/metrics.json`, `reports/policy.json`, `reports/segments.md`,
`reports/cohort_shift.json`, `reports/diagnostics.json`. Executed EDA in
`notebooks/eda.ipynb`. [Model card](docs/model_card.md) states intended and
out-of-scope use. [Operations](docs/operations.md), [experiments](docs/experiments.md),
[rubric audit](docs/rubric.md).

## Service and lifecycle extensions

```bash
uvicorn telco_churn.api:app --host 127.0.0.1 --port 8000
curl -H 'Content-Type: application/json' --data @sample_customer.json http://127.0.0.1:8000/predict
```

`/health`, `/predict`, `/predict/batch` (1–100 profiles), `/docs`. Pydantic validation,
clean 422s, bounded bodies, load-once startup checks, atomic batch failure, no
profile or identifier logging.

| Extension | Value | Scope / limitation |
|---|---|---|
| Decision layer | Expected value per customer, sensitivity surface | Assumptions are inputs, not measurements |
| Worklist | Ranked contacts with per-customer break-even | Priorities for review, not approved actions |
| Segment report | Where the model is unusable | Holdout diagnostic; not a fairness audit |
| Cohort transfer | Degradation under realistic shift | Tenure is a cohort proxy, not a timestamp |
| Calibration | Probabilities usable as risk levels | Opt-in artifact; default unchanged |
| Docker | Repeatable serving image, non-root | Built and smoke-tested in CI; not deployed |
| MLflow | Aggregate metrics, params, data hash | Opt-in local tracking; no customer rows |
| SHAP | Top raw-feature contributions | Forest-only; associations, not causes |
| PSI drift | Training reference vs current inputs | Offline, numeric-only; not concept drift |

Install `requirements-optional.txt` for MLflow and SHAP.

## What the numbers do not say

The expected-value figures are **arithmetic on stated assumptions, not forecasts**. The
defaults — 30% margin on charges, 12-month horizon, $2 contact, $60 offer, 50%
acceptance, 25% uplift — are documented planning benchmarks, not measurements from this
dataset. The sensitivity grid is the actual deliverable: **8 of 25 assumption
combinations return nothing at all**, and viability turns on the save rate, which nothing
here measures. The decision to run such a campaign depends more on that unmeasured
number than on the model.

This is a production-shaped reference implementation, not a deployed system. No
authentication, rate limiting, TLS, SLA, labelled feedback loop or retraining trigger.
`joblib` uses pickle: **only load trusted local models**; the checksum detects corruption,
not a hostile publisher. The extended work is separate from the brief's 2.5-hour core,
and no claim is made that all of it was built inside that time.

## With two more days

1. **Measure the uplift.** Everything above is sensitivity analysis because the save
   rate is unknown. A randomised holdback on the next campaign turns the whole decision
   layer from parameterised arithmetic into a measured business case. Nothing else on
   this list matters as much.
2. **Fix the segment blind spot.** The model cannot flag long-contract customers at any
   usual cutoff. Per-segment thresholds, or a model trained within those segments, with
   validation declared before looking — not retuned against the holdout already read.
3. **Close the feedback loop.** Churn labels arrive months after the prediction, so
   monitoring needs delayed-label joins before any retraining trigger is honest. Add
   categorical drift, subgroup calibration, ownership and rollback criteria. No blind
   retraining.
