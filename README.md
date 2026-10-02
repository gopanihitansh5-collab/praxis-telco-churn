# Telco customer churn

A churn classifier and the retention decision it supports: leakage-safe training, a
raw-JSON inference interface, and an expected-value layer that converts a score into a
contact decision with costs attached.

```bash
make setup && make predict          # macOS / Linux
.\setup.ps1                          # Windows PowerShell
```

```json
{"prediction": 1, "churn": "Yes", "churn_probability": 0.7975335395589493, "threshold": 0.5}
```

Python 3.10 or 3.11. The trained model and its checksum manifest are committed, so
inference needs no dataset and no retraining. Commands run from any directory.

---

## Headline results

### The model ranks well, but its probabilities are not risk levels

Across training outer folds the tuned forest predicts a mean churn probability of
**0.3928** against an observed rate of **0.2654** — overstating risk by 48% in relative
terms. Sigmoid calibration brings the mean to **0.2646** against 0.2654 observed:

| | Mean predicted | Brier | Log loss | ROC-AUC |
|---|---:|---:|---:|---:|
| Tuned forest | 0.3928 | 0.1570 | 0.4692 | 0.8467 |
| Calibrated | **0.2646** | **0.1352** | **0.4176** | 0.8465 |

Ranking is unchanged, as a monotone map implies. What changes is that the output becomes
usable as a risk level — which is the precondition for costing a decision on it. The
calibrated model is a separate artifact; the reviewed default is untouched.

### At the served threshold the classifier is blind to 45% of the customer base

Blended holdout ROC-AUC is 0.845. By contract type:

| Contract | n | Churn rate | Flagged at 0.5 | Churners found | Max score |
|---|---:|---:|---:|---:|---:|
| Month-to-month | 773 | 42.6% | 538 | 290 of 329 | 0.947 |
| One year | 300 | 12.0% | 4 | **0 of 36** | 0.544 |
| Two year | 336 | 2.7% | **0** | **0 of 9** | **0.417** |

No two-year customer can cross 0.5 at any input: 0.417 is the highest score the model
assigns anywhere in that segment. These two segments are **45% of the holdout** and hold
**54% of every churner the model misses**.

Ranking inside them is sound — ROC-AUC 0.738 and 0.766. The failure is a fixed cutoff
meeting a low base rate, which no blended metric can surface. A low score on a long
contract is therefore *no information*, not low risk.

### The correct retention policy is not a threshold

Under documented cost assumptions, on the training partition:

| Policy | Contacted | Expected value | Value-destroying contacts |
|---|---:|---:|---:|
| Contact nobody | 0 | 0 | 0 |
| Contact everybody | 5,634 | **−81,262** | 4,303 |
| Served threshold 0.5 | 1,277 | 17,814 | 197 |
| Best single threshold (0.46) | 1,439 | 18,201 | 243 |
| **Per-customer expected value** | 1,331 | **21,123** | **0** |

Contacting each customer whose own expected value is positive beats the best available
threshold by **16%**, while contacting fewer people and making no value-destroying
contacts. A high-margin customer is worth saving at 40% risk; a low-margin one is not at
60%. No single cutoff can be correct for both.

The served 0.5 was already near-optimal among thresholds. The threshold was not the
error — the instrument was.

**The programme's viability rests on the save rate, not on the model.** Across uplift and
offer cost, **8 of 25 assumption combinations return nothing at all**. At a 5% save rate
the campaign is value-destroying unless the offer is free. That is the result the
sensitivity surface exists to produce, and it is why measuring the uplift leads the
roadmap below.

---

## Model development

| Model | CV ROC-AUC | CV AP | CV F1 | Holdout ROC-AUC | Holdout AP | Holdout F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8459 | 0.6600 | 0.6279 | 0.8413 | 0.6326 | 0.6136 |
| Tuned Random Forest | 0.8480 | 0.6643 | 0.6324 | 0.8449 | 0.6523 | 0.6332 |

- Seed 42; stratified 80/20 split: 5,634 training and 1,409 holdout rows. Training churn
  prevalence 26.54%. Balanced class weights, no pre-CV oversampling.
- Cleaning, numeric median imputation and scaling, categorical mode imputation and
  one-hot encoding, and the estimator form one sklearn `Pipeline`, so learned
  preprocessing stays inside every fold. A test asserts the imputer never learns from
  prediction-time rows.
- Five-fold stratified CV across eight forest configurations. Selection uses **training
  CV average precision**; holdout scores played no part. Selected: 150 trees, depth 6,
  leaf size 2, feature fraction 0.7.
- Holdout precision 0.5351, recall 0.7754, TN/FP/FN/TP = 783/252/84/290 at threshold 0.5.
  Two in five flagged customers would not have churned, which is what makes the cost
  layer necessary rather than decorative.
- 95% bootstrap intervals (500 resamples): ROC-AUC 0.8222–0.8669, AP 0.5986–0.7037. The
  forest's margin over the baseline falls inside these intervals, so the forest is
  reported as the winner of a pre-declared selection rule, not as statistically superior.

**Why accuracy is insufficient.** Always-predict-no-churn scores ~73.5% on this data and
finds nobody. ROC-AUC measures ranking, average precision emphasises the
precision/recall trade-off under 26.5% prevalence, and F1 reflects a chosen operating
point. PR-AUC here means sklearn average precision.

## Transfer to newer customers

Random validation assumes training and serving data share a distribution. A deployed
model scores customers who arrive later. This dataset carries no timestamps, so tenure
serves as a proxy for arrival cohort.

| Established (12+ months) → recent (under 12) | ROC-AUC | Average precision |
|---|---:|---:|
| Cross-cohort transfer | 0.7707 | 0.7481 |
| Random split, same population and sizes | **0.8357** | 0.6420 |
| Within-cohort reference | 0.7898 | 0.7584 |

The 0.065 ROC-AUC gap against the control is the result: the control fixes population and
sample size, so the gap is attributable to cohort structure rather than to either. The
within-cohort reference separates transfer loss from the intrinsic difficulty of new
customers, who churn at 48.2% against 17.6%. Average precision rises under transfer as a
base-rate artifact of that higher churn rate, so ROC-AUC is the comparable metric.

Tenure is a cohort proxy, not a timestamp; this is distribution-shift evidence, not
temporal validation.

## Running it

```bash
make test        # 122 tests
make lint        # ruff check and format
make reports     # regenerate every report below
make data        # fetch the dataset, verifying its SHA-256
```

```bash
python -m telco_churn.calibrate                                             # calibrated artifact + out-of-fold scores
python -m telco_churn.policy --probabilities artifacts/oof_probabilities.csv  # decision analysis
python -m telco_churn.score_batch --budget 200                              # ranked worklist
python -m telco_churn.segments                                              # per-segment performance
python -m telco_churn.cohort                                                # cohort transfer
```

`reports/worklist.csv` is the operational output: customers ordered by expected value,
each with their own break-even probability, so the reason a customer qualified is visible
on the row. Regenerating every report reproduces the committed files byte for byte.

The dataset is not redistributed. `make data` fetches it and verifies the checksum;
`python download_data.py --csv <file>` adopts a manual download with no network call.

Evidence: `artifacts/metrics.json`, `reports/policy.json`, `reports/segments.md`,
`reports/cohort_shift.json`, `reports/diagnostics.json`, and executed EDA in
`notebooks/eda.ipynb`. The [model card](docs/model_card.md) states intended and
out-of-scope use. See also [operations](docs/operations.md),
[experiments](docs/experiments.md) and the [requirement audit](docs/rubric.md).

## Engineering

```bash
uvicorn telco_churn.api:app --host 127.0.0.1 --port 8000
curl -H 'Content-Type: application/json' --data @sample_customer.json http://127.0.0.1:8000/predict
```

`/health`, `/predict`, `/predict/batch` (1–100 profiles), `/docs`. Pydantic validation
with clean 422s, bounded request bodies, load-once startup checks, atomic batch failure,
and no logging of profiles or identifiers.

Input handling is strict by contract: all 19 feature fields required, `customerID`
optional and never modelled, numeric strings and nulls accepted, blanks imputed. Invalid
scalar types, negative or non-finite numbers, fractional tenure, out-of-range
`SeniorCitizen`, duplicate JSON keys and unexpected fields each fail with a specific
message. Unknown category strings are accepted by the fitted encoder.

The model file and its manifest are a pair: `load_model` verifies schema version,
scikit-learn version and SHA-256, and fails closed rather than serving predictions from a
corrupt or mismatched artifact. 122 tests cover contracts, model integrity,
serialisation round-trips, the leakage boundary, the decision layer and the CLI. CI runs
them on Python 3.10 and 3.11, lints, builds the container and smoke-tests a real
prediction through it.

| Component | Purpose | Boundary |
|---|---|---|
| Decision layer | Expected value per customer, with a sensitivity surface | Assumptions are inputs, reported alongside results |
| Worklist | Ranked contacts with per-customer break-even | Priorities for review, not approved actions |
| Segment report | Where the model should not be trusted | Holdout diagnostic; not a fairness audit |
| Cohort transfer | Degradation under realistic distribution shift | Tenure proxies cohort, not time |
| Calibration | Probabilities usable as risk levels | Opt-in artifact; default unchanged |
| Container | Repeatable serving image, non-root user | Built and smoke-tested in CI |
| MLflow | Metrics, parameters, data hash, artifacts | Opt-in local tracking; no customer rows |
| SHAP | Top raw-feature contributions | Forest only; associations, not causes |
| PSI drift | Training reference against current inputs | Numeric covariate shift, not concept drift |

Optional dependencies for MLflow and SHAP are in `requirements-optional.txt`.

## Scope and assumptions

Every currency figure is arithmetic on stated assumptions: 30% margin on charges, a
12-month horizon, $2 per contact, a $60 offer, 50% acceptance, 25% uplift. These are
documented planning benchmarks rather than measurements from this dataset, which is why
the sensitivity surface — not any single figure — is the deliverable, and why no number
here is presented as a forecast or an approved operating threshold.

The system serves a model and a decision locally. Authentication, rate limiting, TLS,
load testing and a labelled feedback loop are outside its scope; the model card records
where the model should not be used, including any decision about an individual's price or
eligibility. `joblib` uses pickle, so only trusted local artifacts are loaded — the
manifest checksum detects corruption, not a hostile publisher.

The dataset is a single static public extract with no timestamps, tariff detail or record
of prior retention action. It supports the analysis above and cannot establish future
business performance for a real operator.

## With two more days

1. **Measure the uplift.** Every currency figure here is sensitivity analysis because the
   save rate is unknown. A randomised holdback on the next campaign converts the decision
   layer from parameterised arithmetic into a measured business case. Nothing else on
   this list changes as much.
2. **Close the long-contract blind spot.** The classifier cannot flag one- and two-year
   customers at any conventional cutoff, and they hold over half the missed churners.
   Per-segment operating points, or a model fitted within those segments, with the
   validation plan declared before looking — not retuned against a holdout already read.
3. **Build the feedback loop.** Churn labels arrive months after the prediction, so
   monitoring needs delayed-label joins before a retraining trigger can be honest. Add
   categorical drift, subgroup calibration, clear ownership and rollback criteria.
