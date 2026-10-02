# Model card: telco churn classifier

One page, written so an operator can decide whether to trust this model for a given
use. Every number here is measured in this repository; nothing is projected.

## Model details

| Field | Value |
|---|---|
| Task | Binary classification; probability that a customer cancels |
| Served artifact | `artifacts/model.joblib` — sklearn Pipeline, tuned Random Forest |
| Companion artifact | `artifacts/model_calibrated.joblib` — sigmoid-calibrated, used by the decision layer |
| Selected configuration | 150 trees, `max_depth=6`, `min_samples_leaf=2`, `max_features=0.7`, balanced class weights |
| Selection rule | Training cross-validated average precision, five stratified folds. The holdout was never used to choose |
| Decision threshold | 0.5 for the raw score output; the retention policy derives its own threshold from costs |
| Integrity | `model.manifest.json` pins schema version, sklearn version and SHA-256. Load fails closed on mismatch |
| Seed | 42 throughout |
| Versions | scikit-learn 1.7.2, pandas 2.3.3, Python 3.10 or 3.11 |

## Intended use

Ranking existing customers by cancellation risk, to prioritise a finite retention
budget. The model is a **prioritisation aid for a human-run campaign**. The expected
value layer in `src/telco_churn/policy.py` exists so that the ranking is attached to
an explicit cost assumption rather than used as if any score above 0.5 justified
spending money.

## Out-of-scope use

Do not use this model for any of the following. Each is a real limitation, not
boilerplate.

- **Any decision about an individual that affects their price, credit or service
  eligibility.** This is an association model trained on one static historical
  extract. It carries no causal claim.
- **Explaining why a customer will leave.** SHAP contributions and permutation
  importance describe this fitted model's behaviour, not the customer's reasons.
  Correlated inputs share credit arbitrarily between themselves.
- **New-customer segments the training data does not cover**, or a different market,
  tariff structure or time period. See the cohort transfer result below.
- **Treating the probability as a calibrated business risk** without reading the
  calibration evidence. The uncalibrated forest overstates risk in several score
  bands; that is why a separate calibrated artifact exists.
- **Automated action without a human in the loop.** There is no monitoring,
  authentication, rate limiting, SLA or audit trail in this repository.

## Training data

Kaggle *Telco Customer Churn* (blastchar), version 1. 7,043 rows, 21 columns,
SHA-256 recorded in `data/README.md` and in `artifacts/metrics.json`. Stratified
80/20 split: 5,634 training rows, 1,409 holdout rows. Training churn prevalence
26.54%. `customerID` is excluded from the feature set and is verified by test to have
no effect on predictions. The CSV is not redistributed.

This is a public teaching dataset with no timestamps, no geography, no tariff detail
and no record of what retention action was previously taken on any customer. It
cannot establish future business performance for any real operator.

## Evaluation

Holdout, fixed threshold 0.5:

| Metric | Logistic Regression | Tuned Random Forest |
|---|---:|---:|
| ROC-AUC | 0.8413 | 0.8449 |
| Average precision | 0.6326 | 0.6523 |
| F1 | 0.6136 | 0.6332 |
| Precision | 0.5043 | 0.5351 |
| Recall | 0.7834 | 0.7754 |

95% bootstrap intervals on the forest (500 resamples, sampling variation only, and
therefore excluding training and model-selection uncertainty): ROC-AUC
0.8222–0.8669, average precision 0.5986–0.7037, F1 0.5923–0.6656. The forest's
margin over the baseline sits well inside these intervals, so **this repository does
not claim the forest is statistically superior** — it claims it was selected on
training folds by a pre-declared rule.

Probability quality on the holdout: Brier 0.1594, log loss 0.4750. The reliability
curve overestimates churn in several bins.

Accuracy is not reported as a headline because an always-predict-no-churn model
scores about 73.5% on this data while finding nobody.

### Segment performance

The blended figures above are not safe to generalise. At threshold 0.5:

| Segment | n | Churn rate | AP | Recall | Note |
|---|---:|---:|---:|---:|---|
| Contract = Month-to-month | 773 | 42.6% | 0.686 | 0.881 | Where the model works |
| Contract = One year | 300 | 12.0% | 0.259 | **0.000** | 4 flagged, 0 of 36 churners found |
| Contract = Two year | 336 | 2.7% | 0.076 | **0.000** | Max score 0.417; cannot flag anyone |
| tenure_band = 49+ | 450 | 7.8% | 0.330 | 0.286 | |
| InternetService = No | 312 | 8.0% | 0.337 | 0.440 | |
| SeniorCitizen = 1 | 222 | 44.1% | 0.719 | 0.878 | Reporting cut only, not a fairness finding |

**One- and two-year contracts are 45% of the holdout by row count and contain 54% of
every churner the model misses.** Ranking within those segments is not broken — ROC-AUC
is 0.738 and 0.766 — but no score reaches the 0.5 cutoff, so the model is operationally
blind there. Treat a low score in those segments as *no information*, not as low risk.

Full table with every level and sample size: `reports/segments.json` and
`reports/segments.md`. Metrics that are genuinely undefined are reported as null with a
reason rather than as zero.

### Transfer to newer customers

Trained on the 3,996 customers at 12+ months tenure and scored on the 1,638 below it,
ROC-AUC is 0.771 against 0.836 for a random split of the same population at the same
train and test sizes — a 0.065 gap that random validation cannot reveal. A within-cohort
reference (0.790) separates transfer loss from the intrinsic difficulty of new customers,
who churn at 48.2% against 17.6%.

Tenure is a **proxy for arrival cohort, not a timestamp**; this is not temporal
validation. Average precision rises under transfer, which is a base-rate artifact of the
higher churn rate in the recent cohort, not an improvement. Details:
`reports/cohort_shift.json`.

## Known failure modes

- Segment performance is uneven; a single blended AUC hides it. Read
  `reports/segments.json` before applying the model to a narrow population.
- Precision at the served threshold is near 0.54, so roughly **two in five flagged
  customers would not have churned**. Any campaign costed on the assumption that
  flagged customers are churners will overspend.
- Unknown category strings are accepted silently by the fitted encoder rather than
  rejected. This is deliberate for robustness but means a renamed tariff produces a
  prediction with no warning.
- Nulls and blanks are imputed with training statistics. A systematic upstream data
  outage therefore yields confident-looking predictions from mostly imputed inputs.
- `joblib` uses pickle. Only load artifacts you produced. The SHA-256 manifest
  detects corruption; it does not authenticate a publisher.

## Monitoring and retraining

`src/telco_churn/drift.py` compares current numeric inputs against a training-only PSI
reference, with 0.2 as a heuristic review flag. It covers numeric covariate shift,
offline and on demand: no schedule, no alerting, and **distribution shift is not the same
as degraded accuracy**.

There is no labelled feedback loop here. In production, churn labels arrive weeks or
months after the prediction, so performance monitoring would need delayed-label
joins before any retraining trigger could be honest. Nothing in this repository
retrains automatically, and nothing should.

Rollback is documented in [operations.md](operations.md): model file and manifest are
a pair, restored together from one released revision.

## Ownership

This model has no production owner, on-call rotation or support commitment, and is not
registered for promotion. Any deployment decision belongs to a named owner who accepts
the limitations recorded above.
