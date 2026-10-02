# Requirement audit

| Brief requirement | Evidence |
|---|---|
| Distributions and feature types | Executed `notebooks/eda.ipynb`, distribution charts |
| Missing values and string TotalCharges | `data.clean_features`, fold-local numeric/category imputers, tests |
| Categorical encoding | Fold-local OneHotEncoder with unknown category contract |
| Churn imbalance | Training prevalence 26.54%; balanced class weights; AP/F1 rationale |
| Baseline and ensemble | Logistic Regression and Random Forest in `train.py` |
| CV and tuning | Seeded stratified 5-fold, eight forest configurations |
| ROC-AUC, F1, PR-AUC; accuracy rationale | README comparison, artifacts/metrics.json, AP definition |
| No preprocessing leakage | Full sklearn Pipeline inside CV; learned-median invariant test |
| Saved operational model | joblib pipeline, integrity/version manifest, reload/CLI tests |
| Raw one-customer JSON, binary/probability output | `sample_customer.json`, CLI, API contract tests |
| Dependencies and modular engineering | Pinned runtime/dev/optional files, src package, lint/format, CI |
| Brief setup/results/top three next priorities | Main README; extended operational details outside it |
| Pragmatism and time scope | Core CLI path remains sufficient; optional extensions clearly separated |

The extensions are not required by the brief and do not imply production readiness.
They demonstrate failure behavior, reproducibility and lifecycle awareness. There
is no claimed time-to-completion, seniority certification or business-impact number.

## Beyond the brief

The brief asked for a model. These additions ask what to do with it, and where it
should not be trusted. Each is reproducible by one command.

| Addition | Question it answers | Evidence | Stated limit |
|---|---|---|---|
| Expected-value decision layer | Which customers are worth contacting, and when is the programme worth nothing? | `reports/policy.json`, `src/telco_churn/policy.py` | Assumptions are inputs; 8 of 25 combinations return nothing |
| Per-customer rule vs threshold | Is a single cutoff the right instrument at all? | Policy baselines: 21,123 vs 18,201 best threshold | Arithmetic on stated assumptions, not a forecast |
| Calibrated artifact | Are the probabilities usable as risk levels? | `reports/calibration_served.json`; mean 0.3928 to 0.2646 against 0.2654 observed | Opt-in file; reviewed default unchanged |
| Segment breakdown | Where is the model unusable? | `reports/segments.md`; zero flagged on two-year contracts | Holdout diagnostic; not a fairness audit |
| Cohort transfer | Does it hold up on newer customers? | `reports/cohort_shift.json`; 0.771 vs 0.836 control | Tenure proxies cohort, not time |
| Ranked worklist | What does a retention team actually receive? | `reports/worklist.csv` | Priorities for review, not approved actions |
| Model card | Should this be trusted for a given use? | `docs/model_card.md` | No production owner |
| One-command setup | Can a reviewer run it? | `Makefile`, `setup.ps1`, `src/telco_churn/paths.py` | Python 3.10/3.11 only, by manifest design |

The decision layer is deliberately the only addition that touches money, and it
reports the range of assumptions under which the programme is value-destroying rather
than quoting a single return.
