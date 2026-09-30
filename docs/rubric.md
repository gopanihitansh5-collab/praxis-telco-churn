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
