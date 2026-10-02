# Holdout performance by segment

One fixed model on 1409 holdout rows at threshold 0.50. Segments under 30 rows are flagged `insufficient n` and their metrics are unreliable. Tenure bands use inclusive upper bounds [6, 12, 24, 48] months (0-6, 7-12, 13-24, 25-48, 49+); missing tenure becomes `unknown`.

Blended holdout average precision is **0.652** and ROC-AUC is **0.845** over all 1409 rows. Read every row below against its own n.

| Segmentation | Level | n | Churn rate | ROC-AUC | AP | Precision | Recall | F1 | Lift | Flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Contract | Two year | 336 | 0.027 | 0.766 | 0.076 | n/a | 0.000 | n/a | n/a | - |
| Contract | One year | 300 | 0.120 | 0.738 | 0.259 | 0.000 | 0.000 | 0.000 | 0.00 | - |
| tenure_band | 49+ | 450 | 0.078 | 0.826 | 0.330 | 0.323 | 0.286 | 0.303 | 4.15 | - |
| InternetService | No | 312 | 0.080 | 0.882 | 0.337 | 0.393 | 0.440 | 0.415 | 4.90 | - |
| PaymentMethod | Mailed check | 326 | 0.184 | 0.824 | 0.468 | 0.429 | 0.700 | 0.532 | 2.33 | - |
| tenure_band | 25-48 | 304 | 0.211 | 0.780 | 0.472 | 0.440 | 0.625 | 0.516 | 2.09 | - |
| tenure_band | 7-12 | 138 | 0.304 | 0.737 | 0.553 | 0.468 | 0.690 | 0.558 | 1.54 | - |
| PaymentMethod | Credit card (automatic) | 309 | 0.165 | 0.859 | 0.562 | 0.457 | 0.627 | 0.529 | 2.77 | - |
| InternetService | DSL | 484 | 0.200 | 0.831 | 0.584 | 0.530 | 0.629 | 0.575 | 2.65 | - |
| SeniorCitizen | 0 | 1187 | 0.233 | 0.846 | 0.625 | 0.509 | 0.739 | 0.603 | 2.19 | - |
| tenure_band | 13-24 | 206 | 0.291 | 0.819 | 0.647 | 0.481 | 0.833 | 0.610 | 1.65 | - |
| PaymentMethod | Bank transfer (automatic) | 300 | 0.190 | 0.869 | 0.651 | 0.581 | 0.632 | 0.605 | 3.06 | - |
| Contract | Month-to-month | 773 | 0.426 | 0.753 | 0.686 | 0.539 | 0.881 | 0.669 | 1.27 | - |
| InternetService | Fiber optic | 613 | 0.411 | 0.781 | 0.703 | 0.546 | 0.865 | 0.670 | 1.33 | - |
| SeniorCitizen | 1 | 222 | 0.441 | 0.786 | 0.719 | 0.610 | 0.878 | 0.720 | 1.38 | - |
| PaymentMethod | Electronic check | 474 | 0.435 | 0.780 | 0.726 | 0.577 | 0.874 | 0.695 | 1.33 | - |
| tenure_band | 0-6 | 311 | 0.556 | 0.753 | 0.775 | 0.634 | 0.931 | 0.754 | 1.14 | - |

## Caveats

- These are post-selection holdout diagnostics, not a basis for selecting features or retuning on this holdout.
- Segment metrics at small n are high-variance; a difference between two segments is not evidence of a real performance gap, and not evidence of unfairness, without interval estimates.
- This is a performance breakdown, not a fairness audit and not a legal compliance assessment.
- No protected-attribute conclusion follows from SeniorCitizen or gender; SeniorCitizen appears here only as a reporting cut.
- Lift is measured at the reporting threshold on one fixed holdout sample and is not a validated campaign return.
- Segments flagged insufficient_rows or single_class are reported, not dropped, and their metrics must not be quoted as performance.

## Undefined metrics

- `Contract` = `Two year`: no customers scored at or above the threshold: precision, F1 and lift undefined
