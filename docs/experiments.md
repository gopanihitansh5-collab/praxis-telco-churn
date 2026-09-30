# Extended model diagnostics

These measurements describe the unchanged bundled estimator.

## Post-selection diagnostics

`reports/diagnostics.json` and the two plots below add uncertainty, probability
quality and raw-feature explainability without changing the selected model.

| Fixed-model holdout metric | 95% percentile bootstrap interval (500 resamples) |
|---|---:|
| ROC-AUC | 0.8222-0.8669 |
| Average precision | 0.5986-0.7037 |
| F1 at 0.5 | 0.5923-0.6656 |

These intervals include evaluation-sample variation only, not retraining or
model-selection uncertainty. They are descriptive, not evidence of superiority
over the baseline. Brier score is **0.1594** and log loss is **0.4750**. The
reliability curve shows overestimated churn scores in several bins; probabilities
must not be treated as calibrated business risk.

![Evaluation and probability reliability](../reports/evaluation_curves.png)

![Raw-feature permutation importance](../reports/permutation_importance.png)

Permutation importance uses five shuffles per original feature and the decrease
in holdout average precision. Contract, tenure and InternetService have the largest
measured decreases. Correlated variables can share importance; these findings are
associations, not causal recommendations. The holdout diagnostics must not be used
to select new features or tune the next model. More honest improvement requires
training-only validation or a fresh external/temporal sample.

## Optional training-only calibration experiment

```bash
python -m telco_churn.calibration_experiment
```

This separate diagnostic uses three outer folds on the **5,634 training rows
only**, with three-fold hyperparameter search within each outer training fold.
Sigmoid calibration wraps the entire pipeline in internal three-fold calibration.
The outer validation rows are unseen by both tuning and calibration. It does not
change the included inference model or use the 1,409 holdout rows.

| Outer-fold training estimate | Raw tuned forest | Sigmoid calibrated ensemble |
|---|---:|---:|
| Brier score (lower is better) | 0.1565 | 0.1348 |
| Log loss (lower is better) | 0.4707 | 0.4167 |
| ROC-AUC | 0.8454 | 0.8469 |
| Average precision | 0.6513 | 0.6538 |
| F1 at fixed 0.5 | 0.6365 | 0.6023 |

![Training-only calibration experiment](../reports/calibration_experiment.png)

The measured probability-quality improvement is useful, but F1 at 0.5 decreases.
Calibration is not a free improvement to every metric. The calibrated variant also
averages multiple fitted models, so differences are not caused by the sigmoid map
alone. This remains an optional experiment; the reviewed baseline/default artifact
is unchanged. A business threshold needs cost/value information and training-only
validation, not retuning against the already inspected holdout. Full fold parameters
and caveats are in `reports/calibration_experiment.json`.


## Nested training-only threshold study

`python -m telco_churn.threshold` produces `reports/threshold_experiment.json`.
The three outer-fold thresholds are 0.51, 0.55 and 0.53. At outer evaluation,
F1 is 0.6319 at fixed 0.5 and 0.6334 with inner-OOF F1 selection. Precision
rises from 0.5339 to 0.5497 while recall falls from 0.7739 to 0.7472. This
small difference is not proof of superiority. Keep the reviewed default 0.5 rather
than advertising a tuned threshold as an upgrade. No holdout rows were used.
The threshold objectives here ignore retention cost/value; no business policy
can be inferred from these scores.
