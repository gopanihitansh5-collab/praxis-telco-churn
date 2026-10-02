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
alone. The served default artifact is unchanged. A business threshold needs cost/value information and training-only
validation, not retuning against the already inspected holdout. Full fold parameters
and caveats are in `reports/calibration_experiment.json`.


## Nested training-only threshold study

`python -m telco_churn.threshold` produces `reports/threshold_experiment.json`.
The three outer-fold thresholds are 0.51, 0.55 and 0.53. At outer evaluation,
F1 is 0.6319 at fixed 0.5 and 0.6334 with inner-OOF F1 selection. Precision
rises from 0.5339 to 0.5497 while recall falls from 0.7739 to 0.7472. This
small difference is not proof of superiority. Keep the served default 0.5 rather
than advertising a tuned threshold as an upgrade. No holdout rows were used.
The threshold objectives here ignore retention cost/value; no business policy
can be inferred from these scores.

## Segment performance on the holdout

`python -m telco_churn.segments` writes `reports/segments.json`, `reports/segments.md`
and `reports/segments.png`: metrics per contract type, internet service, senior status,
payment method and tenure band, with sample size beside every figure.

The blended holdout ROC-AUC of 0.845 conceals a hard operational limit. At threshold
0.5 the model flags four of 300 one-year-contract customers and none of 336 two-year
customers, finding zero of their 45 churners. The highest score the model assigns
anywhere in the two-year segment is 0.417, so no input crosses the cutoff. Those two
segments are 45% of the holdout and hold 54% of all missed churners.

Ranking inside those segments is not broken: ROC-AUC is 0.738 and 0.766. The failure is
the fixed cutoff meeting a low base rate, which is exactly the kind of problem a single
blended metric cannot surface. Metrics that are genuinely undefined are reported as null
with a reason; sklearn's `zero_division` would have returned 0.0 and erased the
distinction between "scored badly" and "never scored at all".

These remain post-selection holdout diagnostics. They must not be used to pick features
or retune, and segment differences on small samples are not evidence of a real gap
without interval estimates. This is a performance breakdown, not a fairness audit.

## Tenure-cohort transfer

`python -m telco_churn.cohort` writes `reports/cohort_shift.json`. Random stratified
validation assumes training and serving data share a distribution; a deployed model
scores customers who arrive later. This dataset has no timestamps, so tenure is used as
a proxy for arrival cohort.

| Established (12+ months) to recent (under 12) | ROC-AUC | Average precision |
|---|---:|---:|
| Cross-cohort transfer | 0.7707 | 0.7481 |
| Random-split control, same sizes | 0.8357 | 0.6420 |
| Within-cohort reference | 0.7898 | 0.7584 |

The 0.065 ROC-AUC gap against the control is the finding: the control holds population
and sample size fixed, so the gap is not an artifact of either. The within-cohort
reference scores the same recent customers from a model trained inside their own cohort,
which separates transfer loss from the intrinsic difficulty of new customers — they churn
at 48.2% against 17.6%.

Average precision *rises* under transfer. That is a base-rate artifact of the recent
cohort's higher churn rate, not an improvement, and ROC-AUC is the comparable metric
here. Non-tenure PSI confirms the cohorts genuinely differ in input distribution;
tenure's own PSI is definitional and excluded from that evidence.

Tenure is a proxy, not a timestamp, and short-tenure customers churn far more, so part
of any drop is segment difficulty rather than shift. The two controls exist precisely
because that confound cannot be waved away. Only the training partition is used.

## Expected-value decision policy

`python -m telco_churn.policy --probabilities artifacts/oof_probabilities.csv` writes
`reports/policy.json` and `reports/policy_curve.png`.

Per customer, `EV = p * uplift * margin_rate * MonthlyCharges * horizon - contact_cost -
acceptance_rate * offer_cost`. Retained value comes from each customer's own charges, so
the contact decision is per customer. The closed form for the flat-value case,
`p* = (c + alpha*d) / (u*M)`, is exposed directly and is the most explainable result
here: at the mean retained value of 233.75 the break-even probability is 0.548.

| Policy | Contacted | Expected value | Realised | Negative-EV contacts |
|---|---:|---:|---:|---:|
| Contact nobody | 0 | 0 | 0 | 0 |
| Contact everybody | 5,634 | −81,262 | −79,564 | 4,303 |
| Served threshold 0.5 | 1,277 | 17,814 | 19,301 | 197 |
| Best single threshold 0.46 | 1,439 | 18,201 | 19,084 | 243 |
| Per-customer EV rule | 1,331 | 21,123 | 21,755 | 0 |

The per-customer rule dominates every threshold by 16% while contacting fewer customers
and making no value-destroying contacts. A threshold cannot avoid them: two customers
can share a probability and differ in margin, so any single cutoff contacts a
low-value customer or skips a high-value one. The served 0.5 was already close to the
best available threshold, so the original choice was sound — the instrument was not.

The sensitivity grid is the real deliverable. Over uplift and offer cost, **8 of 25
combinations return nothing at all**. At a 5% save rate the programme is dead unless the
offer is free; at the default 25% with a 120 offer it returns 14. Viability turns on the
save rate, which this dataset does not measure, so the honest conclusion is that the
decision to run the campaign depends more on that unmeasured number than on the model.

Every default is a documented planning benchmark, not a measurement. No currency figure
here is a forecast, a claimed ROI or a business-approved threshold. Measuring the uplift
needs a randomised holdback, which is why it is the first item in the README's next
priorities.
