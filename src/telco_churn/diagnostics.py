"""Post-selection diagnostics. These must never drive holdout-based tuning."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                            average_precision_score, f1_score, brier_score_loss,
                            log_loss)
from .data import SEED


def bootstrap_intervals(y, probability, repeats=500, seed=SEED):
    """Percentile bootstrap for one fixed model and fixed evaluation sample.

    Includes sampling uncertainty only, not training/model-selection uncertainty.
    """
    y, probability = np.asarray(y), np.asarray(probability)
    rng = np.random.default_rng(seed)
    samples = {'roc_auc': [], 'pr_auc_ap': [], 'f1': []}
    for _ in range(repeats):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        samples['roc_auc'].append(roc_auc_score(y[ix], probability[ix]))
        samples['pr_auc_ap'].append(average_precision_score(y[ix], probability[ix]))
        samples['f1'].append(f1_score(y[ix], probability[ix] >= 0.5))
    return {key: {'lower': float(np.percentile(values, 2.5)),
                  'upper': float(np.percentile(values, 97.5))}
            for key, values in samples.items()}


def create_diagnostics(model, X_test, y_test, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prob = model.predict_proba(X_test)[:, 1]
    intervals = bootstrap_intervals(y_test, prob)
    importance = permutation_importance(model, X_test, y_test, scoring='average_precision',
                                        n_repeats=5, random_state=SEED, n_jobs=2)
    ranked = sorted([{'feature': name, 'ap_decrease_mean': float(mean),
                      'ap_decrease_std': float(std)}
                     for name, mean, std in zip(X_test.columns, importance.importances_mean,
                                               importance.importances_std)],
                    key=lambda row: row['ap_decrease_mean'], reverse=True)
    report = {'bootstrap_repeats': 500, 'confidence_level': 0.95,
              'confidence_intervals': intervals,
              'brier_score': float(brier_score_loss(y_test, prob)),
              'log_loss': float(log_loss(y_test, prob)),
              'permutation_importance': ranked,
              'caveats': ['Fixed-model sampling intervals exclude training and selection uncertainty.',
                          'Permutation importance is associative, not causal; correlated features share signal.',
                          'These are post-selection diagnostics, not a basis for retuning on this holdout.']}
    (output / 'diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    fpr, tpr, _ = roc_curve(y_test, prob)
    precision, recall, _ = precision_recall_curve(y_test, prob)
    observed, predicted = calibration_curve(y_test, prob, n_bins=8, strategy='quantile')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(fpr, tpr, color='#315d88', label=f'ROC-AUC {roc_auc_score(y_test, prob):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC: fixed holdout')
    axes[0].legend(loc='lower right')
    axes[1].plot(recall, precision, color='#315d88', label=f'AP {average_precision_score(y_test, prob):.3f}')
    axes[1].axhline(np.mean(y_test), color='gray', linestyle='--', label='Churn prevalence')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision / recall')
    axes[1].legend(loc='upper right')
    axes[2].plot(predicted, observed, 'o-', color='#cb7546')
    axes[2].plot([0, 1], [0, 1], '--', color='gray')
    axes[2].set(xlabel='Mean predicted score', ylabel='Observed churn fraction', title='Reliability: quantile bins')
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output / 'evaluation_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    top = ranked[:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([r['feature'] for r in top], [r['ap_decrease_mean'] for r in top],
            xerr=[r['ap_decrease_std'] for r in top], color='#315d88', capsize=3)
    ax.set(xlabel='Average-precision decrease after permutation',
           title='Top raw-feature importance: fixed holdout')
    fig.tight_layout()
    fig.savefig(output / 'permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return report
