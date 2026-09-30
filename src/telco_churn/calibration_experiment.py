"""Nested training-only calibration experiment; never reads holdout labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score, f1_score
from .data import SEED, load_split
from .train import make_pipeline


def experiment(data_path='data/telco.csv', output_dir='reports'):
    X, _, y, _ = load_split(data_path)
    raw_prob, calibrated_prob = np.zeros(len(y)), np.zeros(len(y))
    fold_params = []
    outer = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for train_idx, valid_idx in outer.split(X, y):
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED + 1)
        pipeline = make_pipeline(RandomForestClassifier(n_estimators=150, class_weight='balanced',
                                                        random_state=SEED, n_jobs=1))
        search = GridSearchCV(pipeline, {'model__max_depth': [6, None],
                                        'model__min_samples_leaf': [2, 8],
                                        'model__max_features': ['sqrt', 0.7]},
                              scoring='average_precision', cv=inner, n_jobs=2)
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_params.append(search.best_params_)
        raw_prob[valid_idx] = search.best_estimator_.predict_proba(X.iloc[valid_idx])[:, 1]
        # Calibration folds include the full preprocessing pipeline, not precomputed features.
        calibrated = CalibratedClassifierCV(clone(search.best_estimator_), method='sigmoid',
                                             cv=inner, ensemble=True, n_jobs=2)
        calibrated.fit(X.iloc[train_idx], y.iloc[train_idx])
        calibrated_prob[valid_idx] = calibrated.predict_proba(X.iloc[valid_idx])[:, 1]
    def summarize(p):
        return {'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p)),
                'roc_auc': float(roc_auc_score(y, p)),
                'average_precision': float(average_precision_score(y, p)),
                'f1_at_0_5': float(f1_score(y, p >= .5))}
    report = {'evaluation': '3-fold outer out-of-fold predictions on original training partition only',
              'rows': len(y), 'holdout_used': False, 'inner_folds': 3,
              'fold_best_parameters': fold_params, 'raw_forest': summarize(raw_prob),
              'sigmoid_calibrated_forest': summarize(calibrated_prob),
              'default_model_changed': False,
              'caveats': ['Hyperparameter search stays within each outer training fold.',
                          'Calibration CV uses parameters selected within outer training data; outer validation remains unseen.',
                          'Calibrated ensemble also averages multiple models, so differences are not purely due to sigmoid mapping.',
                          'Lower Brier/log-loss does not establish future calibration or a business-optimal threshold.',
                          'Fixed 0.5 threshold is not tuned; calibrated probabilities can change precision/recall balance.']}
    out=Path(output_dir);out.mkdir(exist_ok=True, parents=True)
    (out/'calibration_experiment.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for p,label,color in [(raw_prob,'Raw forest','#315d88'),(calibrated_prob,'Sigmoid ensemble','#cb7546')]:
        observed,predicted=calibration_curve(y,p,n_bins=8,strategy='quantile')
        axes[0].plot(predicted,observed,'o-',label=label,color=color)
        axes[1].hist(p,bins=20,alpha=.5,label=label,color=color)
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Predicted score',ylabel='Observed churn',title='Training-only outer-fold reliability')
    axes[1].set(xlabel='Predicted score',ylabel='Customers',title='Out-of-fold score distribution')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'calibration_experiment.png',dpi=150,bbox_inches='tight');plt.close(fig)
    return report

if __name__=='__main__':
    print(json.dumps(experiment(),indent=2))
