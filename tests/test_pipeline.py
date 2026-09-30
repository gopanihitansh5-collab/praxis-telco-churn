import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from telco_churn.data import clean_features, load_split
from telco_churn.inference import predict_customer
from telco_churn.train import make_pipeline

@pytest.fixture
def customer():
    return json.loads(open('sample_customer.json').read())


def test_string_charge_and_blank(customer):
    row = clean_features(pd.DataFrame([customer]))
    assert row.TotalCharges.iloc[0] == 29.85
    customer['TotalCharges'] = ' '
    assert np.isnan(clean_features(pd.DataFrame([customer])).TotalCharges.iloc[0])


@pytest.mark.parametrize('mutation', ['missing', 'negative', 'invalid_number', 'infinite', 'bad_senior', 'extra'])
def test_bad_input(customer, mutation):
    if mutation == 'missing':
        del customer['tenure']
    elif mutation == 'negative':
        customer['tenure'] = -1
    elif mutation == 'invalid_number':
        customer['TotalCharges'] = 'abc'
    elif mutation == 'infinite':
        customer['MonthlyCharges'] = float('inf')
    elif mutation == 'bad_senior':
        customer['SeniorCitizen'] = 3
    else:
        customer['Churn'] = 'Yes'
    with pytest.raises(ValueError):
        predict_customer(customer)


def test_saved_model_prediction(customer):
    result = predict_customer(customer)
    assert result['prediction'] in [0, 1]
    assert 0 <= result['churn_probability'] <= 1
    assert result['prediction'] == int(result['churn_probability'] >= 0.5)


def test_unseen_category_and_missing_charge(customer):
    customer['PaymentMethod'] = 'New payment method'
    customer['TotalCharges'] = None
    assert 0 <= predict_customer(customer)['churn_probability'] <= 1


def test_serialization_roundtrip(customer, tmp_path):
    model = joblib.load('artifacts/model.joblib')
    path = tmp_path / 'copy.joblib'
    joblib.dump(model, path)
    assert predict_customer(customer, path) == predict_customer(customer)


def test_preprocessing_learns_only_training_values(customer):
    rows = []
    for tenure, label in [(1, 0), (3, 1), (5, 0), (7, 1)]:
        row = customer.copy()
        row['tenure'] = tenure
        rows.append(row)
    model = make_pipeline(LogisticRegression(max_iter=1000))
    model.fit(pd.DataFrame(rows), [0, 1, 0, 1])
    imputer = model['preprocess'].named_transformers_['numeric']['impute']
    assert imputer.statistics_[1] == 4
    outlier = customer.copy()
    outlier['tenure'] = 999
    model.predict_proba(pd.DataFrame([outlier]))
    assert imputer.statistics_[1] == 4


def test_cli_json_output():
    result = subprocess.run([sys.executable, 'predict.py', '--input', 'sample_customer.json'],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['prediction'] in [0, 1]


def test_cli_error_exit():
    result = subprocess.run([sys.executable, 'predict.py'], input='{}', capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Missing required fields' in json.loads(result.stderr)['error']


def test_non_object_input():
    with pytest.raises(ValueError, match='one JSON object'):
        predict_customer([])


def test_customer_id_does_not_affect_prediction(customer):
    first = predict_customer(customer)
    customer['customerID'] = 'different-identifier'
    assert predict_customer(customer) == first


def test_column_order_does_not_affect_prediction(customer):
    assert predict_customer(dict(reversed(list(customer.items())))) == predict_customer(customer)


def test_deterministic_bootstrap_intervals():
    from telco_churn.diagnostics import bootstrap_intervals
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.6, 0.7, 0.4])
    a = bootstrap_intervals(y, p, repeats=30)
    assert a == bootstrap_intervals(y, p, repeats=30)
    for interval in a.values():
        assert 0 <= interval['lower'] <= interval['upper'] <= 1


def test_calibration_report_preserves_holdout_boundary():
    report = json.loads(open('reports/calibration_experiment.json').read())
    assert report['holdout_used'] is False
    assert report['default_model_changed'] is False
    assert report['rows'] == 5634
    assert len(report['fold_best_parameters']) == 3
    for name in ['raw_forest', 'sigmoid_calibrated_forest']:
        for metric in ['brier', 'roc_auc', 'average_precision', 'f1_at_0_5']:
            assert 0 <= report[name][metric] <= 1
