"""Deterministic raw-data validation and split shared by training and EDA."""
from numbers import Real
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
               'PaperlessBilling', 'PaymentMethod']
FEATURES = NUMERIC + CATEGORICAL
SEED = 42


def clean_features(frame):
    """Stateless conversion only. Learned imputation stays inside each CV fold."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    out = frame[FEATURES].copy()
    for col in NUMERIC:
        if not out[col].map(lambda x: x is None or isinstance(x, (str, Real)) and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{col} must be a numeric scalar, string or null')
        original = out[col].map(lambda x: np.nan if isinstance(x, str) and not x.strip() else x)
        converted = pd.to_numeric(original, errors='coerce')
        if (original.notna() & converted.isna()).any():
            raise ValueError(f'{col} must be numeric or missing')
        if np.isinf(converted).any():
            raise ValueError(f'{col} must be finite')
        if (converted.dropna() < 0).any():
            raise ValueError(f'{col} must be nonnegative')
        out[col] = converted
    if not out['tenure'].dropna().map(lambda x: float(x).is_integer()).all():
        raise ValueError('tenure must be a whole number of months')
    if not out['SeniorCitizen'].dropna().isin([0, 1]).all():
        raise ValueError('SeniorCitizen must be 0 or 1')
    for col in CATEGORICAL:
        if not out[col].map(lambda x: x is None or isinstance(x, str) or isinstance(x, Real) and pd.isna(x)).all():
            raise ValueError(f'{col} must be a string or null')
        out[col] = out[col].map(lambda x: np.nan if pd.isna(x) or str(x).strip() == '' else str(x).strip())
    return out


def load_split(path):
    raw = pd.read_csv(path)
    if not raw['Churn'].isin(['Yes', 'No']).all():
        raise ValueError('Churn must contain only Yes/No labels')
    if raw['customerID'].duplicated().any():
        raise ValueError('Duplicate customer IDs found; review before splitting')
    X = raw[FEATURES]
    y = raw['Churn'].map({'No': 0, 'Yes': 1})
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
