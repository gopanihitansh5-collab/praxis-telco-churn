import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from telco_churn.api import create_app
from telco_churn.inference import Predictor


@pytest.fixture
def payload():
    return json.loads(Path("sample_customer.json").read_text())


def test_batch_matches_single(payload):
    with TestClient(create_app()) as client:
        single = client.post("/predict", json=payload).json()
        result = client.post("/predict/batch", json={"customers": [payload, payload]})
        assert result.status_code == 200
        assert result.json()["predictions"] == [single, single]
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422
        assert (
            client.post(
                "/predict/batch", json={"customers": [payload] * 101}
            ).status_code
            == 422
        )
        assert (
            client.post("/predict/batch", json={"customers": [payload, {}]}).status_code
            == 422
        )


@pytest.mark.parametrize(
    "key,value",
    [("tenure", True), ("gender", 1), ("customerID", 42), ("MonthlyCharges", -1)],
)
def test_pydantic_rejects_invalid(payload, key, value):
    payload[key] = value
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)


def test_shap_additive_and_id_independent(payload):
    pytest.importorskip("shap")
    from telco_churn.explain import Explainer

    predictor = Predictor()
    explainer = Explainer(predictor.model)
    explanation = explainer.explain(pd.DataFrame([payload]))
    probability = predictor.predict(payload)["churn_probability"]
    assert np.isclose(
        explanation["baseline_probability"] + explanation["all_contributions_sum"],
        probability,
    )
    assert len(explanation["top_drivers"]) == 5
    assert "customerID" not in [d["feature"] for d in explanation["top_drivers"]]
    with TestClient(create_app(explanations=True)) as client:
        assert "explanation" in client.post("/predict", json=payload).json()


def test_mlflow_local_tracking(tmp_path):
    pytest.importorskip("mlflow")
    from telco_churn.tracking import track_report
    from mlflow.tracking import MlflowClient

    report = json.loads(Path("artifacts/metrics.json").read_text())
    uri = (tmp_path / "mlruns").as_uri()
    run_id = track_report(report, "artifacts", uri)
    run = MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.params["dataset_sha256"] == report["dataset_sha256"]
    assert (
        run.data.metrics["random_forest_tuned.holdout.roc_auc"]
        == report["results"][1]["holdout"]["roc_auc"]
    )


def test_drift_identical_and_shifted():
    from telco_churn.drift import make_reference, compare

    customer = json.loads(Path("sample_customer.json").read_text())
    X = pd.DataFrame([dict(customer, MonthlyCharges=float(i + 1)) for i in range(200)])
    reference = make_reference(X)
    same = compare(reference, X)
    assert same["features"]["MonthlyCharges"]["psi"] == 0
    shifted = X.copy()
    shifted["MonthlyCharges"] = 1000
    changed = compare(reference, shifted)
    assert changed["features"]["MonthlyCharges"]["psi"] > 0.2
    assert changed["features"]["MonthlyCharges"]["review_flag"] is True
    with pytest.raises(ValueError, match="100"):
        compare(reference, X.iloc[:10])


def test_threshold_selection_is_deterministic():
    from telco_churn.threshold import choose_threshold

    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = choose_threshold(y, probabilities)
    assert 0.2 < threshold <= 0.3
    assert threshold == choose_threshold(y, probabilities)


def test_threshold_report_preserves_boundary():
    report = json.loads(Path("reports/threshold_experiment.json").read_text())
    assert report["holdout_used"] is False
    assert report["default_model_changed"] is False
    assert report["rows"] == 5634
    assert len(report["outer_fold_thresholds"]) == 3


def test_openapi_raw_contract():
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()
        contract = schema["paths"]["/predict"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        assert "tenure" in contract["required"]
        assert contract["additionalProperties"] is False
