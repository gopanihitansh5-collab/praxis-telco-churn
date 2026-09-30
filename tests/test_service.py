import json
from pathlib import Path
import shutil
import pytest
from fastapi.testclient import TestClient
from telco_churn.api import create_app
from telco_churn.artifact import load_model
from telco_churn.json_input import parse_customer


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def test_health_and_prediction(client):
    assert client.get("/health").json() == {"status": "ready"}
    payload = json.loads(Path("sample_customer.json").read_text())
    result = client.post("/predict", json=payload)
    assert result.status_code == 200
    assert 0 <= result.json()["churn_probability"] <= 1


@pytest.mark.parametrize(
    "text", ["{}", "[]", '{"tenure":NaN}', '{"tenure":1,"tenure":2}', "{broken"]
)
def test_invalid_json(client, text):
    assert (
        client.post(
            "/predict", content=text, headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )


def test_size_and_content_type(client):
    assert client.post("/predict", content="x").status_code == 415
    assert (
        client.post(
            "/predict",
            content="x" * 270000,
            headers={"Content-Type": "application/json"},
        ).status_code
        == 413
    )


def test_startup_fails_without_model(tmp_path):
    with pytest.raises(OSError):
        with TestClient(create_app(str(tmp_path / "missing.joblib"))):
            pass


def test_checksum_rejects_corruption(tmp_path):
    path = tmp_path / "model.joblib"
    shutil.copyfile("artifacts/model.joblib", path)
    shutil.copyfile("artifacts/model.manifest.json", path.with_suffix(".manifest.json"))
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_model(path)


def test_version_rejects_mismatch(tmp_path):
    path = tmp_path / "model.joblib"
    shutil.copyfile("artifacts/model.joblib", path)
    manifest = json.loads(Path("artifacts/model.manifest.json").read_text())
    manifest["sklearn_version"] = "0.0.0"
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        load_model(path)


def test_strict_json_duplicate():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_customer('{"a":1,"a":2}')
