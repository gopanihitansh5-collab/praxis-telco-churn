import sys
import pytest
from telco_churn import calibrate, cohort, paths, segments, train


def test_project_root_holds_the_package_and_report_dirs():
    root = paths.project_root()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()
    assert paths.reports_dir() == root / "reports"
    assert paths.artifacts_dir() == root / "artifacts"
    assert paths.data_dir() == root / "data"


def test_dataset_env_override_is_searched_first(monkeypatch, tmp_path):
    csv = tmp_path / "telco.csv"
    csv.write_text("customerID\n")
    monkeypatch.setenv(paths.DATA_ENV, str(csv))
    assert paths.resolve_dataset() == csv
    assert paths.dataset_candidates()[0] == csv


def test_explicit_dataset_beats_env_and_never_substitutes(monkeypatch, tmp_path):
    present = tmp_path / "telco.csv"
    present.write_text("customerID\n")
    monkeypatch.setenv(paths.DATA_ENV, str(present))
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        paths.resolve_dataset(tmp_path / "absent.csv")


def test_missing_dataset_names_the_fixing_command_and_locations(monkeypatch, tmp_path):
    # Fixed candidate list: the real kagglehub cache must not decide this test.
    monkeypatch.setattr(
        paths, "dataset_candidates", lambda: [tmp_path / "data" / "telco.csv"]
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.resolve_dataset()
    assert "download_data.py" in str(excinfo.value)
    assert "Searched:" in str(excinfo.value)


def test_model_env_override_is_honoured_even_when_calibrated(monkeypatch, tmp_path):
    override = tmp_path / "chosen.joblib"
    monkeypatch.setenv(paths.MODEL_ENV, str(override))
    # The env var names a file, so it wins over the calibrated/default name choice
    # instead of being silently dropped.
    assert paths.resolve_model(calibrated=True) == override
    assert paths.resolve_model(calibrated=False) == override
    assert paths.resolve_model(tmp_path / "explicit.joblib") == (
        tmp_path / "explicit.joblib"
    )


def test_model_defaults_without_env(monkeypatch):
    monkeypatch.delenv(paths.MODEL_ENV, raising=False)
    assert paths.resolve_model().name == "model.joblib"
    assert paths.resolve_model(calibrated=True).name == "model_calibrated.joblib"


@pytest.mark.parametrize("module", [segments, cohort, calibrate, train])
def test_commands_resolve_the_project_dataset_from_any_directory(
    module, monkeypatch, tmp_path
):
    """Each CLI must reach the project dataset with no arguments and no fitting."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(paths.DATA_ENV, raising=False)
    seen = {}

    def record(data_path, *rest):
        seen["data"] = data_path
        seen["rest"] = [str(item) for item in rest]
        raise RuntimeError("stop before any fitting")

    # calibrate and train take the resolved output directories as arguments; the
    # other two resolve inside main and hand only the dataset to load_split.
    attribute = {calibrate: "run", train: "train"}.get(module, "load_split")
    monkeypatch.setattr(module, attribute, record)
    monkeypatch.setattr(sys, "argv", [module.__name__])
    with pytest.raises(RuntimeError, match="stop before any fitting"):
        module.main()
    assert seen["data"] == paths.resolve_dataset()
    assert all(item.startswith(str(paths.project_root())) for item in seen["rest"])
