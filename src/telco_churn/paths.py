"""Working-directory-independent path resolution; no network access and no downloads."""

import os
from pathlib import Path

MARKERS = ("pyproject.toml", ".git")
DATASET_NAME = "telco.csv"
KAGGLE_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_ENV = "CHURN_DATA"
MODEL_ENV = "CHURN_MODEL_PATH"

HINT = (
    "Dataset not found. Fetch it once with:\n"
    "    python download_data.py\n"
    "or point the code at an existing copy:\n"
    "    set CHURN_DATA=C:\\path\\to\\telco.csv     (Windows)\n"
    "    export CHURN_DATA=/path/to/telco.csv       (macOS/Linux)\n"
    "The CSV is deliberately not committed; see data/README.md."
)


def project_root(start: str | Path | None = None) -> Path:
    """Walk upward to the package root so commands work from any directory."""
    here = Path(start) if start else Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if candidate.is_dir() and any((candidate / m).exists() for m in MARKERS):
            return candidate
    # Installed without the source tree alongside it; fall back to the caller's directory.
    return Path.cwd()


def data_dir() -> Path:
    return project_root() / "data"


def reports_dir() -> Path:
    return project_root() / "reports"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def dataset_candidates() -> list[Path]:
    """Ordered search locations, most explicit first."""
    found = []
    env = os.environ.get(DATA_ENV)
    if env:
        found.append(Path(env).expanduser())
    found += [
        data_dir() / DATASET_NAME,
        Path.cwd() / "data" / DATASET_NAME,
        Path.cwd() / DATASET_NAME,
    ]
    cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "blastchar"
    if cache.is_dir():
        found += sorted(cache.rglob(KAGGLE_CSV))
    return found


def resolve_dataset(explicit: str | Path | None = None, required: bool = True) -> Path:
    """Return the first dataset that exists, or raise with the command that fixes it."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        # An explicit path is a user instruction: never silently substitute another file.
        if path.exists():
            return path
        if required:
            raise FileNotFoundError(f"Dataset not found at {path}.\n\n{HINT}")
        return path
    for candidate in dataset_candidates():
        if candidate.exists():
            return candidate
    if required:
        searched = "\n".join(f"  - {c}" for c in dataset_candidates())
        raise FileNotFoundError(f"{HINT}\n\nSearched:\n{searched}")
    return data_dir() / DATASET_NAME


def resolve_model(explicit: str | Path | None = None, calibrated: bool = False) -> Path:
    """Resolve a model artifact; the manifest beside it is loaded by artifact.load_model.

    Precedence is explicit path, then `$CHURN_MODEL_PATH`, then the default or
    calibrated artifact name. The environment variable names a file, so it also
    overrides `calibrated=True`: a deliberate override is never silently dropped.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get(MODEL_ENV)
    if env:
        return Path(env).expanduser()
    name = "model_calibrated.joblib" if calibrated else "model.joblib"
    local = Path.cwd() / "artifacts" / name
    return local if local.exists() else artifacts_dir() / name
