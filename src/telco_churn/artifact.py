"""Versioned artifact loading. Integrity checks do not authenticate an untrusted publisher."""

import hashlib
import json
from pathlib import Path
from typing import Any
import joblib
import sklearn


def write_manifest(model_path: str | Path) -> None:
    path = Path(model_path)
    manifest = {
        "schema_version": 1,
        "sklearn_version": sklearn.__version__,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_model(model_path: str | Path) -> Any:
    """Check corruption/version mismatch before deserializing trusted local files."""
    path = Path(model_path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model manifest schema")
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            "Model sklearn version differs; install pinned dependencies or retrain"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("model_sha256"):
        raise ValueError(
            "Model checksum mismatch; restore a trusted artifact or retrain"
        )
    return joblib.load(path)
