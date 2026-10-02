"""Fetch the assessment's named Kaggle dataset, or adopt an already-downloaded copy."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

# Intentionally self-contained: this is the first command a new user runs, before
# `pip install -e .`, so it must not import the telco_churn package.
DATASET = "blastchar/telco-customer-churn/versions/1"
KAGGLE_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"

CREDENTIAL_HELP = """
Kaggle credentials are required to download. Either:

  1. Put kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
     or ~/.kaggle/kaggle.json (macOS/Linux). Get it from
     https://www.kaggle.com/settings -> API -> Create New Token

  2. Or set environment variables:
       set KAGGLE_USERNAME=... && set KAGGLE_KEY=...        (Windows)
       export KAGGLE_USERNAME=... KAGGLE_KEY=...            (macOS/Linux)

No Kaggle account? Download the CSV manually from
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
then adopt it without any network call:

    python download_data.py --csv path/to/downloaded.csv
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    actual = sha256(target)
    if actual != EXPECTED_SHA256:
        # Not fatal: Kaggle may publish a revision. Surface it rather than hide it,
        # because artifacts/metrics.json records the hash the model was trained on.
        print(
            f"WARNING: SHA-256 {actual}\n"
            f"         expected {EXPECTED_SHA256}\n"
            "         Metrics reproduced from this file may differ from the committed run.",
            file=sys.stderr,
        )
    else:
        print(f"Checksum verified: {actual}")
    return target


def from_kaggle() -> Path:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Run: python -m pip install -r requirements.txt"
        )
    if not (
        (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    ):
        raise SystemExit(CREDENTIAL_HELP)
    try:
        cached = Path(kagglehub.dataset_download(DATASET))
    except Exception as exc:
        raise SystemExit(f"Kaggle download failed: {exc}\n{CREDENTIAL_HELP}") from exc
    found = next(iter(sorted(cached.rglob(KAGGLE_CSV))), None)
    if found is None:
        raise SystemExit(f"{KAGGLE_CSV} not found inside the downloaded dataset.")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", help="Adopt an already-downloaded CSV instead of calling Kaggle"
    )
    parser.add_argument(
        "--output", default=None, help="Destination (default data/telco.csv)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing copy"
    )
    args = parser.parse_args()

    default = Path(__file__).resolve().parent / "data" / "telco.csv"
    target = Path(args.output) if args.output else default
    if target.exists() and not args.force:
        print(f"{target} already exists. Use --force to replace it.")
        print(f"Checksum: {sha256(target)}")
        return

    if args.csv:
        source = Path(args.csv).expanduser()
        if not source.exists():
            raise SystemExit(f"No such file: {source}")
    else:
        source = from_kaggle()

    install(source, target)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
