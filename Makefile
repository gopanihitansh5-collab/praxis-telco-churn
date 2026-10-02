# Convenience targets. Everything here is also a plain command documented in README.md.
# Requires Python 3.10 or 3.11 on PATH. Windows users without make: use .\setup.ps1

PY ?= python
VENV := .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif
VPY := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup data predict serve test lint reports policy clean verify all

help:
	@echo "setup    Create .venv and install pinned dependencies"
	@echo "data     Download the Kaggle dataset to data/telco.csv (needs Kaggle credentials)"
	@echo "predict  Score sample_customer.json with the bundled model"
	@echo "serve    Start the local inference API on 127.0.0.1:8000"
	@echo "test     Run the test suite"
	@echo "lint     Run ruff check and format check"
	@echo "reports  Regenerate calibration, policy, segment and cohort reports (needs data)"
	@echo "verify   setup + test + lint + predict, the full offline check"
	@echo "all      verify + data + reports"

setup:
	$(PY) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r requirements-dev.txt
	$(VPY) -m pip install -e .
	@echo "Done. Activate with: source $(BIN)/activate  (Windows: $(BIN)\\activate)"

data:
	$(VPY) download_data.py

predict:
	$(VPY) predict.py --input sample_customer.json

serve:
	$(BIN)/uvicorn telco_churn.api:app --host 127.0.0.1 --port 8000

test:
	$(VPY) -m pytest -q

lint:
	$(VPY) -m ruff check src tests predict.py download_data.py
	$(VPY) -m ruff format --check src tests predict.py download_data.py

reports:
	$(VPY) -m telco_churn.calibrate
	$(VPY) -m telco_churn.segments
	$(VPY) -m telco_churn.cohort
	$(VPY) -m telco_churn.policy --probabilities artifacts/oof_probabilities.csv
	$(VPY) -m telco_churn.score_batch

verify: setup test lint predict

all: verify data reports

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
