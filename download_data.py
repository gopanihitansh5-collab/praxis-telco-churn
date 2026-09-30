"""Download the assessment's named Kaggle dataset, version 1."""

from pathlib import Path
import shutil
import kagglehub

path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn/versions/1"))
source = path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
Path("data").mkdir(exist_ok=True)
shutil.copyfile(source, "data/telco.csv")
print("Saved data/telco.csv")
