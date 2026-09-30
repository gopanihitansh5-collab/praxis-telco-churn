# Data source

Kaggle: **Telco Customer Churn**, uploaded by blastchar.
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The assessment names this dataset. `download_data.py` downloads version 1 and
copies `WA_Fn-UseC_-Telco-Customer-Churn.csv` to `data/telco.csv`.
The CSV is excluded from version control. Do not substitute private customer data.

Expected CSV SHA-256:
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

The downloaded CSV contains 7,043 rows and 21 columns. `customerID` is excluded
from training. `Churn` is mapped No=0 / Yes=1.
