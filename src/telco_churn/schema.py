"""Raw customer contract. Pre-validation rejects coercion of invalid scalar types."""

from typing import Any
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from .data import clean_features


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customerID: str | None = None
    gender: str | None
    SeniorCitizen: int | float | str | None
    Partner: str | None
    Dependents: str | None
    tenure: int | float | str | None
    PhoneService: str | None
    MultipleLines: str | None
    InternetService: str | None
    OnlineSecurity: str | None
    OnlineBackup: str | None
    DeviceProtection: str | None
    TechSupport: str | None
    StreamingTV: str | None
    StreamingMovies: str | None
    Contract: str | None
    PaperlessBilling: str | None
    PaymentMethod: str | None
    MonthlyCharges: int | float | str | None
    TotalCharges: int | float | str | None

    @model_validator(mode="before")
    @classmethod
    def check_raw(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("Input must be one JSON object")
        clean_features(pd.DataFrame([value]))
        return value
