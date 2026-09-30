"""Bounded local API. No customer payloads or identifiers are logged."""

from contextlib import asynccontextmanager
import json
import os
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from .inference import Predictor
from .json_input import parse_customer
from .schema import Customer

MAX_BODY_BYTES = 262_144


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customers: list[Customer] = Field(min_length=1, max_length=100)


async def body(request: Request) -> dict:
    if (
        request.headers.get("content-type", "").split(";")[0].strip()
        != "application/json"
    ):
        raise HTTPException(415, "Content-Type must be application/json")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BODY_BYTES:
            raise HTTPException(413, "Payload exceeds 256 KiB")
    try:
        return parse_customer(content.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, str(exc)) from exc


def create_app(
    model_path: str = "artifacts/model.joblib", explanations: bool = False
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = Predictor(model_path)
        app.state.explainer = None
        if explanations:
            from .explain import Explainer

            app.state.explainer = Explainer(app.state.predictor.model)
        yield

    app = FastAPI(title="Telco Churn Inference", version="0.1.0", lifespan=lifespan)

    def predict_one(customer: Customer) -> dict:
        payload = customer.model_dump()
        result = app.state.predictor.predict(payload)
        if app.state.explainer is not None:
            import pandas as pd

            result["explanation"] = app.state.explainer.explain(pd.DataFrame([payload]))
        return result

    def validate(schema, payload):
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            # JSON serialization removes ValueError objects from Pydantic context.
            raise HTTPException(422, json.loads(exc.json(include_input=False))) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/predict")
    async def predict(request: Request) -> dict:
        return predict_one(validate(Customer, await body(request)))

    @app.post("/predict/batch")
    async def predict_batch(request: Request) -> dict:
        batch = validate(Batch, await body(request))
        return {"predictions": [predict_one(customer) for customer in batch.customers]}

    # Custom parsing is needed for duplicate-key rejection; still expose typed OpenAPI contracts.
    app.openapi_schema = None
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    for path, contract in [("/predict", Customer), ("/predict/batch", Batch)]:
        schema["paths"][path]["post"]["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": contract.model_json_schema()}},
        }
    # Batch schema contains its own local $defs; references must resolve inside the embedded schema.
    batch_schema = schema["paths"]["/predict/batch"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    batch_schema["properties"]["customers"]["items"] = Customer.model_json_schema()
    batch_schema.pop("$defs", None)
    app.openapi_schema = schema
    return app


app = create_app(
    os.environ.get("CHURN_MODEL_PATH", "artifacts/model.joblib"),
    os.environ.get("CHURN_EXPLAIN", "0") == "1",
)
