FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 appuser
COPY artifacts/model.joblib artifacts/model.manifest.json ./artifacts/
USER appuser
EXPOSE 8000
CMD ["uvicorn", "telco_churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
