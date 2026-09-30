"""Opt-in, local MLflow tracking of aggregate metrics and reproducibility metadata."""


def track_report(report: dict, output_dir: str, tracking_uri: str) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("telco-churn")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "seed": report["seed"],
                "dataset_sha256": report["dataset_sha256"],
                "selected_model": report["selected_model"],
                **report["forest_best_params"],
            }
        )
        for result in report["results"]:
            for split in ["cv", "holdout"]:
                for key, value in result[split].items():
                    if isinstance(value, (float, int)):
                        mlflow.log_metric(f"{result['model']}.{split}.{key}", value)
        mlflow.log_artifacts(str(output_dir), artifact_path="artifacts")
        return run.info.run_id
