import mlflow


class MLflowKerasCallback:
    """
    Custom Keras callback that logs per-epoch training metrics to MLflow.

    Logs: train loss, val loss, train accuracy, val accuracy, val_macro_recall, learning rate.

    Usage:
        cb = MLflowKerasCallback()
        model.fit(..., callbacks=[cb.get_callback()])
    """

    def __init__(self):
        self._callback = None

    def get_callback(self):
        """Returns a LambdaCallback that logs metrics to the active MLflow run."""
        import tensorflow as tf

        def on_epoch_end(epoch, logs):
            if not mlflow.active_run():
                return

            metrics_to_log = {}

            # Standard Keras metrics
            for key in ["loss", "accuracy", "val_loss", "val_accuracy",
                        "categorical_accuracy", "val_categorical_accuracy"]:
                if key in logs:
                    metrics_to_log[key] = float(logs[key])

            # Custom macro recall from MacroRecallCallback
            if "val_macro_recall" in logs:
                metrics_to_log["val_macro_recall"] = float(logs["val_macro_recall"])

            # Learning rate
            if "lr" in logs:
                metrics_to_log["learning_rate"] = float(logs["lr"])

            mlflow.log_metrics(metrics_to_log, step=epoch)

        self._callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=on_epoch_end
        )
        return self._callback


class MLflowTracker:
    """
    Manages an MLflow experiment lifecycle for a training run.

    Handles:
    - Creating/setting experiment
    - Starting/ending runs
    - Logging params, metrics, artifacts, and tags
    """

    def __init__(self, experiment_name: str = "kidney-disease-classification",
                 tracking_uri: str = None):
        import os

        if tracking_uri is None:
            # Store mlruns inside training_model/ directory
            training_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            tracking_uri = os.path.join(training_root, "mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        self._ensure_experiment()

    def _ensure_experiment(self):
        """Create the experiment if it doesn't exist."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str, tags: dict = None):
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)

    def log_params_from_yaml(self, params: dict):
        """Log all parameters from a params dict (flattened)."""
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, list):
                flat_params[key] = str(value)
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_params[f"{key}.{sub_key}"] = str(sub_val)
            else:
                flat_params[key] = str(value)

        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log a dictionary of metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_evaluation_metrics(self, eval_data: dict):
        """Log structured evaluation results from ModelTrainer.evaluate()."""
        # Overall metrics
        if "overall_accuracy" in eval_data:
            mlflow.log_metric("test_accuracy", eval_data["overall_accuracy"])

        # Macro metrics
        macro = eval_data.get("macro_metrics", {})
        for key, val in macro.items():
            mlflow.log_metric(f"test_macro_{key}", val)

        # Weighted metrics
        weighted = eval_data.get("weighted_metrics", {})
        for key, val in weighted.items():
            mlflow.log_metric(f"test_weighted_{key}", val)

        # Per-class metrics
        per_class = eval_data.get("per_class_metrics", {})
        for class_name, class_metrics in per_class.items():
            for metric_name, metric_val in class_metrics.items():
                if metric_name != "support":
                    mlflow.log_metric(f"test_{class_name.lower()}_{metric_name}", metric_val)

    def log_artifact(self, file_path: str, artifact_path: str = None):
        """Log a file as an MLflow artifact."""
        import os
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path)

    def log_artifacts_dir(self, dir_path: str, artifact_path: str = None):
        """Log an entire directory as MLflow artifacts."""
        import os
        if os.path.exists(dir_path):
            mlflow.log_artifacts(dir_path, artifact_path)

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        mlflow.end_run(status=status)

    def get_run_id(self) -> str:
        """Get the current active run ID."""
        run = mlflow.active_run()
        return run.info.run_id if run else None

    @staticmethod
    def get_all_runs(experiment_name: str = "kidney-disease-classification") -> list:
        """Retrieve all runs for the experiment, sorted by start time descending."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        import mlflow.search
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        return runs
