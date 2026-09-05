import os
import sys
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path

# Ensure training_model root is on sys.path so we can import training.train_model
TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)

# NOTE: ModelTrainer is imported lazily in run() to avoid loading TensorFlow
# for lightweight CLI commands (list-runs, compare, promote, etc.)



class TrainingPipeline:
    """
    MLOps-aware training pipeline that wraps ModelTrainer with:
    - MLflow experiment tracking (params, per-epoch metrics, evaluation metrics, artifacts)
    - Run-level artifact archival (params snapshot, model, scores, confusion matrix)
    - Local model registry with version management
    - Production model promotion with backup
    """

    RUNS_DIR = os.path.join(TRAINING_ROOT, "runs")
    REGISTRY_PATH = os.path.join(TRAINING_ROOT, "model_registry.json")

    def __init__(
        self,
        run_name: str = None,
        dataset_dir: str = None,
        params_path: str = None,
    ):
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.dataset_dir = dataset_dir or os.path.join(
            TRAINING_ROOT, "..", "artifacts", "data_ingestion", "Kindey_Stone_Dataset"
        )
        self.params_path = params_path or os.path.join(TRAINING_ROOT, "params.yaml")

        # Create a unique run directory
        self.run_dir = os.path.join(self.RUNS_DIR, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "artifacts"), exist_ok=True)

        self._load_params()

        # MLflow tracker (lazy-loaded)
        self._mlflow_tracker = None
        self._mlflow_available = False
        self._check_mlflow()

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------

    def _check_mlflow(self):
        """Check if MLflow is available."""
        try:
            import mlflow
            self._mlflow_available = True
        except ImportError:
            self._mlflow_available = False
            print("[INFO] MLflow not installed — running without experiment tracking.")
            print("       Install with: pip install mlflow")

    def _init_mlflow(self):
        """Initialize MLflow tracker if available."""
        if not self._mlflow_available:
            return

        from training.mlflow_callback import MLflowTracker
        self._mlflow_tracker = MLflowTracker(
            experiment_name="kidney-disease-classification"
        )

    def _get_mlflow_callback(self):
        """Get the MLflow Keras callback for per-epoch logging."""
        if not self._mlflow_available:
            return None

        from training.mlflow_callback import MLflowKerasCallback
        cb = MLflowKerasCallback()
        return cb.get_callback()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_params(self):
        """Load and snapshot the params file for this run."""
        with open(self.params_path, "r") as f:
            self.params = yaml.safe_load(f)

        # Save a copy of params into the run directory for reproducibility
        snapshot_path = os.path.join(self.run_dir, "params_snapshot.yaml")
        with open(snapshot_path, "w") as f:
            yaml.dump(self.params, f, default_flow_style=False)

    def _save_run_metadata(self, status: str, metrics: dict = None, error: str = None):
        """Persist run metadata as a JSON file inside the run directory."""
        meta = {
            "run_name": self.run_name,
            "status": status,
            "dataset_dir": os.path.abspath(self.dataset_dir),
            "params_path": os.path.abspath(self.params_path),
            "started_at": getattr(self, "_started_at", None),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mlflow_run_id": (
                self._mlflow_tracker.get_run_id()
                if self._mlflow_tracker else None
            ),
            "params": self.params,
            "metrics": metrics,
            "error": error,
        }
        meta_path = os.path.join(self.run_dir, "run_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
        return meta

    def _register_model(self, metrics: dict, model_path: str):
        """Add a new entry to the local model registry."""
        registry = []
        if os.path.exists(self.REGISTRY_PATH):
            with open(self.REGISTRY_PATH, "r") as f:
                registry = json.load(f)

        version = len(registry) + 1
        entry = {
            "version": version,
            "run_name": self.run_name,
            "model_path": os.path.abspath(model_path),
            "run_dir": os.path.abspath(self.run_dir),
            "mlflow_run_id": (
                self._mlflow_tracker.get_run_id()
                if self._mlflow_tracker else None
            ),
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "is_production": False,
        }
        registry.append(entry)

        with open(self.REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=4)

        print(f"\n[REGISTRY] Model registered as version {version}")
        return version

    def _copy_artifacts_to_run(self, trainer):
        """Copy training artifacts into the run directory for archival."""
        artifacts_dest = os.path.join(self.run_dir, "artifacts")

        # Copy the trained model
        if os.path.exists(trainer.model_save_path):
            shutil.copy2(trainer.model_save_path, os.path.join(artifacts_dest, "model.h5"))

        # Copy scores
        scores_path = os.path.join("logs", "scores.json")
        if os.path.exists(scores_path):
            shutil.copy2(scores_path, os.path.join(artifacts_dest, "scores.json"))

        # Copy confusion matrix
        cm_path = os.path.join("logs", "confusion_matrix.json")
        if os.path.exists(cm_path):
            shutil.copy2(cm_path, os.path.join(artifacts_dest, "confusion_matrix.json"))

        # Copy class names
        cn_path = os.path.join("model", "class_names.json")
        if os.path.exists(cn_path):
            shutil.copy2(cn_path, os.path.join(artifacts_dest, "class_names.json"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full training pipeline: build → train → fine-tune → evaluate → track → register."""
        # Lazy import to avoid loading TensorFlow for non-training commands
        from training.train_model import ModelTrainer

        self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("=" * 60)
        print(f"  TRAINING PIPELINE: {self.run_name}")
        print(f"  Dataset: {self.dataset_dir}")
        print(f"  Params:  {self.params_path}")
        print(f"  MLflow:  {'Enabled' if self._mlflow_available else 'Disabled'}")
        print("=" * 60)

        # Initialize MLflow
        self._init_mlflow()

        try:
            # Start MLflow run
            if self._mlflow_tracker:
                self._mlflow_tracker.start_run(
                    run_name=self.run_name,
                    tags={
                        "dataset": os.path.basename(self.dataset_dir),
                        "model_type": self.params.get("MODEL_TYPE", "vgg16"),
                        "pipeline_version": "2.0",
                    }
                )
                # Log all hyperparameters
                self._mlflow_tracker.log_params_from_yaml(self.params)

            # Instantiate the existing ModelTrainer
            trainer = ModelTrainer(
                params_path=self.params_path,
                dataset_dir=self.dataset_dir,
            )

            # Stage 0: Build model & data generators
            print("\n[PIPELINE] Building model...")
            base_model, model = trainer.build_vgg16_model()

            print("[PIPELINE] Creating data generators...")
            trainer.create_data_generators()

            class_weight_dict = trainer.calculate_class_weights()

            # Log class weights to MLflow
            if self._mlflow_tracker:
                for cls_idx, weight in class_weight_dict.items():
                    self._mlflow_tracker.log_metrics(
                        {f"class_weight_{cls_idx}": weight}
                    )

            # Get MLflow Keras callback
            mlflow_cb = self._get_mlflow_callback()

            # Inject MLflow callback into trainer's callbacks by monkey-patching
            # We do this by wrapping train_stage1 and fine_tune_stage2
            original_train_stage1 = trainer.train_stage1
            original_fine_tune_stage2 = trainer.fine_tune_stage2

            def patched_train_stage1(model, class_weight_dict):
                """Wraps Stage 1 to inject MLflow callback."""
                if mlflow_cb:
                    # We need to add the callback to the model.fit call
                    # Store original fit method and wrap it
                    import tensorflow as tf
                    original_fit = model.fit

                    def fit_with_mlflow(*args, **kwargs):
                        callbacks = kwargs.get("callbacks", []) or []
                        callbacks.append(mlflow_cb)
                        kwargs["callbacks"] = callbacks
                        return original_fit(*args, **kwargs)

                    model.fit = fit_with_mlflow

                result = original_train_stage1(model, class_weight_dict)

                if self._mlflow_tracker:
                    self._mlflow_tracker.log_metrics({"stage": 1})

                return result

            def patched_fine_tune_stage2(base_model, model, class_weight_dict):
                """Wraps Stage 2 to inject MLflow callback."""
                if mlflow_cb:
                    import tensorflow as tf
                    original_fit = model.fit

                    def fit_with_mlflow(*args, **kwargs):
                        callbacks = kwargs.get("callbacks", []) or []
                        callbacks.append(mlflow_cb)
                        kwargs["callbacks"] = callbacks
                        return original_fit(*args, **kwargs)

                    model.fit = fit_with_mlflow

                result = original_fine_tune_stage2(base_model, model, class_weight_dict)

                if self._mlflow_tracker:
                    self._mlflow_tracker.log_metrics({"stage": 2})

                return result

            # Stage 1: Classifier head training
            print("\n[PIPELINE] Stage 1 — Training classifier head...")
            patched_train_stage1(model, class_weight_dict)

            # Stage 2: Fine-tuning
            print("\n[PIPELINE] Stage 2 — Fine-tuning backbone block 5...")
            patched_fine_tune_stage2(base_model, model, class_weight_dict)

            # Stage 3: Evaluation
            print("\n[PIPELINE] Evaluating on test split...")
            eval_metrics = trainer.evaluate()

            # Log evaluation metrics to MLflow
            if self._mlflow_tracker:
                self._mlflow_tracker.log_evaluation_metrics(eval_metrics)

            # Stage 4: Archive artifacts for this run
            print("\n[PIPELINE] Archiving run artifacts...")
            self._copy_artifacts_to_run(trainer)

            # Log artifacts to MLflow
            if self._mlflow_tracker:
                artifacts_dir = os.path.join(self.run_dir, "artifacts")
                for fname in os.listdir(artifacts_dir):
                    fpath = os.path.join(artifacts_dir, fname)
                    if os.path.isfile(fpath):
                        self._mlflow_tracker.log_artifact(fpath)

                # Also log the params snapshot
                params_snapshot = os.path.join(self.run_dir, "params_snapshot.yaml")
                self._mlflow_tracker.log_artifact(params_snapshot)

            # Stage 5: Register model version
            self._register_model(
                metrics=eval_metrics,
                model_path=trainer.model_save_path,
            )

            # Save run metadata
            self._save_run_metadata(status="SUCCESS", metrics=eval_metrics)

            # End MLflow run
            if self._mlflow_tracker:
                self._mlflow_tracker.end_run(status="FINISHED")

            print("\n" + "=" * 60)
            print(f"  ✓ PIPELINE COMPLETE: {self.run_name}")
            print(f"  Run directory: {self.run_dir}")
            print(f"  Accuracy: {eval_metrics.get('overall_accuracy', 'N/A')}")
            macro = eval_metrics.get("macro_metrics", {})
            print(f"  Macro Recall: {macro.get('recall', 'N/A')}")
            print(f"  Macro F1: {macro.get('f1_score', 'N/A')}")
            if self._mlflow_available:
                print(f"\n  View in MLflow UI:")
                print(f"    cd training_model && mlflow ui")
            print("=" * 60)

            return eval_metrics

        except Exception as e:
            self._save_run_metadata(status="FAILED", error=str(e))
            if self._mlflow_tracker:
                self._mlflow_tracker.end_run(status="FAILED")
            print(f"\n[PIPELINE ERROR] {e}")
            raise

    # ------------------------------------------------------------------
    # Static utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def list_runs():
        """List all completed training runs."""
        runs_dir = os.path.join(TRAINING_ROOT, "runs")
        if not os.path.exists(runs_dir):
            print("No runs found.")
            return []

        runs = []
        for run_name in sorted(os.listdir(runs_dir)):
            meta_path = os.path.join(runs_dir, run_name, "run_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                runs.append(meta)

        if not runs:
            print("No runs found.")
            return []

        print(f"\n{'Run Name':<30} {'Status':<10} {'Accuracy':<12} {'Macro Recall':<14} {'MLflow ID':<15} {'Finished At'}")
        print("-" * 110)
        for r in runs:
            metrics = r.get("metrics") or {}
            macro = metrics.get("macro_metrics", {})
            acc = metrics.get("overall_accuracy", "—")
            recall = macro.get("recall", "—")
            mlflow_id = (r.get("mlflow_run_id") or "—")[:12]
            print(
                f"{r['run_name']:<30} "
                f"{r['status']:<10} "
                f"{str(acc):<12} "
                f"{str(recall):<14} "
                f"{mlflow_id:<15} "
                f"{r.get('finished_at', '—')}"
            )

        return runs

    @staticmethod
    def compare_models():
        """Compare all registered model versions side by side."""
        registry_path = os.path.join(TRAINING_ROOT, "model_registry.json")
        if not os.path.exists(registry_path):
            print("No models registered yet.")
            return []

        with open(registry_path, "r") as f:
            registry = json.load(f)

        if not registry:
            print("No models registered yet.")
            return []

        print(f"\n{'Ver':<5} {'Run Name':<25} {'Accuracy':<12} {'Macro Recall':<14} {'Macro F1':<12} {'Prod':<6} {'Registered At'}")
        print("-" * 105)
        for entry in registry:
            m = entry.get("metrics", {})
            macro = m.get("macro_metrics", {})
            print(
                f"{entry['version']:<5} "
                f"{entry['run_name']:<25} "
                f"{str(m.get('overall_accuracy', '—')):<12} "
                f"{str(macro.get('recall', '—')):<14} "
                f"{str(macro.get('f1_score', '—')):<12} "
                f"{'  ✓' if entry.get('is_production') else '  —':<6} "
                f"{entry.get('registered_at', '—')}"
            )

        return registry

    @staticmethod
    def promote_model(version: int):
        """Promote a registered model version to production (copies to backend/model/model1.h5)."""
        registry_path = os.path.join(TRAINING_ROOT, "model_registry.json")
        if not os.path.exists(registry_path):
            print("No models registered yet.")
            return False

        with open(registry_path, "r") as f:
            registry = json.load(f)

        target = None
        for entry in registry:
            if entry["version"] == version:
                target = entry
                break

        if not target:
            print(f"[ERROR] Version {version} not found in registry.")
            return False

        source_model = target["model_path"]
        if not os.path.exists(source_model):
            # Fallback: check the run artifacts directory
            run_dir = target.get("run_dir", "")
            run_model = os.path.join(run_dir, "artifacts", "model.h5")
            if os.path.exists(run_model):
                source_model = run_model
            else:
                print(f"[ERROR] Model file not found at {source_model}")
                return False

        prod_model_path = os.path.abspath(
            os.path.join(TRAINING_ROOT, "..", "backend", "model", "model1.h5")
        )

        # Backup current production model
        if os.path.exists(prod_model_path):
            backup_path = prod_model_path.replace(
                ".h5", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            )
            shutil.copy2(prod_model_path, backup_path)
            print(f"[BACKUP] Current production model backed up to:")
            print(f"  {backup_path}")

        # Copy new model to production
        shutil.copy2(source_model, prod_model_path)

        # Also copy class_names.json if available
        run_dir = target.get("run_dir", "")
        cn_src = os.path.join(run_dir, "artifacts", "class_names.json")
        cn_dest = os.path.join(TRAINING_ROOT, "..", "backend", "model", "class_names.json")
        if os.path.exists(cn_src):
            shutil.copy2(cn_src, cn_dest)

        # Update registry flags
        for entry in registry:
            entry["is_production"] = (entry["version"] == version)

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)

        print(f"\n[PROMOTED] Version {version} ({target['run_name']}) is now the production model.")
        print(f"  → {prod_model_path}")

        # Log promotion in MLflow if available
        try:
            import mlflow
            if target.get("mlflow_run_id"):
                with mlflow.start_run(run_id=target["mlflow_run_id"]):
                    mlflow.set_tag("promoted_to_production", "true")
                    mlflow.set_tag(
                        "promoted_at",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
        except Exception:
            pass  # MLflow tagging is best-effort

        return True

    @staticmethod
    def show_run_details(run_name: str):
        """Show detailed metrics and artifacts for a specific run."""
        meta_path = os.path.join(TRAINING_ROOT, "runs", run_name, "run_meta.json")
        if not os.path.exists(meta_path):
            print(f"[ERROR] Run '{run_name}' not found.")
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        print(f"\n{'=' * 60}")
        print(f"  Run: {meta['run_name']}")
        print(f"  Status: {meta['status']}")
        print(f"  Started: {meta.get('started_at', '—')}")
        print(f"  Finished: {meta.get('finished_at', '—')}")
        print(f"  Dataset: {meta.get('dataset_dir', '—')}")
        print(f"  MLflow Run ID: {meta.get('mlflow_run_id', '—')}")
        print(f"{'=' * 60}")

        metrics = meta.get("metrics", {})
        if metrics:
            print(f"\n  Overall Accuracy: {metrics.get('overall_accuracy', '—')}")

            macro = metrics.get("macro_metrics", {})
            print(f"  Macro Precision: {macro.get('precision', '—')}")
            print(f"  Macro Recall: {macro.get('recall', '—')}")
            print(f"  Macro F1: {macro.get('f1_score', '—')}")

            per_class = metrics.get("per_class_metrics", {})
            if per_class:
                print(f"\n  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
                print(f"  {'-' * 55}")
                for cls, m in per_class.items():
                    print(
                        f"  {cls:<10} "
                        f"{str(m.get('precision', '—')):<12} "
                        f"{str(m.get('recall', '—')):<12} "
                        f"{str(m.get('f1_score', '—')):<12} "
                        f"{str(m.get('support', '—')):<10}"
                    )

        # List artifacts
        artifacts_dir = os.path.join(TRAINING_ROOT, "runs", run_name, "artifacts")
        if os.path.exists(artifacts_dir):
            print(f"\n  Artifacts:")
            for fname in sorted(os.listdir(artifacts_dir)):
                fpath = os.path.join(artifacts_dir, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"    • {fname} ({size_mb:.1f} MB)")

        params = meta.get("params", {})
        if params:
            print(f"\n  Key Hyperparameters:")
            for key in ["MODEL_TYPE", "BATCH_SIZE", "INITIAL_EPOCHS", "FINETUNE_EPOCHS",
                        "INITIAL_LEARNING_RATE", "FINETUNE_LEARNING_RATE", "DROPOUT",
                        "DENSE_UNITS", "AUGMENTATION"]:
                if key in params:
                    print(f"    {key}: {params[key]}")

        return meta
