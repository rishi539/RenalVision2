import argparse
import sys
import os

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure training_model root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Kidney Disease Classification — MLOps Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--run-name", type=str, default=None,
        help="Human-readable name for this training run (default: auto-generated timestamp)",
    )
    train_parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset directory (must contain train/val/test splits)",
    )
    train_parser.add_argument(
        "--params", type=str, default=None,
        help="Path to params.yaml (default: training_model/params.yaml)",
    )

    # --- list-runs ---
    subparsers.add_parser("list-runs", help="List all past training runs")

    # --- show-run ---
    show_parser = subparsers.add_parser("show-run", help="Show detailed info for a specific run")
    show_parser.add_argument(
        "--name", type=str, required=True,
        help="Name of the run to inspect",
    )

    # --- compare ---
    subparsers.add_parser("compare", help="Compare all registered model versions")

    # --- promote ---
    promote_parser = subparsers.add_parser("promote", help="Promote a model version to production")
    promote_parser.add_argument(
        "--version", type=int, required=True,
        help="Model version number to promote",
    )

    # --- add-dataset ---
    add_ds_parser = subparsers.add_parser("add-dataset", help="Register and version a dataset with DVC")
    add_ds_parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the dataset directory",
    )
    add_ds_parser.add_argument(
        "--name", type=str, default=None,
        help="Human-readable name for this dataset version",
    )

    # --- list-datasets ---
    subparsers.add_parser("list-datasets", help="List all registered dataset versions")

    # --- validate-dataset ---
    val_ds_parser = subparsers.add_parser("validate-dataset", help="Validate a dataset structure")
    val_ds_parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the dataset directory to validate",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing model without re-training")
    eval_parser.add_argument(
        "--model-path", type=str, default="backend/model/model1.h5",
        help="Path to trained model .h5 file",
    )
    eval_parser.add_argument(
        "--dataset", type=str, default="artifacts/data_ingestion/Kindey_Stone_Dataset",
        help="Path to dataset directory",
    )

    # --- mlflow-ui ---
    subparsers.add_parser("mlflow-ui", help="Launch the MLflow tracking UI")

    args = parser.parse_args()


    if args.command is None:
        parser.print_help()
        print("\nExamples:")
        print("  python training_model/main.py train --run-name experiment-v1")
        print("  python training_model/main.py train --dataset path/to/dataset --run-name new-data")
        print("  python training_model/main.py list-runs")
        print("  python training_model/main.py show-run --name experiment-v1")
        print("  python training_model/main.py compare")
        print("  python training_model/main.py promote --version 1")
        print("  python training_model/main.py add-dataset --path path/to/dataset")
        print("  python training_model/main.py list-datasets")
        print("  python training_model/main.py mlflow-ui")
        sys.exit(1)

    # ---- Route commands ----

    if args.command == "train":
        pipeline = TrainingPipeline(
            run_name=args.run_name,
            dataset_dir=args.dataset,
            params_path=args.params,
        )
        pipeline.run()

    elif args.command == "list-runs":
        TrainingPipeline.list_runs()

    elif args.command == "show-run":
        TrainingPipeline.show_run_details(args.name)

    elif args.command == "compare":
        TrainingPipeline.compare_models()

    elif args.command == "promote":
        TrainingPipeline.promote_model(args.version)

    elif args.command == "add-dataset":
        from training.data_manager import DataManager
        DataManager.register_dataset(args.path, args.name)

    elif args.command == "list-datasets":
        from training.data_manager import DataManager
        DataManager.list_datasets()

    elif args.command == "validate-dataset":
        from training.data_manager import DataManager
        result = DataManager.validate_dataset(args.path)
        if result["valid"]:
            print("[OK] Dataset is valid!")
        else:
            print("[FAIL] Dataset validation failed:")
        for err in result.get("errors", []):
            print(f"  [x] {err}")
        for warn in result.get("warnings", []):
            print(f"  [!] {warn}")
        stats = result.get("stats", {})
        if stats:
            print(f"\n  Stats:")
            for key, val in stats.items():
                print(f"    {key}: {val}")

    elif args.command == "evaluate":
        from training.evaluator import evaluate_model
        test_dir = os.path.join(args.dataset, "test") if os.path.exists(os.path.join(args.dataset, "test")) else args.dataset
        evaluate_model(model_path=args.model_path, test_dir=test_dir)

    elif args.command == "mlflow-ui":
        import subprocess
        mlruns_dir = os.path.join(os.path.dirname(__file__), "mlruns")
        print(f"Launching MLflow UI (tracking: {mlruns_dir})...")
        print("Open http://127.0.0.1:5000 in your browser")
        subprocess.run(
            ["mlflow", "ui", "--backend-store-uri", mlruns_dir],
            cwd=os.path.dirname(__file__),
        )


if __name__ == "__main__":
    main()
