import os
import json
import subprocess
from pathlib import Path


# Root of the project (parent of training_model/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DVC_DATASETS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "data_ingestion")
DATASET_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset_registry.json"
)


class DataManager:
    """
    Manages dataset versioning using DVC.

    Provides:
    - register_dataset(): Track a new dataset with DVC
    - list_datasets(): List all tracked dataset versions
    - validate_dataset(): Verify dataset structure (train/val/test + class folders)
    """

    REQUIRED_SPLITS = ["train"]
    EXPECTED_CLASSES = ["Normal", "Cyst", "Stone", "Tumor"]

    @staticmethod
    def validate_dataset(dataset_path: str) -> dict:
        """
        Validate that a dataset has the expected structure.

        Expected:
            dataset_path/
            ├── train/
            │   ├── Normal/
            │   ├── Cyst/
            │   ├── Stone/
            │   └── Tumor/
            ├── val/  (optional, falls back to train)
            └── test/ (optional, falls back to val)
        """
        result = {"valid": True, "errors": [], "warnings": [], "stats": {}}

        if not os.path.exists(dataset_path):
            result["valid"] = False
            result["errors"].append(f"Dataset path does not exist: {dataset_path}")
            return result

        # Check required splits
        for split in DataManager.REQUIRED_SPLITS:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                result["valid"] = False
                result["errors"].append(f"Required split '{split}/' not found")

        # Check optional splits
        for split in ["val", "test"]:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                result["warnings"].append(
                    f"Optional split '{split}/' not found — will fall back"
                )

        # Check class directories inside train/
        train_path = os.path.join(dataset_path, "train")
        if os.path.exists(train_path):
            found_classes = sorted(
                [d for d in os.listdir(train_path)
                 if os.path.isdir(os.path.join(train_path, d))]
            )
            result["stats"]["classes_found"] = found_classes
            result["stats"]["num_classes"] = len(found_classes)

            for cls in DataManager.EXPECTED_CLASSES:
                cls_path = os.path.join(train_path, cls)
                if not os.path.exists(cls_path):
                    result["warnings"].append(
                        f"Expected class directory '{cls}' not found in train/"
                    )
                else:
                    num_images = len([
                        f for f in os.listdir(cls_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                    ])
                    result["stats"][f"train_{cls}_count"] = num_images

            # Count total per split
            for split in ["train", "val", "test"]:
                split_path = os.path.join(dataset_path, split)
                if os.path.exists(split_path):
                    total = 0
                    for cls_dir in os.listdir(split_path):
                        cls_path = os.path.join(split_path, cls_dir)
                        if os.path.isdir(cls_path):
                            total += len([
                                f for f in os.listdir(cls_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                            ])
                    result["stats"][f"{split}_total"] = total

        return result

    @staticmethod
    def register_dataset(dataset_path: str, name: str = None) -> dict:
        """
        Track a dataset directory with DVC.

        Args:
            dataset_path: Path to the dataset folder
            name: Human-readable name for this dataset version

        Returns:
            dict with registration info
        """
        abs_path = os.path.abspath(dataset_path)

        # Validate first
        validation = DataManager.validate_dataset(abs_path)
        if not validation["valid"]:
            print("[ERROR] Dataset validation failed:")
            for err in validation["errors"]:
                print(f"  ✗ {err}")
            return {"error": "Validation failed", "details": validation}

        # Print warnings
        for warn in validation.get("warnings", []):
            print(f"  ⚠ {warn}")

        # Run dvc add
        try:
            result = subprocess.run(
                ["dvc", "add", abs_path],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                print(f"[WARNING] DVC add returned non-zero: {result.stderr}")
                # Don't fail — DVC might not be initialized, still register locally
        except FileNotFoundError:
            print("[WARNING] DVC not installed — skipping DVC tracking. Dataset registered locally only.")

        # Register in local dataset registry
        registry = []
        if os.path.exists(DATASET_REGISTRY_PATH):
            with open(DATASET_REGISTRY_PATH, "r") as f:
                registry = json.load(f)

        from datetime import datetime
        version = len(registry) + 1
        entry = {
            "version": version,
            "name": name or f"dataset-v{version}",
            "path": abs_path,
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": validation["stats"],
        }
        registry.append(entry)

        with open(DATASET_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=4)

        print(f"\n[DATASET] Registered as version {version}: {entry['name']}")
        print(f"  Path: {abs_path}")
        print(f"  Classes: {validation['stats'].get('classes_found', [])}")
        print(f"  Train images: {validation['stats'].get('train_total', '?')}")

        return entry

    @staticmethod
    def list_datasets() -> list:
        """List all registered dataset versions."""
        if not os.path.exists(DATASET_REGISTRY_PATH):
            print("No datasets registered yet.")
            print(f"\nTo register a dataset:")
            print(f"  python training_model/main.py add-dataset --path <dataset_path>")
            return []

        with open(DATASET_REGISTRY_PATH, "r") as f:
            registry = json.load(f)

        if not registry:
            print("No datasets registered yet.")
            return []

        print(f"\n{'Ver':<5} {'Name':<25} {'Classes':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Registered At'}")
        print("-" * 95)
        for entry in registry:
            stats = entry.get("stats", {})
            print(
                f"{entry['version']:<5} "
                f"{entry['name']:<25} "
                f"{str(stats.get('num_classes', '?')):<8} "
                f"{str(stats.get('train_total', '?')):<8} "
                f"{str(stats.get('val_total', '—')):<8} "
                f"{str(stats.get('test_total', '—')):<8} "
                f"{entry.get('registered_at', '—')}"
            )

        return registry
