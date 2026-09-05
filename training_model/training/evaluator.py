import os
import sys

# Disable OneDNN custom ops to prevent MKL memory allocation errors on Windows CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def evaluate_model(model_path: str = None, test_dir: str = None):
    """
    Evaluates an existing model on the test dataset without retraining.
    Updates logs/scores.json and logs/confusion_matrix.json.
    """
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "backend", "model", "model1.h5")
    if test_dir is None:
        test_dir = os.path.join(PROJECT_ROOT, "artifacts", "data_ingestion", "Kindey_Stone_Dataset", "test")

    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.isabs(test_dir):
        test_dir = os.path.join(PROJECT_ROOT, test_dir)

    print("=" * 60)
    print(f"  MODEL EVALUATION (No Training)")
    print(f"  Model Path: {model_path}")
    print(f"  Test Path:  {test_dir}")
    print("=" * 60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    print("\n[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    class_names = ["Normal", "Cyst", "Stone", "Tumor"]

    # Preprocessing matching model1.h5 (rescale=1./255)
    print("\n[INFO] Preparing test data generator (rescale=1./255)...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        directory=test_dir,
        target_size=(224, 224),
        batch_size=16,
        classes=class_names,
        class_mode='categorical',
        shuffle=False
    )

    print("\n[INFO] Running model inference on test images...")
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    # Compute loss manually or with keras categorical_crossentropy to avoid MKL memory error
    y_true_onehot = tf.keras.utils.to_categorical(true_classes, num_classes=len(class_names))
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = float(cce(y_true_onehot, predictions).numpy())

    accuracy = float(accuracy_score(true_classes, predicted_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_classes, predicted_classes, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        true_classes, predicted_classes, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        true_classes, predicted_classes, average='weighted', zero_division=0
    )
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    per_class_metrics = {}
    for i in range(len(class_names)):
        c_name = class_names[i]
        per_class_metrics[c_name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    evaluation_data = {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_metrics": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1_score": round(float(macro_f1), 4)
        },
        "weighted_metrics": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1_score": round(float(weighted_f1), 4)
        },
        "per_class_metrics": per_class_metrics
    }

    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    scores_path = os.path.join(logs_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(evaluation_data, f, indent=4)

    conf_path = os.path.join(logs_dir, "confusion_matrix.json")
    conf_data = {
        "class_names": class_names,
        "matrix": conf_matrix.tolist()
    }
    with open(conf_path, "w") as f:
        json.dump(conf_data, f, indent=4)

    print("\n" + "=" * 60)
    print("  [OK] EVALUATION COMPLETED & SCORES UPDATED")
    print(f"  Loss:           {loss:.4f}")
    print(f"  Test Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Macro Recall:   {macro_r * 100:.2f}%")
    print(f"  Macro F1 Score: {macro_f1 * 100:.2f}%")
    print(f"\n  Scores saved to:    {scores_path}")
    print(f"  Confusion Matrix:   {conf_path}")
    print("=" * 60)

    return evaluation_data

if __name__ == "__main__":
    evaluate_model()
