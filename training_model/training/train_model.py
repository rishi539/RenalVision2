import os
import sys
import json
import random
import yaml
import numpy as np

# Ensure UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class MacroRecallCallback(tf.keras.callbacks.Callback):
    """Custom callback to compute and log validation macro recall every epoch for model selection."""
    def __init__(self, val_gen, class_names):
        super().__init__()
        self.val_gen = val_gen
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_preds = self.model.predict(self.val_gen, verbose=0)
        pred_classes = np.argmax(val_preds, axis=1)
        true_classes = self.val_gen.classes
        _, val_macro_recall, _, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='macro', zero_division=0
        )
        logs['val_macro_recall'] = val_macro_recall
        print(f" — val_macro_recall: {val_macro_recall:.4f}")


class ModelTrainer:
    """
    Model2 Training Pipeline for 4-Class Kidney CT Classification:
    0: Normal, 1: Cyst, 2: Stone, 3: Tumor
    
    Fixes applied:
    1. Synchronized LOSS: categorical_crossentropy with class_mode='categorical'
    2. Synchronized IMAGE_SIZE: [224, 224] from params.yaml
    3. Explicit Class Mapping Order: classes=["Normal", "Cyst", "Stone", "Tumor"] (0=Normal, 1=Cyst, 2=Stone, 3=Tumor)
    4. Added MacroRecallCallback for primary model selection metric tracking (Macro Recall)
    """

    def __init__(self, params_path: str = "training_model/params.yaml", dataset_dir: str = "artifacts/data_ingestion/Kindey_Stone_Dataset"):
        if not os.path.exists(params_path) and os.path.exists("params.yaml"):
            params_path = "params.yaml"

        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        raw_size = self.params.get("IMAGE_SIZE", [224, 224])
        self.image_size = tuple(raw_size[:2])
        self.channels = self.params.get("CHANNELS", 3)
        self.batch_size = self.params.get("BATCH_SIZE", 16)
        self.classes = self.params.get("CLASSES", 4)
        self.class_names = self.params.get("CLASS_NAMES", ["Normal", "Cyst", "Stone", "Tumor"])
        self.dataset_dir = dataset_dir
        self.loss_func = self.params.get("LOSS", "categorical_crossentropy")

        self.model_save_path = "model/model2.h5"
        self.artifact_model_save_path = "artifacts/training/model2.h5"

        os.makedirs("artifacts/training", exist_ok=True)
        os.makedirs("model", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def build_vgg16_model(self) -> (tf.keras.Model, tf.keras.Model):
        """Constructs VGG16 backbone with custom classification head."""
        print("[INFO] Initializing pre-trained VGG16 backbone (ImageNet)...")
        base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=(self.image_size[0], self.image_size[1], self.channels),
            weights=self.params.get("WEIGHTS", "imagenet"),
            include_top=False
        )

        base_model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(self.params.get("DENSE_UNITS", 256))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(self.params.get("DROPOUT", 0.40))(x)
        output = tf.keras.layers.Dense(units=self.classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
        if os.path.exists(self.model_save_path):
            print(f"[INFO] Loading existing checkpoint weights from {self.model_save_path}...")
            try:
                model.load_weights(self.model_save_path)
                print("[INFO] Successfully loaded existing weights.")
            except Exception as e:
                print(f"[WARNING] Could not load weights: {e}")
        return base_model, model

    def create_data_generators(self):
        """Create image generators using VGG16 preprocess_input and explicit class mapping."""
        train_path = os.path.join(self.dataset_dir, "train")
        val_path = os.path.join(self.dataset_dir, "val")
        test_path = os.path.join(self.dataset_dir, "test")

        valid_dir = val_path if os.path.exists(val_path) else train_path
        test_dir = test_path if os.path.exists(test_path) else valid_dir

        dataflow_kwargs = dict(
            target_size=self.image_size,
            batch_size=self.batch_size,
            classes=self.class_names,  # EXPLICIT MAPPING: 0=Normal, 1=Cyst, 2=Stone, 3=Tumor
            interpolation="bilinear"
        )

        vgg_preprocess = tf.keras.applications.vgg16.preprocess_input

        val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=vgg_preprocess
        )

        if self.params.get("AUGMENTATION", True):
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=vgg_preprocess,
                rotation_range=int(self.params.get("ROTATION_FACTOR", 0.05) * 360),
                width_shift_range=self.params.get("TRANSLATION_FACTOR", 0.10),
                height_shift_range=self.params.get("TRANSLATION_FACTOR", 0.10),
                zoom_range=self.params.get("ZOOM_FACTOR", 0.10),
                horizontal_flip=self.params.get("HORIZONTAL_FLIP", True),
                vertical_flip=self.params.get("VERTICAL_FLIP", False),
                fill_mode='nearest'
            )
        else:
            train_datagen = val_test_datagen

        self.train_gen = train_datagen.flow_from_directory(
            directory=train_path,
            class_mode='categorical',
            shuffle=True,
            seed=SEED,
            **dataflow_kwargs
        )

        self.val_gen = val_test_datagen.flow_from_directory(
            directory=valid_dir,
            class_mode='categorical',
            shuffle=False,
            **dataflow_kwargs
        )

        self.test_gen = val_test_datagen.flow_from_directory(
            directory=test_dir,
            class_mode='categorical',
            shuffle=False,
            **dataflow_kwargs
        )

        # Save class mapping (e.g., {"0": "Normal", "1": "Cyst", "2": "Stone", "3": "Tumor"})
        class_indices = self.train_gen.class_indices
        class_mapping = {v: k for k, v in class_indices.items()}

        with open("model/class_names.json", "w") as f:
            json.dump(class_mapping, f, indent=4)
        with open("logs/class_names.json", "w") as f:
            json.dump(class_mapping, f, indent=4)

        print(f"[INFO] Verified Explicit Class Indices Mapping: {class_mapping}")

    def calculate_class_weights(self) -> dict:
        """Computes balanced class weights strictly from training labels."""
        train_labels = self.train_gen.classes
        unique_classes = np.unique(train_labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
        print(f"[INFO] Computed Training Class Weights: {class_weight_dict}")
        return class_weight_dict

    def train_stage1(self, model: tf.keras.Model, class_weight_dict: dict):
        """Stage 1: Train classifier head only with Adam(1e-3)."""
        print("\n[STAGE 1] Training Classifier Head (Backbone Frozen)...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get("INITIAL_LEARNING_RATE", 1e-3)),
            loss=self.loss_func,
            metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
        )

        macro_recall_cb = MacroRecallCallback(self.val_gen, self.class_names)

        callbacks = [
            macro_recall_cb,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.params.get("EARLY_STOPPING_PATIENCE_INITIAL", 4),
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.params.get("REDUCE_LR_FACTOR", 0.2),
                patience=self.params.get("REDUCE_LR_PATIENCE", 2),
                min_lr=self.params.get("MIN_LEARNING_RATE", 1e-7),
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        ]

        history_stage1 = model.fit(
            self.train_gen,
            epochs=self.params.get("INITIAL_EPOCHS", 10),
            initial_epoch=8,
            validation_data=self.val_gen,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )
        return history_stage1

    def fine_tune_stage2(self, base_model: tf.keras.Model, model: tf.keras.Model, class_weight_dict: dict):
        """Stage 2: Fine-tune VGG16 Block 5 with Adam(1e-5, clipnorm=1.0)."""
        print("\n[STAGE 2] Fine-Tuning VGG16 Block 5...")
        
        block5_layers = ['block5_conv1', 'block5_conv2', 'block5_conv3']
        for layer in base_model.layers:
            if layer.name in block5_layers:
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.params.get("FINETUNE_LEARNING_RATE", 1e-5),
                clipnorm=self.params.get("GRADIENT_CLIPNORM", 1.0)
            ),
            loss=self.loss_func,
            metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
        )

        macro_recall_cb = MacroRecallCallback(self.val_gen, self.class_names)

        callbacks = [
            macro_recall_cb,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.params.get("EARLY_STOPPING_PATIENCE_FINETUNE", 5),
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.params.get("REDUCE_LR_FACTOR", 0.2),
                patience=self.params.get("REDUCE_LR_PATIENCE", 2),
                min_lr=self.params.get("MIN_LEARNING_RATE", 1e-7),
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        ]

        history_stage2 = model.fit(
            self.train_gen,
            epochs=self.params.get("FINETUNE_EPOCHS", 25),
            validation_data=self.val_gen,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )
        return history_stage2

    def evaluate(self):
        """Evaluates best saved model2.h5 on held-out test split."""
        print("\n[EVALUATION] Evaluating Best Model (model2.h5) on Test Split...")

        best_model = tf.keras.models.load_model(self.model_save_path)

        predictions = best_model.predict(self.test_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_gen.classes
        class_mapping = {v: k for k, v in self.test_gen.class_indices.items()}

        accuracy = accuracy_score(true_classes, predicted_classes)
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
        for i in range(len(class_mapping)):
            c_name = class_mapping[i]
            per_class_metrics[c_name] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(support[i])
            }

        evaluation_data = {
            "model_name": "model2.h5",
            "overall_accuracy": round(float(accuracy), 4),
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

        with open("logs/scores_model2.json", "w") as f:
            json.dump(evaluation_data, f, indent=4)
        with open("logs/scores.json", "w") as f:
            json.dump(evaluation_data, f, indent=4)

        conf_data = {
            "class_names": [class_mapping[i] for i in range(len(class_mapping))],
            "matrix": conf_matrix.tolist()
        }
        with open("logs/confusion_matrix.json", "w") as f:
            json.dump(conf_data, f, indent=4)

        best_model.save(self.artifact_model_save_path)
        print(f"[SAVE] Model 2 successfully saved to {self.model_save_path} and {self.artifact_model_save_path}")
        print(f"[COMPLETE] Evaluation Complete!")
        print(f"   Overall Test Accuracy: {accuracy * 100:.2f}%")
        print(f"   Macro Recall (Primary Selection Metric): {macro_r * 100:.2f}%")
        print(f"   Macro F1 Score: {macro_f1 * 100:.2f}%")

        return evaluation_data


def run_training_pipeline():
    trainer = ModelTrainer()
    base_model, model = trainer.build_vgg16_model()
    trainer.create_data_generators()
    class_weight_dict = trainer.calculate_class_weights()

    # Stage 1: Classifier Head Training
    trainer.train_stage1(model, class_weight_dict)

    # Stage 2: Fine-Tuning Block 5
    trainer.fine_tune_stage2(base_model, model, class_weight_dict)

    # Final Evaluation
    trainer.evaluate()


if __name__ == "__main__":
    run_training_pipeline()
