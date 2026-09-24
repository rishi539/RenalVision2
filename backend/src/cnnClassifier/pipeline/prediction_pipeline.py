import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier.utils.gradcam import generate_gradcam_heatmap, generate_gradcam_visualizations
from cnnClassifier import logger


class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

        # ═══════════════════════════════════════════════════════════════════
        # DUAL-MODEL ARCHITECTURE
        # ─────────────────────────────────────────────────────────────────
        # Prediction Model : renalvision_v2_1_best.keras  (accurate classification)
        # Grad-CAM Model   : model1.h5                    (high-quality heatmaps)
        # ═══════════════════════════════════════════════════════════════════

        # --- Prediction Model (.keras) ---
        self.prediction_model_path = os.path.join(BASE_DIR, "backend", "model", "renalvision_v2_1_best.keras")
        if os.path.exists(self.prediction_model_path):
            logger.info(f"Loading PREDICTION model from: {self.prediction_model_path}")
            self.prediction_model = load_model(self.prediction_model_path, compile=False)
        else:
            raise FileNotFoundError(f"Prediction model not found at {self.prediction_model_path}")

        # Check if prediction model handles normalization internally (e.g. ConvNeXt, EfficientNet)
        self.pred_has_internal_normalization = any(
            isinstance(l, (tf.keras.layers.Normalization, tf.keras.layers.Rescaling)) or
            "normalization" in l.name.lower() or "rescaling" in l.name.lower()
            for l in self.prediction_model.layers[:5]
        )
        logger.info(f"Prediction model internal normalization: {self.pred_has_internal_normalization}")

        # --- Grad-CAM Model (.h5) ---
        self.gradcam_model_path = os.path.join(BASE_DIR, "backend", "model", "model1.h5")
        if os.path.exists(self.gradcam_model_path):
            logger.info(f"Loading GRAD-CAM model from: {self.gradcam_model_path}")
            self.gradcam_model = load_model(self.gradcam_model_path, compile=False)
        else:
            logger.warning(f"Grad-CAM model not found at {self.gradcam_model_path}, falling back to prediction model for Grad-CAM.")
            self.gradcam_model = self.prediction_model
            self.gradcam_model_path = self.prediction_model_path

        # Check if gradcam model handles normalization internally
        self.gradcam_has_internal_normalization = any(
            isinstance(l, (tf.keras.layers.Normalization, tf.keras.layers.Rescaling)) or
            "normalization" in l.name.lower() or "rescaling" in l.name.lower()
            for l in self.gradcam_model.layers[:5]
        )
        logger.info(f"Grad-CAM model internal normalization: {self.gradcam_has_internal_normalization}")

        # Load class names mapping
        class_names_path = os.path.join(BASE_DIR, "backend", "model", "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = {"0": "Normal", "1": "Cyst", "2": "Stone", "3": "Tumor"}

        logger.info("Dual-model pipeline initialized successfully.")

    def predict(self) -> list:
        """Runs dual-model inference: .keras for prediction, .h5 for Grad-CAM explainability."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Input image file not found: {self.filename}")

        # Load image with target input dimensions (224, 224)
        test_img_raw = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(test_img_raw)
        img_array_expanded = np.expand_dims(img_array.copy(), axis=0)

        # ── STEP 1: Prediction using .keras model ──────────────────────
        # Scale pixels appropriately for prediction model
        if self.pred_has_internal_normalization:
            pred_input_tensor = img_array_expanded.astype(np.float32)
        else:
            pred_input_tensor = (img_array_expanded / 255.0).astype(np.float32)

        logger.info("Running prediction inference with .keras model...")
        predictions = self.prediction_model.predict(pred_input_tensor, verbose=0)
        probabilities = predictions[0]

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.class_names.get(str(predicted_idx), f"Class_{predicted_idx}")
        confidence = float(probabilities[predicted_idx]) * 100

        all_probabilities = {
            self.class_names.get(str(idx), f"Class_{idx}"): round(float(prob) * 100, 2)
            for idx, prob in enumerate(probabilities)
        }

        normal_pct = all_probabilities.get("Normal", 0.0)
        disease_classes = {
            idx: prob for idx, prob in enumerate(probabilities)
            if self.class_names.get(str(idx), "").lower() != "normal"
        }

        # Check if any disease (Cyst, Stone, Tumor) has >= 10% probability
        highest_disease_idx = max(disease_classes, key=disease_classes.get) if disease_classes else None
        max_disease_pct = (float(disease_classes[highest_disease_idx]) * 100.0) if highest_disease_idx is not None else 0.0

        # CLINICAL GRAD-CAM HIGHLIGHTING RULE:
        # 1. If any abnormal class (Stone, Tumor, Cyst) is >= 10%, ALWAYS highlight Grad-CAM for that disease.
        # 2. If Normal is >= 80% and all diseases are < 10%, DO NOT highlight Grad-CAM (clean healthy scan).
        should_highlight = False
        target_idx = predicted_idx

        if max_disease_pct >= 10.0:
            should_highlight = True
            # Target the disease class for explainability
            target_idx = highest_disease_idx if predicted_class.lower() == "normal" else predicted_idx
            alpha = 0.55
        elif normal_pct >= 80.0:
            should_highlight = False
            target_idx = predicted_idx
            alpha = 0.0
        elif predicted_class.lower() != "normal":
            should_highlight = True
            target_idx = predicted_idx
            alpha = 0.50
        else:
            should_highlight = False
            target_idx = predicted_idx
            alpha = 0.0

        import base64
        with open(self.filename, "rb") as f:
            orig_b64 = base64.b64encode(f.read()).decode("utf-8")
        orig_data_url = f"data:image/jpeg;base64,{orig_b64}"

        gradcam_overlay = None
        gradcam_heatmap_img = None

        # ── STEP 2: Grad-CAM using .h5 model ──────────────────────────
        if should_highlight:
            try:
                # Prepare input tensor for Grad-CAM model (may have different normalization)
                if self.gradcam_has_internal_normalization:
                    gradcam_input_tensor = img_array_expanded.astype(np.float32)
                else:
                    gradcam_input_tensor = (img_array_expanded / 255.0).astype(np.float32)

                logger.info(f"Generating Grad-CAM heatmap with .h5 model for class index {target_idx}...")
                heatmap = generate_gradcam_heatmap(
                    model=self.gradcam_model,
                    img_array=gradcam_input_tensor,
                    pred_index=target_idx
                )
                visualizations = generate_gradcam_visualizations(
                    original_img_path=self.filename,
                    heatmap=heatmap,
                    alpha=alpha
                )
                gradcam_overlay = visualizations["overlay"]
                gradcam_heatmap_img = visualizations["heatmap"]
                logger.info("Grad-CAM heatmap generated successfully with .h5 model.")
            except Exception as e:
                logger.error(f"Grad-CAM generation failed with .h5 model: {e}")
                gradcam_overlay = orig_data_url
        else:
            # Healthy Normal Scan: Suppress Grad-CAM highlighting to prevent false-positive artifacts
            gradcam_overlay = orig_data_url
            gradcam_heatmap_img = None

        target_class_name = self.class_names.get(str(target_idx), predicted_class)
        highlight_msg = (
            f"Highlighting {target_class_name} activation region (significant disease probability detected)."
            if should_highlight else
            "Healthy / Normal Scan (Normal >= 80%). Grad-CAM heatmap highlighting is suppressed for normal findings."
        )

        # Identify which models were used
        pred_model_name = os.path.basename(self.prediction_model_path)
        gradcam_model_name = os.path.basename(self.gradcam_model_path)

        return [{
            "prediction": predicted_class,
            "image": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": all_probabilities,
            "gradcam": gradcam_overlay,
            "heatmap": gradcam_heatmap_img,
            "heatmap_class": target_class_name,
            "is_highlighted": should_highlight,
            "highlight_reason": highlight_msg,
            "prediction_model": pred_model_name,
            "gradcam_model": gradcam_model_name
        }]
