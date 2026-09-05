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

        # Primary Model: ConvNeXt-based V2.1 model (.keras native format)
        self.model_path = os.path.join(BASE_DIR, "backend", "model", "renalvision_v2_1_best.keras")
        if not os.path.exists(self.model_path):
            logger.info("Primary V2.1 model not found, falling back to model1.h5...")
            self.model_path = os.path.join(BASE_DIR, "backend", "model", "model1.h5")
        
        if os.path.exists(self.model_path):
            logger.info(f"Loading Model from: {self.model_path}")
            self.model = load_model(self.model_path, compile=False)
        else:
            raise FileNotFoundError(f"No valid model found at {self.model_path}")

        # Check if model handles normalization internally (e.g. ConvNeXt, EfficientNet)
        self.has_internal_normalization = any(
            isinstance(l, (tf.keras.layers.Normalization, tf.keras.layers.Rescaling)) or
            "normalization" in l.name.lower() or "rescaling" in l.name.lower()
            for l in self.model.layers[:5]
        )
        logger.info(f"Model internal normalization detected: {self.has_internal_normalization}")

        # Load class names mapping
        class_names_path = os.path.join(BASE_DIR, "backend", "model", "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = {"0": "Normal", "1": "Cyst", "2": "Stone", "3": "Tumor"}

    def predict(self) -> list:
        """Runs model inference on input image and returns predicted class, probabilities, and Grad-CAM."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Input image file not found: {self.filename}")

        # Load image with target input dimensions (224, 224)
        test_img_raw = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(test_img_raw)
        img_array_expanded = np.expand_dims(img_array.copy(), axis=0)

        # Scale pixels appropriately: ConvNeXt/EfficientNet has built-in Normalization expecting [0, 255]
        if self.has_internal_normalization:
            input_tensor = img_array_expanded.astype(np.float32)
        else:
            input_tensor = (img_array_expanded / 255.0).astype(np.float32)

        # Model inference
        predictions = self.model.predict(input_tensor, verbose=0)
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

        if should_highlight:
            try:
                heatmap = generate_gradcam_heatmap(
                    model=self.model,
                    img_array=input_tensor,
                    pred_index=target_idx
                )
                visualizations = generate_gradcam_visualizations(
                    original_img_path=self.filename,
                    heatmap=heatmap,
                    alpha=alpha
                )
                gradcam_overlay = visualizations["overlay"]
                gradcam_heatmap_img = visualizations["heatmap"]
            except Exception as e:
                logger.error(f"Grad-CAM generation failed: {e}")
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

        return [{
            "prediction": predicted_class,
            "image": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": all_probabilities,
            "gradcam": gradcam_overlay,
            "heatmap": gradcam_heatmap_img,
            "heatmap_class": target_class_name,
            "is_highlighted": should_highlight,
            "highlight_reason": highlight_msg
        }]
