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

        self.model_path = os.path.join(BASE_DIR, "backend", "model", "model1.h5")
        
        if os.path.exists(self.model_path):
            logger.info(f"Loading primary Model from: {self.model_path}")
            self.model = load_model(self.model_path, compile=False)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        class_names_path = os.path.join(BASE_DIR, "backend", "model", "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = {"0": "Normal", "1": "Cyst", "2": "Stone", "3": "Tumor"}

    def predict(self) -> list:
        """Runs model inference on input image using VGG16 preprocessing and returns predicted class, probabilities, and Grad-CAM."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Input image file not found: {self.filename}")

        # Load and preprocess image with VGG16 preprocess_input
        test_img_raw = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(test_img_raw)
        img_array_expanded = np.expand_dims(img_array.copy(), axis=0)

        # The original model was trained with rescale=1./255
        img_array_rescaled = img_array_expanded / 255.0

        # Model prediction
        predictions = self.model.predict(img_array_rescaled)
        probabilities = predictions[0]

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.class_names.get(str(predicted_idx), f"Class_{predicted_idx}")
        confidence = float(probabilities[predicted_idx]) * 100

        all_probabilities = {
            self.class_names.get(str(idx), f"Class_{idx}"): round(float(prob) * 100, 2)
            for idx, prob in enumerate(probabilities)
        }

        # Generate Grad-CAM Heatmap & Base64 Overlays
        gradcam_overlay = None
        gradcam_heatmap_img = None
        
        # Determine normal and maximum disease probabilities for dynamic alpha calculation
        normal_idx_str = next((k for k, v in self.class_names.items() if v.lower() == "normal"), None)
        normal_prob = float(probabilities[int(normal_idx_str)]) if normal_idx_str is not None else 0.0

        disease_classes = {
            idx: prob for idx, prob in enumerate(probabilities)
            if self.class_names.get(str(idx), "").lower() != "normal"
        }
        
        target_idx = predicted_idx
        alpha = 0.45

        max_disease_prob = 0.0
        if disease_classes:
            max_disease_idx = max(disease_classes, key=disease_classes.get)
            max_disease_prob = float(disease_classes[max_disease_idx])
            target_idx = max_disease_idx

        # Adjust Grad-CAM alpha dynamically based on conditions
        if normal_prob > 0.80:
            alpha = 0.0
        elif max_disease_prob > 0.60:
            alpha = 0.65
        elif max_disease_prob > 0.30:
            alpha = 0.35
        elif max_disease_prob > 0.20:
            alpha = 0.20
        else:
            alpha = 0.0

        try:
            if alpha > 0.0:
                heatmap = generate_gradcam_heatmap(
                    model=self.model,
                    img_array=img_array_rescaled,
                    pred_index=target_idx
                )
                visualizations = generate_gradcam_visualizations(
                    original_img_path=self.filename,
                    heatmap=heatmap,
                    alpha=alpha
                )
                gradcam_overlay = visualizations["overlay"]
                gradcam_heatmap_img = visualizations["heatmap"]
            else:
                import base64
                with open(self.filename, "rb") as f:
                    orig_b64 = base64.b64encode(f.read()).decode("utf-8")
                gradcam_overlay = f"data:image/jpeg;base64,{orig_b64}"
                gradcam_heatmap_img = None
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")

        return [{
            "prediction": predicted_class,
            "image": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": all_probabilities,
            "gradcam": gradcam_overlay,
            "heatmap": gradcam_heatmap_img,
            "heatmap_class": self.class_names.get(str(target_idx), "Unknown")
        }]
