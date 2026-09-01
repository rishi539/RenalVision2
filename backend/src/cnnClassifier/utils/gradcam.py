import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from cnnClassifier import logger


def find_conv_layers(model: tf.keras.Model) -> list:
    """
    Finds names of target Conv2D layers across distinct architectural blocks (e.g. block5_conv3 and block4_conv3).
    """
    conv_layers = []
    seen_blocks = set()
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            block_prefix = layer.name.split("_")[0] if "_" in layer.name else layer.name
            if block_prefix not in seen_blocks:
                seen_blocks.add(block_prefix)
                conv_layers.append(layer.name)
            if len(conv_layers) == 2:
                break
    
    if not conv_layers:
        raise ValueError("No convolutional layer found in model.")
    
    conv_layers.reverse()  # [penultimate_block_conv, last_block_conv]
    logger.info(f"Detected target convolutional layers for Grad-CAM: {conv_layers}")
    return conv_layers


def generate_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray, pred_index: int = None) -> np.ndarray:
    """
    Generates a high-precision multi-layer fused Grad-CAM heatmap array [0, 1].
    Combines deep semantic features (block5_conv3) with fine structural features (block4_conv3).
    """
    conv_layer_names = find_conv_layers(model)
    last_conv_name = conv_layer_names[-1]
    
    # Check if penultimate layer is available for multi-scale fusion
    penultimate_conv_name = conv_layer_names[0] if len(conv_layer_names) > 1 else None

    if penultimate_conv_name and penultimate_conv_name != last_conv_name:
        last_conv = model.get_layer(last_conv_name)
        penultimate_conv = model.get_layer(penultimate_conv_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[penultimate_conv.output, last_conv.output, model.output]
        )

        with tf.GradientTape(persistent=True) as tape:
            pen_outputs, last_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list):
                predictions = predictions[0]
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradients for last conv layer (block5_conv3 - deep semantics)
        grads_last = tape.gradient(class_channel, last_outputs)
        pooled_grads_last = tf.reduce_mean(grads_last, axis=(0, 1, 2))
        conv_outputs_last = last_outputs[0]
        heatmap_last = conv_outputs_last @ pooled_grads_last[..., tf.newaxis]
        heatmap_last = tf.squeeze(heatmap_last)
        heatmap_last = tf.maximum(heatmap_last, 0)

        # Gradients for penultimate conv layer (block4_conv3 - spatial structure)
        grads_pen = tape.gradient(class_channel, pen_outputs)
        pooled_grads_pen = tf.reduce_mean(grads_pen, axis=(0, 1, 2))
        conv_outputs_pen = pen_outputs[0]
        heatmap_pen = conv_outputs_pen @ pooled_grads_pen[..., tf.newaxis]
        heatmap_pen = tf.squeeze(heatmap_pen)
        heatmap_pen = tf.maximum(heatmap_pen, 0)

        del tape

        # Normalize individual heatmaps
        max_last = tf.reduce_max(heatmap_last)
        if max_last > 0:
            heatmap_last = heatmap_last / max_last

        max_pen = tf.reduce_max(heatmap_pen)
        if max_pen > 0:
            heatmap_pen = heatmap_pen / max_pen

        # Resize last heatmap to match penultimate heatmap resolution (14x14)
        last_h, last_w = heatmap_last.shape
        pen_h, pen_w = heatmap_pen.shape
        
        heatmap_last_img = Image.fromarray(heatmap_last.numpy()).resize((pen_w, pen_h), Image.Resampling.BILINEAR)
        heatmap_last_resized = np.array(heatmap_last_img)

        # Fuse 70% deep semantic + 30% structural detail
        fused_heatmap = 0.70 * heatmap_last_resized + 0.30 * heatmap_pen.numpy()
        heatmap = fused_heatmap

    else:
        # Fallback to single-layer Grad-CAM
        last_conv = model.get_layer(last_conv_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list):
                predictions = predictions[0]
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0).numpy()

    # Final normalization
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap


def generate_gradcam_visualizations(
    original_img_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    power: float = 1.2,
    threshold: float = 0.02,
    colormap: str = "jet"
) -> dict:
    """
    Applies focal sharpening, background noise suppression, and tissue-aware masking
    to generate highly accurate, localized Grad-CAM overlay visualizations.
    """

    # Open original CT image
    orig_img = Image.open(original_img_path).convert("RGB")
    width, height = orig_img.size

    # Convert to grayscale for CT foreground tissue masking
    orig_gray = np.array(orig_img.convert("L")).astype(np.float32) / 255.0

    # 2. Resize raw heatmap to image dimensions
    heatmap_raw_img = Image.fromarray(heatmap).resize((width, height), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_raw_img)

    # 3. Apply Heatmap directly without Tissue Mask
    heatmap_masked = heatmap_resized

    # 4. Noise Floor Suppression & Soft Thresholding
    heatmap_thresh = np.maximum(heatmap_masked - 0.0, 0)
    max_thresh = np.max(heatmap_thresh)
    if max_thresh > 0:
        heatmap_thresh = heatmap_thresh / max_thresh

    # 5. Non-linear Focal Contrast Power Scaling (Sharpen peak over lesion mass)
    heatmap_sharpened = heatmap_thresh ** power
    max_sharp = np.max(heatmap_sharpened)
    if max_sharp > 0:
        heatmap_sharpened = heatmap_sharpened / max_sharp

    # 6. Apply Jet Colormap
    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)

    heatmap_uint8 = np.uint8(255 * heatmap_sharpened)
    heatmap_colored = (cmap(heatmap_uint8)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)

    # 7. Superimpose heatmap onto original CT scan
    orig_np = np.array(orig_img).astype(np.float32)
    heatmap_np = np.array(heatmap_img).astype(np.float32)

    # Dynamic alpha blending: blend stronger over high activation zones, lighter over neutral zones
    alpha_map = (heatmap_sharpened * alpha)[..., np.newaxis]
    overlay_np = (1 - alpha_map) * orig_np + alpha_map * heatmap_np
    overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay_np)

    # Encode overlay image to Base64
    buf_overlay = io.BytesIO()
    overlay_img.save(buf_overlay, format="JPEG", quality=95)
    overlay_b64 = base64.b64encode(buf_overlay.getvalue()).decode("utf-8")

    # Encode standalone heatmap image to Base64
    buf_heatmap = io.BytesIO()
    heatmap_img.save(buf_heatmap, format="JPEG", quality=95)
    heatmap_b64 = base64.b64encode(buf_heatmap.getvalue()).decode("utf-8")

    return {
        "overlay": f"data:image/jpeg;base64,{overlay_b64}",
        "heatmap": f"data:image/jpeg;base64,{heatmap_b64}"
    }
