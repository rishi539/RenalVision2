import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from cnnClassifier import logger


def get_multiscale_layers(model: tf.keras.Model) -> dict:
    """
    Identifies the high-resolution spatial feature layer
    and the deep semantic feature layer for full-region Grad-CAM.
    """
    layer_names = [l.name for l in model.layers]
    
    # 1. ConvNeXt architectures
    if "convnext_tiny_stage_2_block_8_depthwise_conv" in layer_names and "layer_normalization" in layer_names:
        return {
            "spatial": "convnext_tiny_stage_2_block_8_depthwise_conv",
            "semantic": "layer_normalization"
        }
        
    # 2. VGG16 architectures
    if "block5_conv3" in layer_names and "block4_conv3" in layer_names:
        return {
            "spatial": "block4_conv3",
            "semantic": "block5_conv3"
        }

    # 3. Dynamic search for pre-pooling 4D layers
    conv_4d = []
    for l in model.layers:
        shape = getattr(l, 'output_shape', None)
        if shape is None and hasattr(l, 'output'):
            shape = l.output.shape
        if shape and len(shape) == 4 and not isinstance(l, tf.keras.layers.InputLayer):
            conv_4d.append(l.name)
            
    if len(conv_4d) >= 2:
        return {"spatial": conv_4d[-2], "semantic": conv_4d[-1]}
    elif conv_4d:
        return {"spatial": conv_4d[-1], "semantic": conv_4d[-1]}
    else:
        raise ValueError("No 4D spatial feature layers found in model.")


def generate_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray, pred_index: int = None) -> np.ndarray:
    """
    Generates a full-volume, high-precision Grad-CAM heatmap array [0, 1].
    Combines deep semantic guidance (Class 'What') with anatomical tissue guidance (Spatial 'Where').
    Preserves all affected regions across the organ rather than collapsing into single dots.
    """
    layers_dict = get_multiscale_layers(model)
    spatial_layer_name = layers_dict["spatial"]
    semantic_layer_name = layers_dict["semantic"]

    spatial_layer = model.get_layer(spatial_layer_name)
    semantic_layer = model.get_layer(semantic_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[spatial_layer.output, semantic_layer.output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape:
        out_spatial, out_semantic, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients for semantic layer (captures overall affected lesion volume)
    grads_semantic = tape.gradient(class_channel, out_semantic)
    pooled_semantic = tf.reduce_mean(tf.maximum(grads_semantic, 0), axis=(0, 1, 2))
    cam_semantic = tf.maximum(out_semantic[0] @ pooled_semantic[..., tf.newaxis], 0).numpy().squeeze()
    cam_semantic = np.nan_to_num(cam_semantic, nan=0.0)
    if np.max(cam_semantic) > 0:
        cam_semantic = cam_semantic / np.max(cam_semantic)

    # Gradients for spatial layer (captures fine organ and lesion contours)
    grads_spatial = tape.gradient(class_channel, out_spatial)
    pooled_spatial = tf.reduce_mean(tf.maximum(grads_spatial, 0), axis=(0, 1, 2))
    cam_spatial = tf.maximum(out_spatial[0] @ pooled_spatial[..., tf.newaxis], 0).numpy().squeeze()
    cam_spatial = np.nan_to_num(cam_spatial, nan=0.0)
    if np.max(cam_spatial) > 0:
        cam_spatial = cam_spatial / np.max(cam_spatial)

    del tape

    # Resize both feature maps to high-resolution (e.g. 224x224)
    cam_sem_resized = np.array(
        Image.fromarray(cam_semantic).resize((224, 224), Image.Resampling.BICUBIC)
    ).astype(np.float32)
    cam_spa_resized = np.array(
        Image.fromarray(cam_spatial).resize((224, 224), Image.Resampling.BICUBIC)
    ).astype(np.float32)

    # Additive regional fusion: 55% broad semantic coverage + 45% anatomical tissue contour
    # Ensures all affected parts of the lesion/organ are illuminated
    fused_heatmap = 0.55 * cam_sem_resized + 0.45 * cam_spa_resized
    fused_heatmap = np.nan_to_num(fused_heatmap, nan=0.0)
    if np.max(fused_heatmap) > 0:
        fused_heatmap = fused_heatmap / np.max(fused_heatmap)

    return fused_heatmap


def generate_gradcam_visualizations(
    original_img_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.60,
    power: float = 1.12,
    colormap: str = "jet"
) -> dict:
    """
    Applies anatomical tissue masking and smooth thermal contrast to highlight
    the entire affected pathology region with proper clinical intensity.
    """
    orig_img = Image.open(original_img_path).convert("RGB")
    width, height = orig_img.size

    # Resize raw heatmap to original image dimensions with bicubic smoothing
    heatmap_raw_img = Image.fromarray(heatmap).resize((width, height), Image.Resampling.BICUBIC)
    heatmap_resized = np.array(heatmap_raw_img).astype(np.float32)

    # CT Anatomical Tissue Mask: Suppress empty black background and scanner bed
    orig_gray = np.array(orig_img.convert("L")).astype(np.float32) / 255.0
    tissue_mask = np.where(orig_gray > 0.12, 1.0, 0.0)
    
    # Soften tissue mask edges with Gaussian blur
    mask_pil = Image.fromarray((tissue_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=8))
    soft_tissue_mask = np.array(mask_pil).astype(np.float32) / 255.0

    # Apply tissue mask to keep heatmap within bodily tissues
    heatmap_masked = heatmap_resized * soft_tissue_mask
    heatmap_masked = np.nan_to_num(heatmap_masked, nan=0.0, posinf=1.0, neginf=0.0)
    max_masked = np.max(heatmap_masked)
    if max_masked > 0:
        heatmap_masked = heatmap_masked / max_masked

    # Gentle noise gate (5%) to preserve surrounding affected tissue context
    heatmap_thresh = np.maximum(heatmap_masked - 0.05, 0.0)
    max_thresh = np.max(heatmap_thresh)
    if max_thresh > 0:
        heatmap_thresh = heatmap_thresh / max_thresh

    # Smooth non-linear thermal contrast (power=1.12 maintains broad regional coverage)
    heatmap_thresh = np.clip(heatmap_thresh, 0.0, 1.0)
    heatmap_sharpened = heatmap_thresh ** power
    max_sharp = np.max(heatmap_sharpened)
    if max_sharp > 0:
        heatmap_sharpened = heatmap_sharpened / max_sharp

    # Apply Jet Colormap
    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)

    heatmap_uint8 = np.uint8(255 * heatmap_sharpened)
    heatmap_colored = (cmap(heatmap_uint8)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)

    # Superimpose heatmap onto original CT scan
    orig_np = np.array(orig_img).astype(np.float32)
    heatmap_np = np.array(heatmap_img).astype(np.float32)

    # Dynamic alpha blending: 60% intensity at peak, fading smoothly across affected tissue
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
