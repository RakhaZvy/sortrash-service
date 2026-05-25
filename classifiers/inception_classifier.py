import os
import numpy as np
from PIL import Image

CLASSES = ["Cardboard", "Trash", "Plastic", "Metal", "Glass", "Paper"]

# Loaded lazily so the app still starts when TensorFlow is not installed
_model = None
_available = False


def _load():
    global _model, _available
    if _available:
        return True
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "inception_model.keras")
    # Also accept the .h5 name used by the training notebook
    h5_path = os.path.join(os.path.dirname(__file__), "..", "models", "garbage_classification_model_inception.h5")
    target = None
    if os.path.exists(model_path):
        target = model_path
    elif os.path.exists(h5_path):
        target = h5_path
    if target is None:
        return False
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(target)
        _available = True
        print(f"Classifier loaded from {target}")
    except Exception as exc:
        print(f"Classifier load error: {exc}")
    return _available


def is_available() -> bool:
    return _load()


def predict(pil_image: Image.Image) -> dict:
    """Classify a PIL image. Returns class name, confidence, and all per-class scores."""
    if not _load():
        raise RuntimeError("InceptionV3 model is not available")
    img = pil_image.convert("RGB").resize((512, 384))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = _model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {
        "class": CLASSES[idx] if idx < len(CLASSES) else "Unknown",
        "confidence": float(preds[idx]),
        "all_scores": {cls: float(preds[i]) for i, cls in enumerate(CLASSES)},
    }
