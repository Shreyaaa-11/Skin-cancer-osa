import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.app.utils.config import settings


@lru_cache(maxsize=1)
def load_model_singleton() -> tf.keras.Model:
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


@lru_cache(maxsize=1)
def load_thresholds_singleton():
    thresholds_path = Path(settings.thresholds_path)
    if not thresholds_path.exists():
        return {
            "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            "thresholds": [0.0] * 7,
        }
    with open(thresholds_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["thresholds"] = np.asarray(payload["thresholds"], dtype="float32")
    return payload
