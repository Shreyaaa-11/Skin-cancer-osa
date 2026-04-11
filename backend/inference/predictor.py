from dataclasses import dataclass
from time import perf_counter

import numpy as np

from backend.app.utils.config import settings
from backend.inference.model_loader import load_model_singleton, load_thresholds_singleton
from backend.inference.preprocessing import apply_tta, preprocess_pil_image


@dataclass
class PredictionOutput:
    label: str
    class_index: int
    probabilities: list
    latency_ms: float


class SkinCancerModel:
    def __init__(self):
        self.model = load_model_singleton()
        payload = load_thresholds_singleton()
        self.class_names = payload["class_names"]
        self.thresholds = np.asarray(payload["thresholds"], dtype="float32")

    def preprocess(self, image):
        return preprocess_pil_image(image, image_size=224)

    def apply_tta(self, batch: np.ndarray):
        if not settings.enable_tta:
            return batch
        return apply_tta(batch)

    def apply_thresholds(self, probabilities: np.ndarray) -> int:
        adjusted = probabilities - self.thresholds
        return int(np.argmax(adjusted))

    def predict(self, image) -> PredictionOutput:
        t0 = perf_counter()
        batch = self.preprocess(image)
        batch_aug = self.apply_tta(batch)
        probs = self.model.predict(batch_aug, verbose=0)
        probs_mean = probs.mean(axis=0)
        cls_idx = self.apply_thresholds(probs_mean)
        latency_ms = (perf_counter() - t0) * 1000.0

        return PredictionOutput(
            label=self.class_names[cls_idx],
            class_index=cls_idx,
            probabilities=probs_mean.tolist(),
            latency_ms=float(latency_ms),
        )
