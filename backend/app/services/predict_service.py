import logging
from functools import lru_cache

from backend.inference.predictor import SkinCancerModel
from backend.inference.preprocessing import read_and_validate_image

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> SkinCancerModel:
    return SkinCancerModel()


def run_prediction(file_bytes: bytes):
    image = read_and_validate_image(file_bytes)
    pred = get_model().predict(image)
    logger.info("prediction_done class=%s latency_ms=%.2f", pred.label, pred.latency_ms)
    return pred
