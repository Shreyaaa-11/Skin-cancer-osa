from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score

from ml.evaluation.metrics import apply_thresholds


@dataclass
class OptimizationResult:
    best_thresholds: np.ndarray
    best_score: float
    history: List[float]
    metadata: Dict


def fitness_macro_f1(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> float:
    y_pred = apply_thresholds(probs, thresholds)
    return float(f1_score(y_true, y_pred, average="macro"))


def sample_uniform_thresholds(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(dim,))


def clip_thresholds(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)
