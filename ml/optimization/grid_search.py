import itertools

import numpy as np

from ml.optimization.base import OptimizationResult, fitness_macro_f1


def run_grid_search(y_true, probs, levels=5):
    dim = probs.shape[1]
    grid_values = np.linspace(0.0, 1.0, levels)
    best_t = np.zeros(dim)
    best_score = -1.0
    history = []

    for candidate in itertools.product(grid_values, repeat=dim):
        t = np.asarray(candidate)
        score = fitness_macro_f1(y_true, probs, t)
        history.append(score)
        if score > best_score:
            best_score = score
            best_t = t

    return OptimizationResult(
        best_thresholds=best_t,
        best_score=best_score,
        history=history,
        metadata={"levels": levels},
    )
