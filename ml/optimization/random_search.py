import numpy as np

from ml.optimization.base import OptimizationResult, fitness_macro_f1, sample_uniform_thresholds


def run_random_search(y_true, probs, n_iter=200, seed=42):
    dim = probs.shape[1]
    rng = np.random.default_rng(seed)
    best_t = np.zeros(dim)
    best_score = -1.0
    history = []

    for _ in range(n_iter):
        t = sample_uniform_thresholds(rng, dim)
        score = fitness_macro_f1(y_true, probs, t)
        history.append(max(history[-1], score) if history else score)
        if score > best_score:
            best_score = score
            best_t = t

    return OptimizationResult(
        best_thresholds=best_t,
        best_score=best_score,
        history=history,
        metadata={"n_iter": n_iter, "seed": seed},
    )
