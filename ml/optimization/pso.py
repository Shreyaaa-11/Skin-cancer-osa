import numpy as np

from ml.optimization.base import OptimizationResult, clip_thresholds, fitness_macro_f1


def run_pso(
    y_true,
    probs,
    n_particles=40,
    n_iter=100,
    w=0.7,
    c1=1.6,
    c2=1.6,
    seed=42,
):
    rng = np.random.default_rng(seed)
    dim = probs.shape[1]
    x = rng.uniform(0, 1, size=(n_particles, dim))
    v = rng.normal(0, 0.1, size=(n_particles, dim))
    pbest = x.copy()
    pbest_scores = np.array([fitness_macro_f1(y_true, probs, xi) for xi in x])
    gbest_idx = int(np.argmax(pbest_scores))
    gbest = pbest[gbest_idx].copy()
    gbest_score = float(pbest_scores[gbest_idx])
    history = [gbest_score]

    for _ in range(n_iter):
        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = clip_thresholds(x + v)

        scores = np.array([fitness_macro_f1(y_true, probs, xi) for xi in x])
        improved = scores > pbest_scores
        pbest[improved] = x[improved]
        pbest_scores[improved] = scores[improved]

        gbest_idx = int(np.argmax(pbest_scores))
        if pbest_scores[gbest_idx] > gbest_score:
            gbest_score = float(pbest_scores[gbest_idx])
            gbest = pbest[gbest_idx].copy()
        history.append(gbest_score)

    return OptimizationResult(
        best_thresholds=gbest,
        best_score=gbest_score,
        history=history,
        metadata={"n_particles": n_particles, "n_iter": n_iter, "seed": seed},
    )
