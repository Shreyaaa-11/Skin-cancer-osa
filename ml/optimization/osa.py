import numpy as np

from ml.optimization.base import OptimizationResult, clip_thresholds, fitness_macro_f1


def run_owl_search(
    y_true,
    probs,
    population_size=40,
    n_iter=100,
    beta=1.5,
    seed=42,
):
    """
    Owl Search Algorithm (OSA) for threshold optimization.

    Conceptual mapping:
    - Each owl = one 7D threshold vector.
    - Owls assess prey intensity = fitness (macro-F1).
    - Position updates combine exploration noise + exploitation toward current best owl.
    - Exploration weight decays with iterations to shift from global search to local refinement.
    """
    rng = np.random.default_rng(seed)
    dim = probs.shape[1]
    owls = rng.uniform(0.0, 1.0, size=(population_size, dim))
    fitness = np.array([fitness_macro_f1(y_true, probs, o) for o in owls])

    best_idx = int(np.argmax(fitness))
    best_owl = owls[best_idx].copy()
    best_score = float(fitness[best_idx])
    history = [best_score]

    eps = 1e-9
    for it in range(1, n_iter + 1):
        a = 2.0 * (1.0 - it / n_iter)
        f_min, f_max = float(np.min(fitness)), float(np.max(fitness))

        for i in range(population_size):
            normalized_intensity = (fitness[i] - f_min) / (f_max - f_min + eps)
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            step_exploit = beta * r1 * normalized_intensity * (best_owl - owls[i])
            step_explore = a * (2.0 * r2 - 1.0)
            owls[i] = clip_thresholds(owls[i] + step_exploit + step_explore)

        fitness = np.array([fitness_macro_f1(y_true, probs, o) for o in owls])
        cur_best_idx = int(np.argmax(fitness))
        if fitness[cur_best_idx] > best_score:
            best_score = float(fitness[cur_best_idx])
            best_owl = owls[cur_best_idx].copy()
        history.append(best_score)

    return OptimizationResult(
        best_thresholds=best_owl,
        best_score=best_score,
        history=history,
        metadata={"population_size": population_size, "n_iter": n_iter, "beta": beta, "seed": seed},
    )
