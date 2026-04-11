import numpy as np

from ml.optimization.base import (
    OptimizationResult,
    clip_thresholds,
    fitness_macro_f1,
    sample_uniform_thresholds,
)


def run_genetic_algorithm(
    y_true,
    probs,
    pop_size=40,
    n_generations=80,
    crossover_rate=0.8,
    mutation_sigma=0.08,
    elite_size=4,
    seed=42,
):
    rng = np.random.default_rng(seed)
    dim = probs.shape[1]
    population = np.vstack([sample_uniform_thresholds(rng, dim) for _ in range(pop_size)])
    history = []

    def tournament_select(scores, k=3):
        idx = rng.integers(0, len(scores), size=k)
        return idx[np.argmax(scores[idx])]

    best_t = population[0].copy()
    best_score = -1.0

    for _ in range(n_generations):
        scores = np.array([fitness_macro_f1(y_true, probs, ind) for ind in population])
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_score:
            best_score = float(scores[best_idx])
            best_t = population[best_idx].copy()
        history.append(best_score)

        elite_idx = np.argsort(scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < pop_size:
            p1 = population[tournament_select(scores)]
            p2 = population[tournament_select(scores)]

            if rng.random() < crossover_rate:
                alpha = rng.random(dim)
                child = alpha * p1 + (1.0 - alpha) * p2
            else:
                child = p1.copy()

            mutation = rng.normal(0.0, mutation_sigma, size=dim)
            child = clip_thresholds(child + mutation)
            new_population.append(child)

        population = np.asarray(new_population)

    return OptimizationResult(
        best_thresholds=best_t,
        best_score=best_score,
        history=history,
        metadata={"pop_size": pop_size, "n_generations": n_generations, "seed": seed},
    )
