import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.evaluation.metrics import apply_thresholds, evaluate_predictions
from ml.optimization.ga import run_genetic_algorithm
from ml.optimization.grid_search import run_grid_search
from ml.optimization.osa import run_owl_search
from ml.optimization.pso import run_pso
from ml.optimization.random_search import run_random_search


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def maybe_tta_probs(probs: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    uniform = np.full_like(probs, 1.0 / probs.shape[1])
    return (1.0 - alpha) * probs + alpha * uniform


def benchmark_optimizer(name, fn, y_true, probs, runs=3):
    scores, runtimes, histories, thresholds = [], [], [], []
    for seed in range(runs):
        t0 = perf_counter()
        result = fn(y_true, probs, seed)
        dt = (perf_counter() - t0) * 1000.0
        scores.append(result.best_score)
        runtimes.append(dt)
        histories.append(result.history)
        thresholds.append(result.best_thresholds)

    return {
        "name": name,
        "f1_mean": float(np.mean(scores)),
        "f1_std": float(np.std(scores)),
        "latency_ms_mean": float(np.mean(runtimes)),
        "best_thresholds": thresholds[int(np.argmax(scores))].tolist(),
        "best_history": histories[int(np.argmax(scores))],
    }


def main(args):
    artifact_dir = Path("ml/experiments/results")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.predictions_npz)
    y_true = data["y_true"]
    probs = data["probs"]
    probs_tta = maybe_tta_probs(probs, alpha=0.05)

    baseline_pred = np.argmax(probs, axis=1)
    baseline_tta_pred = np.argmax(probs_tta, axis=1)
    baseline_metrics = evaluate_predictions(y_true, baseline_pred, CLASS_NAMES)
    baseline_tta_metrics = evaluate_predictions(y_true, baseline_tta_pred, CLASS_NAMES)

    optimizers = {
        "Grid": lambda yt, pr, s: run_grid_search(yt, pr, levels=3),
        "Random": lambda yt, pr, s: run_random_search(yt, pr, n_iter=200, seed=s),
        "GA": lambda yt, pr, s: run_genetic_algorithm(yt, pr, n_generations=80, seed=s),
        "PSO": lambda yt, pr, s: run_pso(yt, pr, n_iter=100, seed=s),
        "OSA": lambda yt, pr, s: run_owl_search(yt, pr, n_iter=100, seed=s),
    }

    results = [benchmark_optimizer(name, fn, y_true, probs_tta, runs=args.runs) for name, fn in optimizers.items()]
    results_df = pd.DataFrame(results)[["name", "f1_mean", "f1_std", "latency_ms_mean"]]
    results_df.to_csv(artifact_dir / "optimizer_comparison.csv", index=False)

    # Use OSA best thresholds for deployment.
    osa_result = next(r for r in results if r["name"] == "OSA")
    osa_thresholds = np.asarray(osa_result["best_thresholds"])
    y_pred_osa = apply_thresholds(probs_tta, osa_thresholds)
    osa_metrics = evaluate_predictions(y_true, y_pred_osa, CLASS_NAMES)

    payload = {
        "baseline_argmax": baseline_metrics,
        "baseline_tta": baseline_tta_metrics,
        "tta_plus_osa": osa_metrics,
        "optimizer_table": results_df.to_dict(orient="records"),
    }
    with open(artifact_dir / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for row in results:
        plt.plot(row["best_history"], label=row["name"])
    plt.xlabel("Iteration")
    plt.ylabel("Best Macro-F1")
    plt.title("Convergence Curves: Threshold Optimization")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(artifact_dir / "convergence.png", dpi=150)

    Path("backend/artifacts").mkdir(parents=True, exist_ok=True)
    with open("backend/artifacts/thresholds.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": CLASS_NAMES,
                "thresholds": osa_result["best_thresholds"],
                "source": "OSA on validation/test prediction dump",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_npz", type=str, required=True)
    parser.add_argument("--runs", type=int, default=3)
    main(parser.parse_args())
