from __future__ import annotations

import numpy as np

from ddp_pricing.methods.common import (
    append_metric,
    batch_size,
    make_trace,
    update_after,
    update_before,
)


class GZOHS:
    name = "GZO_HS"

    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self, problem, rng: np.random.RandomState, run_context):
        p = self.params
        x = problem.initial_x
        mu = float(p["mu0"])
        beta = float(p["beta0"])
        alpha = float(p["alpha0"])
        window = int(p["window"])
        behavior = str(p.get("behavior", "reference"))
        decay_before = bool(p.get("decay_before_step", True))
        counts = [0]
        values = [problem.evaluate(x, run_context.metric_samples, rng)]
        sample_count = 0
        iteration = 0
        historical_batches: list[np.ndarray] = []
        guide: np.ndarray | float = 0.0

        while True:
            mk = batch_size(p, iteration)
            beta = update_before(beta, float(p["beta_decay"]), decay_before)

            scalar = rng.normal()
            isotropic = rng.normal(size=problem.n)
            direction = np.sqrt(alpha / problem.n) * isotropic + np.sqrt(1.0 - alpha) * scalar * guide

            plus = x + mu * direction
            minus = x - mu * direction
            plus_losses, plus_demands = problem.sample_losses(plus, mk, rng)
            minus_losses, minus_demands = problem.sample_losses(minus, mk, rng)
            gradient = (plus_losses.mean() - minus_losses.mean()) / (2.0 * mu) * direction
            historical_batches.append(np.concatenate([plus_demands, minus_demands], axis=0))
            sample_count += 2 * mk

            x = x - beta * gradient

            if behavior == "reference":
                active = min(window, iteration)
                guide_acc: np.ndarray | float = 0.0
                for offset in range(active):
                    selected = historical_batches[iteration - 1 - offset]
                    guide_acc = guide_acc + problem.partial_gradients(selected).mean(axis=0)
                guide = guide_acc / window
            else:
                active = min(window, len(historical_batches))
                selected_batches = historical_batches[-active:]
                guide = sum(
                    (problem.partial_gradients(batch).mean(axis=0) for batch in selected_batches),
                    start=np.zeros(problem.n),
                ) / active

            if np.linalg.norm(guide) > 1e-5:
                guide = guide / np.linalg.norm(guide)

            mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
            alpha = 1.0 - float(p["alpha_damping"]) * (1.0 - alpha)
            beta = update_after(beta, float(p["beta_decay"]), decay_before)
            iteration += 1
            append_metric(counts, values, sample_count, x, problem, rng, run_context)
            if sample_count > run_context.max_samples:
                break

        return make_trace(self.name, counts, values, x, iterations=iteration, final_samples=sample_count)
