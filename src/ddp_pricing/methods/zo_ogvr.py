from __future__ import annotations

import numpy as np

from ddp_pricing.methods.common import (
    append_metric,
    batch_size,
    make_trace,
    update_after,
    update_before,
)


class ZOOGVR:
    name = "ZO_OGVR"

    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self, problem, rng: np.random.RandomState, run_context):
        p = self.params
        x = problem.initial_x
        initial_samples = int(p["initial_samples"])
        initial_losses, initial_demands = problem.sample_losses(x, initial_samples, rng)
        control = float(initial_losses.mean())
        mu = float(p["mu0"])
        beta = float(p["beta0"])
        decay_before = bool(p.get("decay_before_step", True))
        counts = [0]
        values = [problem.evaluate(x, run_context.metric_samples, rng)]
        sample_count = initial_samples
        iteration = 0

        evaluation_points: list[np.ndarray] = [x.copy()]
        batch_sizes: list[int] = [initial_samples]
        demand_batches: list[np.ndarray] = [initial_demands]

        while True:
            mk = batch_size(p, iteration)
            beta = update_before(beta, float(p["beta_decay"]), decay_before)
            direction = rng.normal(size=problem.n)
            evaluation_point = x + mu * direction
            evaluation_points.append(evaluation_point.copy())
            losses, demands = problem.sample_losses(evaluation_point, mk, rng)
            demand_batches.append(demands)
            gradient = (losses.mean() - control) / mu * direction
            sample_count += mk
            x = x - beta * gradient
            mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
            beta = update_after(beta, float(p["beta_decay"]), decay_before)
            iteration += 1

            s = min(int(p["s_max"]), iteration)
            b = np.zeros(s, dtype=float)
            for i in range(s):
                source = iteration - 1 - i
                b[s - 1 - i] = (
                    float(p["M"]) * np.linalg.norm(x - evaluation_points[source]) ** 2
                    + 1.0 / batch_sizes[source]
                )
            weights = (1.0 / b) / np.sum(1.0 / b)
            control = 0.0
            for i in range(s):
                source = iteration - 1 - i
                losses_at_x = problem.loss_from_demands(x, demand_batches[source])
                control += weights[s - 1 - i] * float(losses_at_x.mean())

            batch_sizes.append(mk)
            append_metric(counts, values, sample_count, x, problem, rng, run_context)
            if sample_count > run_context.max_samples:
                break

        return make_trace(self.name, counts, values, x, iterations=iteration, final_samples=sample_count)
