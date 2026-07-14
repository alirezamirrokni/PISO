from __future__ import annotations

import numpy as np

from ddp_pricing.methods.common import (
    append_metric,
    batch_size,
    make_trace,
    resolve_beta0,
    update_after,
    update_before,
)


class ZOTG:
    name = "ZO_TG"

    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self, problem, rng: np.random.RandomState, run_context):
        p = self.params
        x = problem.initial_x
        mu = float(p["mu0"])
        beta = resolve_beta0(p, problem.n)
        decay_before = bool(p.get("decay_before_step", True))
        scale = 1.0 if p.get("direction_covariance") == "identity" else 1.0 / np.sqrt(problem.n)
        counts = [0]
        values = [problem.evaluate(x, run_context.metric_samples, rng)]
        sample_count = 0
        iteration = 0

        while True:
            mk = batch_size(p, iteration)
            beta = update_before(beta, float(p["beta_decay"]), decay_before)
            direction = scale * rng.normal(size=problem.n)
            plus_losses, _ = problem.sample_losses(x + mu * direction, mk, rng)
            minus_losses, _ = problem.sample_losses(x - mu * direction, mk, rng)
            gradient = (plus_losses.mean() - minus_losses.mean()) / (2.0 * mu) * direction
            sample_count += 2 * mk
            x = x - beta * gradient
            mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
            beta = update_after(beta, float(p["beta_decay"]), decay_before)
            iteration += 1
            append_metric(counts, values, sample_count, x, problem, rng, run_context)
            if sample_count > run_context.max_samples:
                break

        return make_trace(self.name, counts, values, x, iterations=iteration, final_samples=sample_count)
