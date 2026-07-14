from __future__ import annotations

import numpy as np

from ddp_pricing.methods.common import (
    append_metric,
    batch_size,
    make_trace,
    update_after,
    update_before,
)


class GZONS:
    name = "GZO_NS"

    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self, problem, rng: np.random.RandomState, run_context):
        p = self.params
        x = problem.initial_x
        mu = float(p["mu0"])
        beta = float(p["beta0"])
        alpha = float(p["alpha0"])
        decay_before = bool(p.get("decay_before_step", True))
        counts = [0]
        values = [problem.evaluate(x, run_context.metric_samples, rng)]
        sample_count = 0
        iteration = 0

        while True:
            mk = batch_size(p, iteration)
            beta = update_before(beta, float(p["beta_decay"]), decay_before)

            guide_demands = problem.sample_demands(x, mk, rng)
            guide = problem.partial_gradients(guide_demands).mean(axis=0)
            norm = np.linalg.norm(guide)
            if norm > 0.0:
                guide = guide / norm
            scalar = rng.normal()
            isotropic = rng.normal(size=problem.n)
            direction = np.sqrt(alpha / problem.n) * isotropic + np.sqrt(1.0 - alpha) * scalar * guide
            sample_count += mk

            plus = x + mu * direction
            minus = x - mu * direction
            plus_losses, _ = problem.sample_losses(plus, mk, rng)
            minus_losses, _ = problem.sample_losses(minus, mk, rng)
            gradient = (plus_losses.mean() - minus_losses.mean()) / (2.0 * mu) * direction
            sample_count += 2 * mk

            x = x - beta * gradient
            mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
            alpha = 1.0 - float(p["alpha_damping"]) * (1.0 - alpha)
            beta = update_after(beta, float(p["beta_decay"]), decay_before)
            iteration += 1
            append_metric(counts, values, sample_count, x, problem, rng, run_context)
            if sample_count > run_context.max_samples:
                break

        return make_trace(self.name, counts, values, x, iterations=iteration, final_samples=sample_count)
