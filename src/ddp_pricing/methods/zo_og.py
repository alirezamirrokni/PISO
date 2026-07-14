from __future__ import annotations

import numpy as np

from ddp_pricing.methods.common import (
    append_metric,
    batch_size,
    make_trace,
    update_after,
    update_before,
)


class ZOOG:
    name = "ZO_OG"

    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self, problem, rng: np.random.RandomState, run_context):
        p = self.params
        x = problem.initial_x
        mu = float(p["mu"])
        beta = float(p["beta0"])
        decay_before = bool(p.get("decay_before_step", True))
        counts = [0]
        values = [problem.evaluate(x, run_context.metric_samples, rng)]
        sample_count = 0
        iteration = 0

        while True:
            mk = batch_size(p, iteration)
            beta = update_before(beta, float(p["beta_decay"]), decay_before)
            direction = rng.normal(size=problem.n)
            losses, _ = problem.sample_losses(x + mu * direction, mk, rng)
            gradient = losses.mean() / mu * direction
            sample_count += mk
            x = x - beta * gradient
            beta = update_after(beta, float(p["beta_decay"]), decay_before)
            append_metric(counts, values, sample_count, x, problem, rng, run_context)
            if sample_count > run_context.max_samples:
                break
            iteration += 1

        return make_trace(self.name, counts, values, x, iterations=iteration + 1, final_samples=sample_count)
