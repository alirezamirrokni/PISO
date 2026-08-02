from __future__ import annotations

import numpy as np

from src.classification_methods.common import (
    batch_size,
    exponential,
    finish,
    initial_state,
    record,
    restore_or_initialize,
    save_step,
)


class ZOOG:
    name = "ZO_OG"

    def __init__(self, params: dict) -> None:
        self.p = params

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initial_state(problem, context.metric_samples, rng),
        )
        while state["sample_count"] <= context.max_samples:
            k = int(state["iteration"])
            mk = batch_size(p, k)
            beta = exponential(p["beta0"], p.get("beta_decay", 1.0), k)
            mu = float(p["mu"])
            direction = rng.normal(size=problem.n)
            losses, _ = problem.sample_losses(state["x"] + mu * direction, mk, rng)
            gradient = losses.mean() / mu * direction
            state["sample_count"] += mk
            state["x"] = state["x"] - beta * gradient
            state["iteration"] = k + 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
