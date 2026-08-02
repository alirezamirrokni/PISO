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


class ZOTG:
    name = "ZO_TG"

    def __init__(self, params: dict) -> None:
        self.p = params

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initial_state(
                problem, context.metric_samples, rng, mu=float(p["mu0"])
            ),
        )
        while state["sample_count"] <= context.max_samples:
            k = int(state["iteration"])
            mk = batch_size(p, k)
            beta = exponential(p["beta0"], p["beta_decay"], k)
            direction = rng.normal(size=problem.n) / np.sqrt(problem.n)
            plus, _ = problem.sample_losses(
                state["x"] + state["mu"] * direction, mk, rng
            )
            minus, _ = problem.sample_losses(
                state["x"] - state["mu"] * direction, mk, rng
            )
            gradient = (plus.mean() - minus.mean()) / (2.0 * state["mu"]) * direction
            state["sample_count"] += 2 * mk
            state["x"] = state["x"] - beta * gradient
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]), float(p["mu_min"])
            )
            state["iteration"] = k + 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
