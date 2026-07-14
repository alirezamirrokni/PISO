from __future__ import annotations

import numpy as np

from piso.methods.common import batch_size, finish, initial_state, record, restore_or_initialize, save_step


class ZOOG:
    name = "ZO_OG"

    def __init__(self, params: dict) -> None:
        self.p = params

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initial_state(
                problem,
                context.metric_samples,
                rng,
                mu=float(p["mu"]),
                beta=float(p["beta0"]),
            ),
        )
        if progress is not None:
            progress.update(min(state["sample_count"], context.max_samples) - progress.n)

        while state["sample_count"] <= context.max_samples:
            mk = batch_size(p, state["iteration"])
            state["beta"] *= float(p["beta_decay"])
            direction = rng.normal(size=problem.n)
            losses, _ = problem.sample_losses(state["x"] + state["mu"] * direction, mk, rng)
            gradient = losses.mean() / state["mu"] * direction
            state["sample_count"] += mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
