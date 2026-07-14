from __future__ import annotations

import numpy as np

from piso.methods.common import batch_size, finish, initial_state, record, restore_or_initialize, save_step


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
                problem,
                context.metric_samples,
                rng,
                mu=float(p["mu0"]),
                beta=float(p["beta0"]),
            ),
        )
        if progress is not None:
            progress.update(min(state["sample_count"], context.max_samples) - progress.n)

        while state["sample_count"] <= context.max_samples:
            mk = batch_size(p, state["iteration"])
            state["beta"] *= float(p["beta_decay"])
            direction = rng.normal(size=problem.n)
            plus, _ = problem.sample_losses(state["x"] + state["mu"] * direction, mk, rng)
            minus, _ = problem.sample_losses(state["x"] - state["mu"] * direction, mk, rng)
            gradient = (plus.mean() - minus.mean()) / (2.0 * state["mu"]) * direction
            state["sample_count"] += 2 * mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["mu"] = max(state["mu"] * float(p["mu_decay"]), float(p["mu_min"]))
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
