from __future__ import annotations

import numpy as np

from src.classification_methods.common import (
    batch_size,
    exponential,
    finish,
    guide_batch_size,
    initial_state,
    known_gradient,
    normalized,
    record,
    restore_or_initialize,
    save_step,
)


class GZONS:
    name = "GZO_NS"

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
                alpha=float(p["alpha0"]),
            ),
        )
        while state["sample_count"] <= context.max_samples:
            k = int(state["iteration"])
            mk = batch_size(p, k)
            nk = guide_batch_size(p, k)
            beta = exponential(p["beta0"], p["beta_decay"], k)

            guide_obs = problem.sample_observations(state["x"], nk, rng)
            guide = normalized(known_gradient(problem, state["x"], guide_obs))
            direction = (
                np.sqrt(state["alpha"] / problem.n) * rng.normal(size=problem.n)
                + np.sqrt(1.0 - state["alpha"]) * float(rng.normal()) * guide
            )
            plus, _ = problem.sample_losses(
                state["x"] + state["mu"] * direction, mk, rng
            )
            minus, _ = problem.sample_losses(
                state["x"] - state["mu"] * direction, mk, rng
            )
            gradient = (plus.mean() - minus.mean()) / (2.0 * state["mu"]) * direction
            state["sample_count"] += nk + 2 * mk
            state["x"] = state["x"] - beta * gradient
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]), float(p["mu_min"])
            )
            state["alpha"] = 1.0 - float(p["alpha_damping"]) * (
                1.0 - state["alpha"]
            )
            state["iteration"] = k + 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
