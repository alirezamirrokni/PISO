from __future__ import annotations

import numpy as np

from src.methods.common import (
    batch_size,
    finish,
    initial_state,
    record,
    restore_or_initialize,
    save_step,
)


class PISO:
    name = "PISO"
    private_rng = True

    def __init__(self, params: dict) -> None:
        self.p = params
        residual_alpha = float(params["residual_alpha"])
        if not 0.0 < residual_alpha <= 1.0:
            raise ValueError("PISO residual_alpha must satisfy 0 < residual_alpha <= 1")

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        residual_alpha = float(p["residual_alpha"])
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

        while state["sample_count"] <= context.max_samples:
            mk = batch_size(p, state["iteration"])
            state["beta"] *= float(p["beta_decay"])

            # The known-gradient batch is independent of both u and the two
            # function-estimation batches, as required by the control variate.
            direction = rng.normal(size=problem.n)
            plus, _ = problem.sample_losses(
                state["x"] + state["mu"] * direction,
                mk,
                rng,
            )
            minus, _ = problem.sample_losses(
                state["x"] - state["mu"] * direction,
                mk,
                rng,
            )
            known_demands = problem.sample_demands(state["x"], mk, rng)
            known_gradient = problem.partial_gradients(known_demands).mean(axis=0)

            finite_difference = (
                (plus.mean() - minus.mean()) / (2.0 * state["mu"])
            ) * direction
            residual = (
                finite_difference
                - float(np.dot(known_gradient, direction)) * direction
            )
            gradient = residual_alpha * residual + known_gradient

            state["sample_count"] += 3 * mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]),
                float(p["mu_min"]),
            )
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
