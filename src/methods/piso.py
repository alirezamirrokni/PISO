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
        alpha0 = float(params["alpha0"])
        damping = float(params["alpha_damping"])
        if not 0.0 <= alpha0 <= 1.0:
            raise ValueError("PISO alpha0 must satisfy 0 <= alpha0 <= 1")
        if not 0.0 <= damping < 1.0:
            raise ValueError("PISO alpha_damping must satisfy 0 <= alpha_damping < 1")

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        damping = float(p["alpha_damping"])
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initial_state(
                problem,
                context.metric_samples,
                rng,
                mu=float(p["mu0"]),
                beta=float(p["beta0"]),
                residual_weight=float(p["alpha0"]),
            ),
        )

        while state["sample_count"] <= context.max_samples:
            mk = batch_size(p, state["iteration"])
            state["beta"] *= float(p["beta_decay"])

            # The known-gradient batch is independent of the perturbation and
            # the two function-estimation batches.
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
            gradient = state["residual_weight"] * residual + known_gradient

            state["sample_count"] += 3 * mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]),
                float(p["mu_min"]),
            )
            state["residual_weight"] = (
                1.0 - damping * (1.0 - state["residual_weight"])
            )
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
