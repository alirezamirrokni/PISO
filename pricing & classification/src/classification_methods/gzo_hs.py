from __future__ import annotations

import numpy as np

from src.classification_methods.common import (
    batch_size,
    combine_observations,
    exponential,
    finish,
    initial_state,
    known_gradient,
    normalized,
    record,
    restore_or_initialize,
    save_step,
)


class GZOHS:
    name = "GZO_HS"

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
                history=[],
                guide=np.zeros(problem.n, dtype=float),
            ),
        )
        while state["sample_count"] <= context.max_samples:
            k = int(state["iteration"])
            mk = batch_size(p, k)
            beta = exponential(p["beta0"], p["beta_decay"], k)
            direction = (
                np.sqrt(state["alpha"] / problem.n) * rng.normal(size=problem.n)
                + np.sqrt(1.0 - state["alpha"])
                * float(rng.normal())
                * state["guide"]
            )
            plus, plus_obs = problem.sample_losses(
                state["x"] + state["mu"] * direction, mk, rng
            )
            minus, minus_obs = problem.sample_losses(
                state["x"] - state["mu"] * direction, mk, rng
            )
            gradient = (plus.mean() - minus.mean()) / (2.0 * state["mu"]) * direction
            state["history"].append(combine_observations(plus_obs, minus_obs))
            state["sample_count"] += 2 * mk
            state["x"] = state["x"] - beta * gradient

            window = min(int(p["window"]), len(state["history"]))
            gradients = [
                known_gradient(problem, state["x"], state["history"][-1 - offset])
                for offset in range(window)
            ]
            state["guide"] = normalized(np.mean(gradients, axis=0), tolerance=1.0e-5)
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
