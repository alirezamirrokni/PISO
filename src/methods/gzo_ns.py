from __future__ import annotations

import numpy as np

from src.methods.common import batch_size, finish, initial_state, record, restore_or_initialize, save_step


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
                beta=float(p["beta0"]),
                alpha=float(p["alpha0"]),
            ),
        )
        if progress is not None:
            progress.update(min(state["sample_count"], context.max_samples) - progress.n)

        while state["sample_count"] <= context.max_samples:
            k = state["iteration"]
            mk = batch_size(p, k)
            state["beta"] *= float(p["beta_decay"])
            guide_demands = problem.sample_demands(state["x"], mk, rng)
            guide = problem.partial_gradients(guide_demands).mean(axis=0)
            norm = np.linalg.norm(guide)
            if norm > 0:
                guide /= norm
            scalar = rng.normal()
            isotropic = rng.normal(size=problem.n)
            direction = (
                np.sqrt(state["alpha"] / problem.n) * isotropic
                + np.sqrt(1.0 - state["alpha"]) * scalar * guide
            )
            state["sample_count"] += mk
            plus, _ = problem.sample_losses(state["x"] + state["mu"] * direction, mk, rng)
            minus, _ = problem.sample_losses(state["x"] - state["mu"] * direction, mk, rng)
            gradient = (plus.mean() - minus.mean()) / (2.0 * state["mu"]) * direction
            state["sample_count"] += 2 * mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["mu"] = max(state["mu"] * float(p["mu_decay"]), float(p["mu_min"]))
            state["alpha"] = 1.0 - float(p["alpha_damping"]) * (1.0 - state["alpha"])
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
