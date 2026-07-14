from __future__ import annotations

import numpy as np

from src.methods.common import batch_size, finish, initial_state, record, restore_or_initialize, save_step


class ZOOGVR:
    name = "ZO_OGVR"

    def __init__(self, params: dict) -> None:
        self.p = params

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p

        def initialize():
            initial_samples = int(p["initial_samples"])
            x = problem.initial_x
            losses, demands = problem.sample_losses(x, initial_samples, rng)
            state = {
                "x": x,
                "samples": [0],
                "objectives": [problem.evaluate(x, context.metric_samples, rng)],
                "sample_count": initial_samples,
                "iteration": 0,
                "mu": float(p["mu0"]),
                "beta": float(p["beta0"]),
                "control": float(losses.mean()),
                "evaluation_points": [x.copy()],
                "batch_sizes": [initial_samples],
                "demand_batches": [demands],
            }
            return state

        state, _ = restore_or_initialize(cache, rng, initialize)
        if progress is not None:
            progress.update(min(state["sample_count"], context.max_samples) - progress.n)

        while state["sample_count"] <= context.max_samples:
            k = state["iteration"]
            mk = batch_size(p, k)
            state["beta"] *= float(p["beta_decay"])
            direction = rng.normal(size=problem.n)
            point = state["x"] + state["mu"] * direction
            state["evaluation_points"].append(point.copy())
            losses, demands = problem.sample_losses(point, mk, rng)
            state["demand_batches"].append(demands)
            gradient = (losses.mean() - state["control"]) / state["mu"] * direction
            state["sample_count"] += mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["mu"] = max(state["mu"] * float(p["mu_decay"]), float(p["mu_min"]))
            state["iteration"] += 1

            s = min(int(p["s_max"]), state["iteration"])
            b = np.zeros(s)
            for i in range(s):
                source = state["iteration"] - 1 - i
                b[s - 1 - i] = (
                    float(p["M"]) * np.linalg.norm(state["x"] - state["evaluation_points"][source]) ** 2
                    + 1.0 / state["batch_sizes"][source]
                )
            weights = (1.0 / b) / (1.0 / b).sum()
            state["control"] = 0.0
            for i in range(s):
                source = state["iteration"] - 1 - i
                losses_at_x = problem.loss_from_demands(state["x"], state["demand_batches"][source])
                state["control"] += weights[s - 1 - i] * float(losses_at_x.mean())
            state["batch_sizes"].append(mk)
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
