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


class ZOOGVR:
    name = "ZO_OGVR"

    def __init__(self, params: dict) -> None:
        self.p = params

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p

        def initialize():
            count = int(p["initial_samples"])
            x = problem.initial_x
            losses, observations = problem.sample_losses(x, count, rng)
            state = initial_state(problem, context.metric_samples, rng)
            state.update(
                sample_count=count,
                mu=float(p["mu0"]),
                control=float(losses.mean()),
                evaluation_points=[x.copy()],
                batch_sizes=[count],
                observation_batches=[observations],
            )
            return state

        state, _ = restore_or_initialize(cache, rng, initialize)
        while state["sample_count"] <= context.max_samples:
            k = int(state["iteration"])
            mk = batch_size(p, k)
            beta_initial = p["beta0"]
            if isinstance(beta_initial, str):
                if beta_initial != "inverse_sqrt_dimension":
                    raise ValueError(
                        "ZO_OGVR beta0 string must be inverse_sqrt_dimension"
                    )
                beta_initial = 1.0 / np.sqrt(problem.n)
            beta = exponential(beta_initial, p["beta_decay"], k)
            direction = rng.normal(size=problem.n)
            point = state["x"] + state["mu"] * direction
            losses, observations = problem.sample_losses(point, mk, rng)
            gradient = (losses.mean() - state["control"]) / state["mu"] * direction
            state["evaluation_points"].append(point.copy())
            state["observation_batches"].append(observations)
            state["sample_count"] += mk
            state["x"] = state["x"] - beta * gradient
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]), float(p["mu_min"])
            )
            state["iteration"] = k + 1

            window = min(int(p["s_max"]), state["iteration"])
            b_values = np.empty(window, dtype=float)
            for offset in range(window):
                source = state["iteration"] - 1 - offset
                b_values[window - 1 - offset] = (
                    float(p["M"])
                    * np.linalg.norm(
                        state["x"] - state["evaluation_points"][source]
                    )
                    ** 2
                    + 1.0 / state["batch_sizes"][source]
                )
            weights = (1.0 / b_values) / np.sum(1.0 / b_values)
            control = 0.0
            for offset in range(window):
                source = state["iteration"] - 1 - offset
                losses_at_x = problem.loss_from_observations(
                    state["x"], state["observation_batches"][source]
                )
                control += weights[window - 1 - offset] * float(losses_at_x.mean())
            state["control"] = control
            state["batch_sizes"].append(mk)
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
