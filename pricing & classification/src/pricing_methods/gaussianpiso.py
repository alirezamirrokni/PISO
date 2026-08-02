from __future__ import annotations

import numpy as np

from src.pricing_methods.common import batch_size, finish, initial_state, record, restore_or_initialize, save_step
from src.pricing_methods.general_piso import (
    allocate_batches,
    known_gradient,
    perturbation_radius,
    step_size,
    validate_common_params,
)


class GaussianPISO:
    name = "GaussianPISO"
    private_rng = True
    seed_alias = "GeneralPISO"

    def __init__(self, params: dict) -> None:
        self.p = params
        validate_common_params(params, self.name)

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p

        def initialize():
            state = initial_state(problem, context.metric_samples, rng)
            state["residual_momentum"] = np.zeros(problem.n, dtype=float)
            return state

        state, _ = restore_or_initialize(cache, rng, initialize)
        gamma = float(p["gamma"])
        rho = float(p["rho"])

        while state["sample_count"] <= context.max_samples:
            iteration = int(state["iteration"])
            mk = batch_size(p, iteration)
            known_count, plus_count, minus_count = allocate_batches(p, mk)
            radius = perturbation_radius(p, iteration)
            mu = radius / np.sqrt(problem.n)
            beta = step_size(p, iteration)

            direction = rng.normal(size=problem.n)
            plus, _ = problem.sample_losses(state["x"] + mu * direction, plus_count, rng)
            minus, _ = problem.sample_losses(state["x"] - mu * direction, minus_count, rng)
            demands = problem.sample_demands(state["x"], known_count, rng)
            gradient_known = known_gradient(problem, demands)

            derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))
            control = gamma * gradient_known + state["residual_momentum"]
            innovation = derivative - float(np.dot(control, direction))
            residual = state["residual_momentum"] + innovation * direction
            new_momentum = rho * state["residual_momentum"] + (1.0 - rho) * residual
            gradient = gamma * gradient_known + new_momentum

            state["sample_count"] += known_count + plus_count + minus_count
            state["x"] = state["x"] - beta * gradient
            state["residual_momentum"] = new_momentum
            state["iteration"] = iteration + 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
