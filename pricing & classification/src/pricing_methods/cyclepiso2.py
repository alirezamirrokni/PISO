from __future__ import annotations

import numpy as np

from src.pricing_methods.common import batch_size, finish, initial_state, record, restore_or_initialize, save_step
from src.pricing_methods.general_piso import (
    allocate_batches,
    haar_frame,
    known_gradient,
    perturbation_radius,
    step_size,
    validate_common_params,
)
from src.pricing_methods.general_piso2 import two_level_residual, validate_two_level_params


class CyclePISO2:
    name = "CyclePISO2"
    private_rng = True
    seed_alias = "GeneralPISO"

    def __init__(self, params: dict) -> None:
        self.p = params
        validate_common_params(params, self.name)
        validate_two_level_params(params, self.name)

    @staticmethod
    def _cycle_length(value, dimension: int) -> int:
        length = int(value)
        if not 1 <= length <= dimension:
            raise ValueError("CyclePISO2 cycle_length must lie in [1, dimension]")
        return length

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        cycle_length = self._cycle_length(p["cycle_length"], problem.n)

        def initialize():
            state = initial_state(problem, context.metric_samples, rng)
            state.update(
                residual_momentum=np.zeros(problem.n, dtype=float),
                direction_frame=haar_frame(rng, problem.n, cycle_length),
                cycle_position=0,
            )
            return state

        state, _ = restore_or_initialize(cache, rng, initialize)
        gamma = float(p["gamma"])
        rho = float(p["rho"])
        mixing = float(p["lambda"])

        while state["sample_count"] <= context.max_samples:
            iteration = int(state["iteration"])
            mk = batch_size(p, iteration)
            known_count, plus_count, minus_count = allocate_batches(p, mk)
            radius = perturbation_radius(p, iteration)
            mu = radius / np.sqrt(problem.n)
            beta = step_size(p, iteration)

            position = int(state["cycle_position"])
            unit_direction = np.asarray(state["direction_frame"][:, position], dtype=float)
            direction = np.sqrt(problem.n) * unit_direction

            plus, _ = problem.sample_losses(state["x"] + mu * direction, plus_count, rng)
            minus, _ = problem.sample_losses(state["x"] - mu * direction, minus_count, rng)
            demands = problem.sample_demands(state["x"], known_count, rng)
            gradient_known = known_gradient(problem, demands)

            derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))
            control = gamma * gradient_known + state["residual_momentum"]
            innovation = derivative - float(np.dot(control, direction))
            residual = state["residual_momentum"] + innovation * direction
            new_momentum, two_level = two_level_residual(
                residual,
                state["residual_momentum"],
                rho,
                mixing,
            )
            gradient = gamma * gradient_known + two_level

            state["sample_count"] += known_count + plus_count + minus_count
            state["x"] = state["x"] - beta * gradient
            state["residual_momentum"] = new_momentum

            position += 1
            if position == cycle_length:
                state["direction_frame"] = haar_frame(rng, problem.n, cycle_length)
                position = 0
            state["cycle_position"] = position
            state["iteration"] = iteration + 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
