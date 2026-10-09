from __future__ import annotations

import numpy as np

from .common import (batch_size, can_afford, finish, initialize_state, record,
                     restore_or_initialize, save_step)
from .piso_utils import (allocate_batches, haar_frame, known_gradient, radius,
                         step_size, validate_common_params, validate_two_level)


class _BasePISO:

    variant = "gaussian"
    two_level = False
    name = "PISO"
    seed_alias = "PISO"
    cache_implementation = "centered-v1-independent-warmstart"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        validate_common_params(self.p, self.name)
        if not self.two_level and float(self.p.get("lambda", 0.0)) != 0.0:
            raise ValueError("PISO requires lambda=0; use PISO2 for readout mixing")
        if self.two_level:
            validate_two_level(self.p, self.name)
        if self.variant == "cycle" and int(self.p["cycle_length"]) <= 0:
            raise ValueError(f"{self.name} cycle_length must be positive")

    def _initialize(self, problem, rng, evaluation_interval):
        state = initialize_state(problem, evaluation_interval)
        state["residual_state"] = np.zeros(problem.n, dtype=float)
        state["known_state"] = None
        if self.variant == "cycle":
            cycle_length = min(int(self.p["cycle_length"]), problem.n)
            state["direction_frame"] = haar_frame(rng, problem.n, cycle_length)
            state["cycle_position"] = 0
        return state

    def _direction_and_score(self, problem, rng, state):
        iteration = int(state["iteration"])
        current_radius = radius(self.p, iteration, int(state["sample_count"]))

        if self.variant == "gaussian":
            direction = rng.normal(size=problem.n)
            return direction, direction, current_radius / np.sqrt(problem.n)

        if self.variant == "cycle":
            frame = np.asarray(state["direction_frame"], dtype=float)
            position = int(state["cycle_position"])
            direction = np.sqrt(problem.n) * frame[:, position]
            return direction, direction, current_radius / np.sqrt(problem.n)

        raise RuntimeError(f"unsupported PISO variant: {self.variant}")

    def _advance_cycle(self, problem, rng, state) -> None:
        if self.variant != "cycle":
            return
        cycle_length = min(int(self.p["cycle_length"]), problem.n)
        position = int(state["cycle_position"]) + 1
        if position == cycle_length:
            state["direction_frame"] = haar_frame(rng, problem.n, cycle_length)
            position = 0
        state["cycle_position"] = position

    def run(self, problem, rng, context, cache):
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: self._initialize(problem, rng, context.evaluation_interval),
            context.evaluation_interval,
        )
        if "residual_state" not in state or "known_state" not in state:
            raise ValueError("Checkpoint is not a centered Algorithm 1 state")

        gamma = float(self.p["gamma"])
        rho = float(self.p["rho"])
        rho_k = float(self.p.get("rho_k", 0.0))
        mixing = float(self.p.get("lambda", 0.0))
        while True:
            iteration = int(state["iteration"])
            iteration_batch = batch_size(self.p, iteration)
            known_count, plus_count, minus_count = allocate_batches(
                self.p, iteration_batch
            )
            if plus_count != minus_count:
                raise RuntimeError("paired PISO queries require equal batch sizes")
            if int(self.p.get("directions_per_update", 1)) != 1:
                raise ValueError("exact Algorithm 1 PISO requires one direction per update")
            cost = known_count + plus_count + minus_count
            warm_cost = known_count if state["known_state"] is None else 0
            if not can_afford(state, cost + warm_cost, context.max_trajectories):
                break

            theta = np.asarray(state["theta"], dtype=float)
            if warm_cost:
                warm_observations = problem.sample_observations(theta, warm_cost, rng)
                state["known_state"] = known_gradient(problem, theta, warm_observations)
                state["sample_count"] += warm_cost
            observations = problem.sample_observations(theta, known_count, rng)
            gradient_known = known_gradient(problem, theta, observations)
            direction, score, mu = self._direction_and_score(
                problem, rng, state
            )
            plus, _, minus, _ = problem.sample_paired_losses(
                theta + mu * direction,
                theta - mu * direction,
                plus_count,
                rng,
            )
            directional = float((plus.mean() - minus.mean()) / (2.0 * mu))

            residual = np.asarray(state["residual_state"], dtype=float)
            innovation = directional - float(
                np.dot(gamma * gradient_known + residual, direction)
            )
            residual_observation = residual + innovation * np.asarray(score)
            residual_next = rho * residual + (1.0 - rho) * residual_observation

            known_previous = state.get("known_state")
            known_next = (
                rho_k * np.asarray(known_previous, dtype=float)
                + (1.0 - rho_k) * gradient_known
            )
            readout = mixing * residual_observation + (1.0 - mixing) * residual_next
            gradient = gamma * known_next + readout
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Algorithm 1 produced a non-finite gradient")
            eta = step_size(self.p, iteration, int(state["sample_count"]))

            state["residual_state"] = residual_next
            state["known_state"] = known_next
            state["sample_count"] += cost
            state["theta"] = theta - eta * gradient
            state["iteration"] = iteration + 1
            self._advance_cycle(problem, rng, state)
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)

        return finish(state, problem, context.evaluation_interval)


class GaussianPISO(_BasePISO):
    name = "GaussianPISO"
    variant = "gaussian"
    seed_alias = "GaussianPISOAlgorithm1Family"


class CyclePISO(_BasePISO):
    name = "CyclePISO"
    variant = "cycle"
    seed_alias = "CyclePISOAlgorithm1Family"


class GaussianPISO2(_BasePISO):
    name = "GaussianPISO2"
    variant = "gaussian"
    two_level = True
    seed_alias = "GaussianPISOAlgorithm1Family"


class CyclePISO2(_BasePISO):
    name = "CyclePISO2"
    variant = "cycle"
    two_level = True
    seed_alias = "CyclePISOAlgorithm1Family"
