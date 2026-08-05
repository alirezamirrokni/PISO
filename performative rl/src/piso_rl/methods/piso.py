from __future__ import annotations

import numpy as np

from .common import (
    batch_size,
    can_afford,
    clip_gradient,
    finish,
    initialize_state,
    record,
    restore_or_initialize,
    save_step,
)
from .piso_utils import (
    allocate_batches,
    clip_vector,
    haar_frame,
    known_gradient,
    radius,
    split_direction_batches,
    step_size,
    two_level_residual,
    update_residual_rls,
    validate_common_params,
    validate_two_level,
)


class _BasePISO:
    """Policy-informed stochastic optimisation using deployable queries only.

    Let ``F(theta) = -J(theta, theta)`` be the loss on the performative
    diagonal.  PISO combines a score-function estimate of the ordinary policy
    component with directional finite differences of the *same diagonal
    objective*.  For direction ``u``,

        dF(theta)[u] = partial_policy F(theta, theta)[u]
                     + partial_shift  F(theta, theta)[u].

    The shift observation is therefore the diagonal finite difference minus the
    current policy-gradient projection.  No query holds the acting policy fixed
    while changing only the environment-inducing policy; that cross-deployment
    operation is not available to Vanilla PG or the ZO/GZO baselines and would
    make the comparison unfair.
    """

    variant = "gaussian"
    two_level = False
    name = "PISO"
    seed_alias = "PISO"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        validate_common_params(self.p, self.name)
        if self.two_level:
            validate_two_level(self.p, self.name)
        if self.variant == "cycle" and int(self.p["cycle_length"]) <= 0:
            raise ValueError(f"{self.name} cycle_length must be positive")

    def _initialize(self, problem, rng, evaluation_interval):
        state = initialize_state(problem, evaluation_interval)
        state["residual_gram"] = np.zeros((problem.n, problem.n), dtype=float)
        state["residual_rhs"] = np.zeros(problem.n, dtype=float)
        state["residual_estimate"] = np.zeros(problem.n, dtype=float)
        state["fast_residual"] = np.zeros(problem.n, dtype=float)
        state["residual_observations"] = 0
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
        state.setdefault("residual_gram", np.zeros((problem.n, problem.n)))
        state.setdefault("residual_rhs", np.zeros(problem.n))
        state.setdefault("residual_estimate", np.zeros(problem.n))
        state.setdefault("fast_residual", np.zeros(problem.n))
        state.setdefault("residual_observations", 0)

        known_weight = float(self.p["gamma"])
        fast_decay = float(self.p["rho"])
        mixing = float(self.p.get("lambda", 0.0))
        directions_per_update = int(self.p.get("directions_per_update", 1))

        while True:
            iteration = int(state["iteration"])
            iteration_batch = batch_size(self.p, iteration)
            known_count, plus_count, minus_count = allocate_batches(
                self.p, iteration_batch
            )
            if plus_count != minus_count:
                raise RuntimeError("paired PISO queries require equal batch sizes")
            cost = known_count + plus_count + minus_count
            if not can_afford(state, cost, context.max_trajectories):
                break

            theta = np.asarray(state["theta"], dtype=float)
            observations = problem.sample_observations(theta, known_count, rng)
            gradient_known = known_gradient(problem, theta, observations)

            direction_batches = split_direction_batches(
                plus_count, directions_per_update
            )
            directions: list[np.ndarray] = []
            scores: list[np.ndarray] = []
            derivatives: list[float] = []
            derivative_clip = self.p.get("directional_derivative_clip")

            for query_count in direction_batches:
                direction, score, mu = self._direction_and_score(
                    problem, rng, state
                )
                plus, _, minus, _ = problem.sample_paired_losses(
                    theta + mu * direction,
                    theta - mu * direction,
                    query_count,
                    rng,
                )
                diagonal_derivative = float(
                    (plus.mean() - minus.mean()) / (2.0 * mu)
                )
                policy_derivative = float(np.dot(gradient_known, direction))
                shift_derivative = diagonal_derivative - policy_derivative
                if derivative_clip is not None:
                    limit = float(derivative_clip)
                    shift_derivative = float(
                        np.clip(shift_derivative, -limit, limit)
                    )
                directions.append(np.asarray(direction, dtype=float))
                scores.append(np.asarray(score, dtype=float))
                derivatives.append(shift_derivative)
                self._advance_cycle(problem, rng, state)

            slow_residual = update_residual_rls(
                state,
                directions,
                derivatives,
                forgetting=float(self.p.get("rls_forgetting", 1.0)),
                ridge=float(self.p.get("rls_ridge", 1.0)),
                clip_norm=self.p.get("residual_clip_norm"),
                warmup_directions=int(
                    self.p.get("residual_warmup_directions", problem.n)
                ),
            )

            instant_residual = np.mean(
                [value * score for value, score in zip(derivatives, scores)],
                axis=0,
            )
            instant_residual = clip_vector(
                instant_residual, self.p.get("residual_clip_norm")
            )
            fast_residual = (
                fast_decay * np.asarray(state["fast_residual"], dtype=float)
                + (1.0 - fast_decay) * instant_residual
            )
            fast_residual = clip_vector(
                fast_residual, self.p.get("residual_clip_norm")
            )
            state["fast_residual"] = fast_residual

            residual_direction = (
                two_level_residual(slow_residual, fast_residual, mixing)
                if self.two_level
                else slow_residual
            )
            gradient = known_weight * gradient_known + residual_direction
            gradient = clip_gradient(gradient, self.p.get("gradient_clip_norm"))
            eta = step_size(self.p, iteration, int(state["sample_count"]))

            state["sample_count"] += cost
            state["theta"] = theta - eta * gradient
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)

        return finish(state, problem, context.evaluation_interval)


class GaussianPISO(_BasePISO):
    name = "GaussianPISO"
    variant = "gaussian"
    seed_alias = "GaussianPISOFamily"


class CyclePISO(_BasePISO):
    name = "CyclePISO"
    variant = "cycle"
    seed_alias = "CyclePISOFamily"


class GaussianPISO2(_BasePISO):
    name = "GaussianPISO2"
    variant = "gaussian"
    two_level = True
    seed_alias = "GaussianPISOFamily"


class CyclePISO2(_BasePISO):
    name = "CyclePISO2"
    variant = "cycle"
    two_level = True
    seed_alias = "CyclePISOFamily"
