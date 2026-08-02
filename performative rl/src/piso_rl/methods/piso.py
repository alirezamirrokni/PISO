from __future__ import annotations

import numpy as np

from .common import (
    batch_size,
    can_afford,
    finish,
    initialize_state,
    record,
    restore_or_initialize,
    save_step,
)
from .piso_utils import (
    allocate_batches,
    guided_covariance_geometry,
    haar_frame,
    known_gradient,
    radius,
    step_size,
    two_level_residual,
    validate_common_params,
    validate_two_level,
)


class _BasePISO:
    variant = "gaussian"
    two_level = False
    name = "PISO"
    seed_alias = "PISO"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        validate_common_params(self.p, self.name)
        if self.two_level:
            validate_two_level(self.p, self.name)
        if self.variant == "guided":
            alpha = float(self.p["alpha"])
            if not 0.0 < alpha <= 1.0:
                raise ValueError(f"{self.name} alpha must lie in (0, 1]")
        if self.variant == "cycle":
            if int(self.p["cycle_length"]) <= 0:
                raise ValueError(f"{self.name} cycle_length must be positive")

    def _initialize(self, problem, rng, evaluation_interval):
        state = initialize_state(problem, evaluation_interval)
        state["residual_momentum"] = np.zeros(problem.n, dtype=float)
        if self.variant == "cycle":
            cycle_length = min(int(self.p["cycle_length"]), problem.n)
            state["direction_frame"] = haar_frame(rng, problem.n, cycle_length)
            state["cycle_position"] = 0
        return state

    def _direction_and_score(
        self,
        problem,
        rng,
        state,
        gradient_known: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        iteration = int(state["iteration"])
        current_radius = radius(self.p, iteration)

        if self.variant == "gaussian":
            direction = rng.normal(size=problem.n)
            return direction, direction, current_radius / np.sqrt(problem.n)

        if self.variant == "cycle":
            frame = np.asarray(state["direction_frame"], dtype=float)
            position = int(state["cycle_position"])
            direction = np.sqrt(problem.n) * frame[:, position]
            return direction, direction, current_radius / np.sqrt(problem.n)

        if gradient_known is None:
            raise RuntimeError("guided PISO requires a known-gradient estimate")
        norm = float(np.linalg.norm(gradient_known))
        hint = (
            gradient_known / norm
            if norm > 0.0
            else np.zeros(problem.n, dtype=float)
        )
        alpha = float(self.p["alpha"])
        parallel, orthogonal = guided_covariance_geometry(alpha, problem.n, hint)
        isotropic = rng.normal(size=problem.n)
        scalar = float(rng.normal())
        direction = (
            np.sqrt(orthogonal) * isotropic
            + np.sqrt(1.0 - alpha) * scalar * hint
        )
        if norm > 0.0:
            projection = float(np.dot(hint, direction))
            parallel_component = projection * hint
            orthogonal_component = direction - parallel_component
            score = parallel_component / parallel + orthogonal_component / orthogonal
        else:
            score = direction / orthogonal
        return direction, score, current_radius

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
        gamma = float(self.p["gamma"])
        rho = float(self.p["rho"])
        mixing = float(self.p.get("lambda", 0.0))

        while True:
            iteration = int(state["iteration"])
            iteration_batch = batch_size(self.p, iteration)
            known_count, plus_count, minus_count = allocate_batches(
                self.p, iteration_batch
            )
            cost = known_count + plus_count + minus_count
            if not can_afford(state, cost, context.max_trajectories):
                break

            # Guided PISO needs g_K before drawing its shaped direction.  The
            # other variants use the same order for deterministic accounting.
            observations = problem.sample_observations(
                state["theta"], known_count, rng
            )
            gradient_known = known_gradient(
                problem, state["theta"], observations
            )
            direction, score, mu = self._direction_and_score(
                problem, rng, state, gradient_known
            )

            plus, _ = problem.sample_losses(
                state["theta"] + mu * direction, plus_count, rng
            )
            minus, _ = problem.sample_losses(
                state["theta"] - mu * direction, minus_count, rng
            )
            derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))

            control = gamma * gradient_known + state["residual_momentum"]
            innovation = derivative - float(np.dot(control, direction))
            residual = state["residual_momentum"] + innovation * score

            if self.two_level:
                new_momentum, residual_direction = two_level_residual(
                    residual,
                    state["residual_momentum"],
                    rho,
                    mixing,
                )
            else:
                new_momentum = (
                    rho * state["residual_momentum"]
                    + (1.0 - rho) * residual
                )
                residual_direction = new_momentum

            gradient = gamma * gradient_known + residual_direction
            eta = step_size(self.p, iteration)

            state["sample_count"] += cost
            state["theta"] = state["theta"] - eta * gradient
            state["residual_momentum"] = new_momentum
            self._advance_cycle(problem, rng, state)
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)

        return finish(state, problem, context.evaluation_interval)


class GaussianPISO(_BasePISO):
    name = "GaussianPISO"
    variant = "gaussian"
    seed_alias = "GaussianPISOFamily"


class GuidedPISO(_BasePISO):
    name = "GuidedPISO"
    variant = "guided"
    seed_alias = "GuidedPISOFamily"


class CyclePISO(_BasePISO):
    name = "CyclePISO"
    variant = "cycle"
    seed_alias = "CyclePISOFamily"


class GaussianPISO2(_BasePISO):
    name = "GaussianPISO2"
    variant = "gaussian"
    two_level = True
    seed_alias = "GaussianPISOFamily"


class GuidedPISO2(_BasePISO):
    name = "GuidedPISO2"
    variant = "guided"
    two_level = True
    seed_alias = "GuidedPISOFamily"


class CyclePISO2(_BasePISO):
    name = "CyclePISO2"
    variant = "cycle"
    two_level = True
    seed_alias = "CyclePISOFamily"
