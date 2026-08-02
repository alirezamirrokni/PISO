from __future__ import annotations

import numpy as np

from src.classification_methods.common import (
    allocate_batches,
    batch_size,
    exponential,
    finish,
    haar_frame,
    initial_state,
    known_gradient,
    normalized,
    record,
    restore_or_initialize,
    save_step,
    validate_piso_params,
)


class ClassificationPISOBase:
    private_rng = True
    seed_alias = "ClassificationPISO"
    variant = "gaussian"
    two_level = False

    def __init__(self, params: dict) -> None:
        self.p = params
        validate_piso_params(params, self.name)
        if self.variant == "guided":
            alpha = float(params["alpha"])
            if not 0.0 < alpha <= 1.0:
                raise ValueError(f"{self.name} alpha must lie in (0, 1]")
        if self.variant == "cycle":
            length = int(params["cycle_length"])
            if length <= 0:
                raise ValueError(f"{self.name} cycle_length must be positive")
        if self.two_level:
            mixing = float(params["lambda"])
            if not 0.0 <= mixing <= 1.0:
                raise ValueError(f"{self.name} lambda must lie in [0, 1]")

    def _initialize(self, problem, context, rng):
        state = initial_state(problem, context.metric_samples, rng)
        state["residual_momentum"] = np.zeros(problem.n, dtype=float)
        if self.variant == "cycle":
            cycle_length = min(int(self.p["cycle_length"]), problem.n)
            state.update(
                direction_frame=haar_frame(rng, problem.n, cycle_length),
                cycle_position=0,
            )
        return state

    def _direction(
        self,
        problem,
        state,
        rng,
        gradient_known: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        iteration = int(state["iteration"])
        radius = exponential(
            self.p["radius"], self.p["radius_decay"], iteration
        )
        if self.variant == "gaussian":
            direction = rng.normal(size=problem.n)
            return direction, direction, radius / np.sqrt(problem.n)

        if self.variant == "cycle":
            cycle_length = min(int(self.p["cycle_length"]), problem.n)
            position = int(state["cycle_position"])
            direction = np.sqrt(problem.n) * np.asarray(
                state["direction_frame"][:, position], dtype=float
            )
            position += 1
            if position == cycle_length:
                state["direction_frame"] = haar_frame(
                    rng, problem.n, cycle_length
                )
                position = 0
            state["cycle_position"] = position
            return direction, direction, radius / np.sqrt(problem.n)

        alpha = float(self.p["alpha"])
        hint = normalized(gradient_known)
        orthogonal = alpha / float(problem.n)
        parallel = (
            1.0 - alpha + orthogonal
            if np.linalg.norm(hint) > 0.0
            else orthogonal
        )
        direction = (
            np.sqrt(orthogonal) * rng.normal(size=problem.n)
            + np.sqrt(1.0 - alpha) * float(rng.normal()) * hint
        )
        if np.linalg.norm(hint) > 0.0:
            projection = float(np.dot(hint, direction))
            parallel_component = projection * hint
            orthogonal_component = direction - parallel_component
            score = (
                parallel_component / parallel
                + orthogonal_component / orthogonal
            )
        else:
            score = direction / orthogonal
        return direction, score, radius

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: self._initialize(problem, context, rng),
        )
        gamma = float(p["gamma"])
        rho = float(p["rho"])
        mixing = float(p.get("lambda", 0.0))

        while state["sample_count"] <= context.max_samples:
            iteration = int(state["iteration"])
            mk = batch_size(p, iteration)
            known_count, plus_count, minus_count = allocate_batches(p, mk)
            beta = exponential(p["step_size"], p["step_decay"], iteration)

            observations = problem.sample_observations(
                state["x"], known_count, rng
            )
            gradient_known = known_gradient(
                problem, state["x"], observations
            )
            direction, score, mu = self._direction(
                problem, state, rng, gradient_known
            )
            plus, _ = problem.sample_losses(
                state["x"] + mu * direction, plus_count, rng
            )
            minus, _ = problem.sample_losses(
                state["x"] - mu * direction, minus_count, rng
            )
            derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))
            momentum = np.asarray(state["residual_momentum"], dtype=float)
            control = gamma * gradient_known + momentum
            innovation = derivative - float(np.dot(control, direction))
            residual = momentum + innovation * score
            new_momentum = rho * momentum + (1.0 - rho) * residual
            if self.two_level:
                residual_term = mixing * residual + (1.0 - mixing) * new_momentum
            else:
                residual_term = new_momentum
            gradient = gamma * gradient_known + residual_term

            state["sample_count"] += known_count + plus_count + minus_count
            state["x"] = state["x"] - beta * gradient
            state["residual_momentum"] = new_momentum
            state["iteration"] = iteration + 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)
        return finish(state)
