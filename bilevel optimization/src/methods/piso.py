from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.methods.common import BaseMethod, MethodTrace
from src.util import stable_seed



def _haar_frame(seed: int, dimension: int, columns: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(dimension, columns))
    q_matrix, r_matrix = np.linalg.qr(gaussian, mode="reduced")
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return q_matrix * signs


def _clip_norm(vector: np.ndarray, maximum: float | None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if maximum is None or maximum <= 0.0:
        return vector
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1.0e-15:
        return vector
    return vector * (maximum / norm)


class PISOBase(BaseMethod):
    """Shared implementation for PISO and PISO^2 bilevel variants.

    The known component is the exact leader partial gradient
    ``g_K = partial_x(x, y*(x))``.  A two-point query estimates a directional
    derivative of the full composite objective.  Subtracting the directional
    projection of ``gamma * g_K + m_k`` produces an innovation for the unknown
    response component, which is incorporated through a randomized Kaczmarz /
    stochastic-approximation update.
    """

    variant = "gaussian"
    two_level = False

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._validate()

    def _validate(self) -> None:
        positive = ("radius", "gamma")
        for key in positive:
            value = float(self.params[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{self.name} {key} must be finite and positive")

        rho = float(self.params["rho"])
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"{self.name} rho must lie in [0, 1)")

        if int(self.params["cycle_length"]) <= 0:
            raise ValueError(f"{self.name} cycle_length must be positive")

        if self.two_level:
            mixing = float(self.params["lambda"])
            if not 0.0 <= mixing <= 1.0:
                raise ValueError(f"{self.name} lambda must lie in [0, 1]")

    def _radius(self, iteration: int) -> float:
        schedule = str(self.params.get("radius_schedule", "constant"))
        radius = float(self.params["radius"])
        if schedule == "constant":
            return radius
        if schedule == "inverse_sqrt":
            return radius / np.sqrt(iteration + 1.0)
        if schedule == "exponential":
            return radius * float(self.params["radius_decay"]) ** iteration
        raise ValueError(f"unknown radius schedule: {schedule}")

    def _direction_and_score(
        self,
        problem: Any,
        base_seed: int,
        instance_id: int,
        iteration: int,
        known_gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        dimension = int(problem.dimension)
        radius = self._radius(iteration)

        if self.variant == "gaussian":
            rng = np.random.default_rng(
                stable_seed(base_seed, problem.family, instance_id, iteration, "piso-gaussian")
            )
            direction = rng.normal(size=dimension)
            return direction, direction, radius / np.sqrt(float(dimension))

        if self.variant == "cycle":
            cycle_length = min(int(self.params["cycle_length"]), dimension)
            block = iteration // cycle_length
            position = iteration % cycle_length
            frame = _haar_frame(
                stable_seed(base_seed, problem.family, instance_id, block, "piso-cycle"),
                dimension,
                cycle_length,
            )
            direction = np.sqrt(float(dimension)) * frame[:, position]
            return direction, direction, radius / np.sqrt(float(dimension))

        raise ValueError(f"unknown PISO direction variant: {self.variant}")


    def run(
        self,
        problem: Any,
        instance_id: int,
        base_seed: int,
        iterations: int,
        checkpoint_interval: int,
        restored: dict[str, Any] | None,
        checkpoint: Callable[[dict[str, Any], bool], None],
    ) -> MethodTrace:
        dimension = int(problem.dimension)

        if restored is None:
            x = problem.initial_decision()
            current_y, current_warm = problem.response(x, None)
            objectives = [problem.objective(x, current_y)]
            oracle_calls = [0]
            momentum = np.zeros(dimension, dtype=float)
            iteration_start = 0
            calls = 0
        else:
            x = np.asarray(restored["x"], dtype=float)
            current_y = np.asarray(restored["current_y"], dtype=float)
            current_warm = restored["current_warm"]
            objectives = list(restored["objectives"])
            oracle_calls = list(restored["oracle_calls"])
            momentum = np.asarray(restored["momentum"], dtype=float)
            iteration_start = int(restored["iteration"])
            calls = int(restored["calls"])

        gamma = float(self.params["gamma"])
        rho = float(self.params["rho"])
        mixing = float(self.params.get("lambda", 0.0))
        innovation_clip = self.params.get("innovation_clip")
        update_clip = self.params.get("update_clip")
        unknown_clip_ratio = float(self.params.get("unknown_clip_ratio", 0.0))
        known_norm_floor = float(self.params.get("known_norm_floor", 1.0))
        innovation_clip_value = None if innovation_clip is None else float(innovation_clip)
        update_clip_value = None if update_clip is None else float(update_clip)

        for iteration in range(iteration_start, iterations):
            # Match PZOS oracle accounting: the central response is treated as
            # one call, followed by the two perturbed response queries.
            calls += 1
            known_gradient = np.asarray(problem.partial_x(x, current_y), dtype=float)
            direction, score, mu = self._direction_and_score(
                problem,
                base_seed,
                instance_id,
                iteration,
                known_gradient,
            )

            plus_y, plus_warm = problem.response(x + mu * direction, current_warm)
            minus_y, minus_warm = problem.response(x - mu * direction, current_warm)
            calls += 2
            plus_value = problem.objective(x + mu * direction, plus_y)
            minus_value = problem.objective(x - mu * direction, minus_y)
            derivative = float((plus_value - minus_value) / (2.0 * mu))

            control = gamma * known_gradient + momentum
            innovation = derivative - float(np.dot(control, direction))
            if innovation_clip_value is not None and innovation_clip_value > 0.0:
                innovation = float(np.clip(innovation, -innovation_clip_value, innovation_clip_value))
            residual = momentum + innovation * score
            new_momentum = rho * momentum + (1.0 - rho) * residual
            if unknown_clip_ratio > 0.0:
                unknown_limit = unknown_clip_ratio * max(
                    float(np.linalg.norm(known_gradient)), known_norm_floor
                )
                new_momentum = _clip_norm(new_momentum, unknown_limit)
                residual = _clip_norm(residual, unknown_limit)

            if self.two_level:
                unknown_component = mixing * residual + (1.0 - mixing) * new_momentum
            else:
                unknown_component = new_momentum
            gradient = gamma * known_gradient + unknown_component
            gradient = _clip_norm(gradient, update_clip_value)

            step = self.step_size(problem, iteration)
            sign = 1.0 if problem.sense == "max" else -1.0
            x = problem.project(x + sign * step * gradient)

            warm_seed = (
                plus_warm
                if np.linalg.norm(plus_y - current_y) <= np.linalg.norm(minus_y - current_y)
                else minus_warm
            )
            current_y, current_warm = problem.response(x, warm_seed)
            objectives.append(problem.objective(x, current_y))
            oracle_calls.append(calls)
            momentum = new_momentum

            state = {
                "iteration": iteration + 1,
                "x": x,
                "current_y": current_y,
                "current_warm": current_warm,
                "objectives": objectives,
                "oracle_calls": oracle_calls,
                "momentum": momentum,
                "calls": calls,
            }
            if (iteration + 1) % checkpoint_interval == 0 or iteration + 1 == iterations:
                checkpoint(state, iteration + 1 == iterations)

        return MethodTrace(
            method=self.name,
            instance_id=instance_id,
            dimension=dimension,
            objectives=np.asarray(objectives, dtype=float),
            oracle_calls=np.asarray(oracle_calls, dtype=int),
        )


class GaussianPISO(PISOBase):
    name = "GaussianPISO"
    variant = "gaussian"


class CyclePISO(PISOBase):
    name = "CyclePISO"
    variant = "cycle"


class GaussianPISO2(PISOBase):
    name = "GaussianPISO2"
    variant = "gaussian"
    two_level = True


class CyclePISO2(PISOBase):
    name = "CyclePISO2"
    variant = "cycle"
    two_level = True
