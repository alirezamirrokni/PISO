from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.methods.common import BaseMethod, MethodTrace
from src.util import stable_seed


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        return np.zeros_like(vector)
    return vector / norm


def _clip_norm(vector: np.ndarray, maximum: float | None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if maximum is None or maximum <= 0.0:
        return vector
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1.0e-15:
        return vector
    return vector * (maximum / norm)


def _warm_from_pair(
    current_y: np.ndarray,
    plus_y: np.ndarray,
    plus_warm: Any,
    minus_y: np.ndarray,
    minus_warm: Any,
) -> Any:
    return (
        plus_warm
        if np.linalg.norm(plus_y - current_y) <= np.linalg.norm(minus_y - current_y)
        else minus_warm
    )


class ZerothOrderBase(BaseMethod):


    def radius(self, iteration: int) -> float:
        radius = float(self.params.get("radius", self.params.get("mu", 0.10)))
        schedule = str(self.params.get("radius_schedule", "constant"))
        if schedule == "constant":
            value = radius
        elif schedule == "inverse_sqrt":
            value = radius / np.sqrt(iteration + 1.0)
        elif schedule == "exponential":
            value = radius * float(self.params.get("radius_decay", 1.00)) ** iteration
        else:
            raise ValueError(f"unknown radius schedule: {schedule}")
        return max(value, float(self.params.get("radius_min", 0.00)))

    def gaussian_direction(
        self,
        base_seed: int,
        family: str,
        instance_id: int,
        iteration: int,
        tag: str,
        dimension: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(
            stable_seed(base_seed, family, instance_id, iteration, tag)
        )
        return rng.normal(size=dimension)

    def apply_update(
        self,
        problem: Any,
        x: np.ndarray,
        gradient: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        maximum = self.params.get("update_clip")
        gradient = _clip_norm(
            gradient,
            None if maximum is None else float(maximum),
        )
        sign = 1.0 if problem.sense == "max" else -1.0
        return problem.project(x + sign * self.step_size(problem, iteration) * gradient)


class ZOTG(ZerothOrderBase):








    name = "ZO_TG"

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
            iteration_start = 0
            calls = 0
        else:
            x = np.asarray(restored["x"], dtype=float)
            current_y = np.asarray(restored["current_y"], dtype=float)
            current_warm = restored["current_warm"]
            objectives = list(restored["objectives"])
            oracle_calls = list(restored["oracle_calls"])
            iteration_start = int(restored["iteration"])
            calls = int(restored["calls"])

        for iteration in range(iteration_start, iterations):
            direction = self.gaussian_direction(
                base_seed,
                problem.family,
                instance_id,
                iteration,
                "zo-tg-direction",
                dimension,
            )
            
            
            mu = self.radius(iteration) / np.sqrt(float(dimension))
            plus_x = x + mu * direction
            minus_x = x - mu * direction
            plus_y, plus_warm = problem.response(plus_x, current_warm)
            minus_y, minus_warm = problem.response(minus_x, current_warm)
            calls += 2
            derivative = (
                problem.objective(plus_x, plus_y)
                - problem.objective(minus_x, minus_y)
            ) / (2.0 * mu)
            gradient = float(derivative) * direction
            x = self.apply_update(problem, x, gradient, iteration)

            warm_seed = _warm_from_pair(
                current_y, plus_y, plus_warm, minus_y, minus_warm
            )
            current_y, current_warm = problem.response(x, warm_seed)
            objectives.append(problem.objective(x, current_y))
            oracle_calls.append(calls)

            state = {
                "iteration": iteration + 1,
                "x": x,
                "current_y": current_y,
                "current_warm": current_warm,
                "objectives": objectives,
                "oracle_calls": oracle_calls,
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


class ZOOG(ZerothOrderBase):


    name = "ZO_OG"

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
            iteration_start = 0
            calls = 0
        else:
            x = np.asarray(restored["x"], dtype=float)
            current_y = np.asarray(restored["current_y"], dtype=float)
            current_warm = restored["current_warm"]
            objectives = list(restored["objectives"])
            oracle_calls = list(restored["oracle_calls"])
            iteration_start = int(restored["iteration"])
            calls = int(restored["calls"])

        for iteration in range(iteration_start, iterations):
            direction = self.gaussian_direction(
                base_seed,
                problem.family,
                instance_id,
                iteration,
                "zo-one-point-direction",
                dimension,
            )
            mu = self.radius(iteration) / np.sqrt(float(dimension))
            query_x = x + mu * direction
            query_y, query_warm = problem.response(query_x, current_warm)
            calls += 1
            query_value = problem.objective(query_x, query_y)
            gradient = float(query_value / mu) * direction
            x = self.apply_update(problem, x, gradient, iteration)

            current_y, current_warm = problem.response(x, query_warm)
            objectives.append(problem.objective(x, current_y))
            oracle_calls.append(calls)

            state = {
                "iteration": iteration + 1,
                "x": x,
                "current_y": current_y,
                "current_warm": current_warm,
                "objectives": objectives,
                "oracle_calls": oracle_calls,
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


class ZOOGVR(ZerothOrderBase):









    name = "ZO_OGVR"

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
            baseline = float(problem.objective(x, current_y))
            query_history = [np.asarray(x, dtype=float).copy()]
            response_history = [np.asarray(current_y, dtype=float).copy()]
            objectives = [baseline]
            
            oracle_calls = [1]
            iteration_start = 0
            calls = 1
        else:
            x = np.asarray(restored["x"], dtype=float)
            current_y = np.asarray(restored["current_y"], dtype=float)
            current_warm = restored["current_warm"]
            baseline = float(restored["baseline"])
            query_history = [
                np.asarray(value, dtype=float) for value in restored["query_history"]
            ]
            response_history = [
                np.asarray(value, dtype=float) for value in restored["response_history"]
            ]
            objectives = list(restored["objectives"])
            oracle_calls = list(restored["oracle_calls"])
            iteration_start = int(restored["iteration"])
            calls = int(restored["calls"])

        history_window = max(1, int(self.params.get("history_window", 10)))
        distance_weight = float(self.params.get("distance_weight", 0.10))

        for iteration in range(iteration_start, iterations):
            direction = self.gaussian_direction(
                base_seed,
                problem.family,
                instance_id,
                iteration,
                "zo-one-point-direction",
                dimension,
            )
            mu = self.radius(iteration) / np.sqrt(float(dimension))
            query_x = x + mu * direction
            query_y, query_warm = problem.response(query_x, current_warm)
            calls += 1
            query_value = float(problem.objective(query_x, query_y))
            gradient = ((query_value - baseline) / mu) * direction
            x = self.apply_update(problem, x, gradient, iteration)

            query_history.append(np.asarray(query_x, dtype=float).copy())
            response_history.append(np.asarray(query_y, dtype=float).copy())
            if len(query_history) > history_window:
                query_history = query_history[-history_window:]
                response_history = response_history[-history_window:]

            distances = np.asarray(
                [
                    distance_weight * float(np.linalg.norm(x - query) ** 2) + 1.0
                    for query in query_history
                ],
                dtype=float,
            )
            inverse = 1.0 / np.maximum(distances, 1.0e-15)
            weights = inverse / np.sum(inverse)
            baseline = float(
                sum(
                    weight * problem.objective(x, response)
                    for weight, response in zip(weights, response_history, strict=True)
                )
            )

            current_y, current_warm = problem.response(x, query_warm)
            objectives.append(problem.objective(x, current_y))
            oracle_calls.append(calls)

            state = {
                "iteration": iteration + 1,
                "x": x,
                "current_y": current_y,
                "current_warm": current_warm,
                "baseline": baseline,
                "query_history": query_history,
                "response_history": response_history,
                "objectives": objectives,
                "oracle_calls": oracle_calls,
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


class GuidedZerothOrderBase(ZerothOrderBase):


    def guided_direction(
        self,
        base_seed: int,
        family: str,
        instance_id: int,
        iteration: int,
        dimension: int,
        alpha: float,
        guide: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.default_rng(
            stable_seed(base_seed, family, instance_id, iteration, "gzo-direction")
        )
        normalized_guide = _normalized(guide)
        if float(np.linalg.norm(normalized_guide)) <= 1.0e-15:
            
            
            
            return rng.normal(size=dimension) / np.sqrt(float(dimension))
        return (
            np.sqrt(alpha / float(dimension)) * rng.normal(size=dimension)
            + np.sqrt(max(0.0, 1.0 - alpha))
            * float(rng.normal())
            * normalized_guide
        )

    def next_alpha(self, alpha: float) -> float:
        damping = float(self.params.get("alpha_damping", 0.98))
        return float(np.clip(1.0 - damping * (1.0 - alpha), 0.0, 1.0))


class GZONS(GuidedZerothOrderBase):


    name = "GZO_NS"

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
            alpha = float(self.params.get("alpha_initial", 0.00))
            iteration_start = 0
            calls = 0
        else:
            x = np.asarray(restored["x"], dtype=float)
            current_y = np.asarray(restored["current_y"], dtype=float)
            current_warm = restored["current_warm"]
            objectives = list(restored["objectives"])
            oracle_calls = list(restored["oracle_calls"])
            alpha = float(restored["alpha"])
            iteration_start = int(restored["iteration"])
            calls = int(restored["calls"])

        for iteration in range(iteration_start, iterations):
            
            calls += 1
            guide = np.asarray(problem.partial_x(x, current_y), dtype=float)
            direction = self.guided_direction(
                base_seed,
                problem.family,
                instance_id,
                iteration,
                dimension,
                alpha,
                guide,
            )
            mu = self.radius(iteration)
            plus_x = x + mu * direction
            minus_x = x - mu * direction
            plus_y, plus_warm = problem.response(plus_x, current_warm)
            minus_y, minus_warm = problem.response(minus_x, current_warm)
            calls += 2
            derivative = (
                problem.objective(plus_x, plus_y)
                - problem.objective(minus_x, minus_y)
            ) / (2.0 * mu)
            
            
            gradient = float(derivative) * direction
            x = self.apply_update(problem, x, gradient, iteration)
            alpha = self.next_alpha(alpha)

            warm_seed = _warm_from_pair(
                current_y, plus_y, plus_warm, minus_y, minus_warm
            )
            current_y, current_warm = problem.response(x, warm_seed)
            objectives.append(problem.objective(x, current_y))
            oracle_calls.append(calls)

            state = {
                "iteration": iteration + 1,
                "x": x,
                "current_y": current_y,
                "current_warm": current_warm,
                "objectives": objectives,
                "oracle_calls": oracle_calls,
                "alpha": alpha,
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


class GZOHS(GuidedZerothOrderBase):


    name = "GZO_HS"

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
            alpha = float(self.params.get("alpha_initial", 0.00))
            guide = np.zeros(dimension, dtype=float)
            response_history: list[np.ndarray] = []
            iteration_start = 0
            calls = 0
        else:
            x = np.asarray(restored["x"], dtype=float)
            current_y = np.asarray(restored["current_y"], dtype=float)
            current_warm = restored["current_warm"]
            objectives = list(restored["objectives"])
            oracle_calls = list(restored["oracle_calls"])
            alpha = float(restored["alpha"])
            guide = np.asarray(restored["guide"], dtype=float)
            response_history = [
                np.asarray(value, dtype=float) for value in restored["response_history"]
            ]
            iteration_start = int(restored["iteration"])
            calls = int(restored["calls"])

        history_window = max(1, int(self.params.get("history_window", 5)))

        for iteration in range(iteration_start, iterations):
            direction = self.guided_direction(
                base_seed,
                problem.family,
                instance_id,
                iteration,
                dimension,
                alpha,
                guide,
            )
            mu = self.radius(iteration)
            plus_x = x + mu * direction
            minus_x = x - mu * direction
            plus_y, plus_warm = problem.response(plus_x, current_warm)
            minus_y, minus_warm = problem.response(minus_x, current_warm)
            calls += 2
            derivative = (
                problem.objective(plus_x, plus_y)
                - problem.objective(minus_x, minus_y)
            ) / (2.0 * mu)
            gradient = float(derivative) * direction
            x = self.apply_update(problem, x, gradient, iteration)

            response_history.extend(
                [
                    np.asarray(plus_y, dtype=float).copy(),
                    np.asarray(minus_y, dtype=float).copy(),
                ]
            )
            maximum_responses = 2 * history_window
            if len(response_history) > maximum_responses:
                response_history = response_history[-maximum_responses:]
            historical_gradients = [
                np.asarray(problem.partial_x(x, response), dtype=float)
                for response in response_history
            ]
            guide = _normalized(np.mean(historical_gradients, axis=0))
            alpha = self.next_alpha(alpha)

            warm_seed = _warm_from_pair(
                current_y, plus_y, plus_warm, minus_y, minus_warm
            )
            current_y, current_warm = problem.response(x, warm_seed)
            objectives.append(problem.objective(x, current_y))
            oracle_calls.append(calls)

            state = {
                "iteration": iteration + 1,
                "x": x,
                "current_y": current_y,
                "current_warm": current_warm,
                "objectives": objectives,
                "oracle_calls": oracle_calls,
                "alpha": alpha,
                "guide": guide,
                "response_history": response_history,
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
