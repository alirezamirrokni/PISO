from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.methods.common import BaseMethod, MethodTrace


class PZOS(BaseMethod):
    name = "PZOS"

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
        mu = float(self.params["mu"])
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
            
            
            calls += 1
            direction = self.direction(base_seed, problem.family, instance_id, iteration, dimension)
            plus_y, plus_warm = problem.response(x + mu * direction, current_warm)
            minus_y, minus_warm = problem.response(x - mu * direction, current_warm)
            calls += 2

            partial_x = problem.partial_x(x, current_y)
            partial_y = problem.partial_y(x, current_y)
            response_difference = plus_y - minus_y
            chain_scalar = float(np.dot(response_difference, partial_y))
            implicit_component = dimension * chain_scalar * direction / (2.0 * mu)
            gradient = partial_x + implicit_component

            step = self.step_size(problem, iteration)
            sign = 1.0 if problem.sense == "max" else -1.0
            x = problem.project(x + sign * step * gradient)

            warm_seed = plus_warm if np.linalg.norm(plus_y - current_y) <= np.linalg.norm(minus_y - current_y) else minus_warm
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
