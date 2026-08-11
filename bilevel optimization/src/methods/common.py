from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.util import stable_seed, unit_sphere_direction


@dataclass
class MethodTrace:
    method: str
    instance_id: int
    dimension: int
    objectives: np.ndarray
    oracle_calls: np.ndarray


class BaseMethod:
    name = "Base"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def step_size(self, problem: Any, iteration: int) -> float:
        schedule = str(self.params.get("step_schedule", "constant"))
        if problem.family == "routing" and self.params.get("scale_by_dimension", False):
            base = float(self.params["step_size"]) / float(problem.dimension)
        elif problem.family == "security":
            threshold = int(self.params.get("dimension_threshold", 100))
            base = float(
                self.params["step_size_low_dimension"]
                if problem.dimension <= threshold
                else self.params["step_size_high_dimension"]
            )
        else:
            base = float(self.params["step_size"])
        if schedule == "inverse_sqrt":
            return base / np.sqrt(iteration + 1.0)
        if schedule != "constant":
            raise ValueError(f"unknown step schedule: {schedule}")
        return base

    def direction(self, base_seed: int, family: str, instance_id: int, iteration: int, dimension: int) -> np.ndarray:
        
        seed = stable_seed(base_seed, family, instance_id, iteration, "direction")
        return unit_sphere_direction(dimension, seed)

    def run(self, *args, **kwargs) -> MethodTrace:
        raise NotImplementedError
