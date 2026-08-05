from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _bisect_positive_root(function, size: int, initial_high: np.ndarray | float) -> np.ndarray:
    low = np.zeros(size, dtype=float)
    high = np.broadcast_to(np.asarray(initial_high, dtype=float), (size,)).copy()
    values = function(high)
    for _ in range(80):
        mask = values > 0.0
        if not np.any(mask):
            break
        high[mask] *= 2.0
        values = function(high)
    for _ in range(80):
        middle = 0.5 * (low + high)
        values = function(middle)
        positive = values > 0.0
        low[positive] = middle[positive]
        high[~positive] = middle[~positive]
    return 0.5 * (low + high)


@dataclass
class SecurityInstance:
    defender_value: np.ndarray
    attacker_value: np.ndarray
    baseline_security: np.ndarray
    defender_cost: np.ndarray
    attacker_cost: np.ndarray
    attacker_budget: float
    initial_x: np.ndarray
    response_tolerance: float
    response_max_iterations: int

    family: str = "security"
    sense: str = "min"

    def __post_init__(self) -> None:
        self.defender_value = np.asarray(self.defender_value, dtype=float)
        self.attacker_value = np.asarray(self.attacker_value, dtype=float)
        self.baseline_security = np.asarray(self.baseline_security, dtype=float)
        self.defender_cost = np.asarray(self.defender_cost, dtype=float)
        self.attacker_cost = np.asarray(self.attacker_cost, dtype=float)
        self.initial_x = np.asarray(self.initial_x, dtype=float)
        self.dimension = int(self.defender_value.size)

    def initial_decision(self) -> np.ndarray:
        return self.initial_x.copy()

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(x, dtype=float), 0.0)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        denominator = np.maximum(x + y + self.baseline_security, 1e-12)
        damage = self.defender_value * y / denominator
        investment = 0.5 * self.defender_cost * x**2
        return float(np.sum(damage + investment))

    def partial_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        denominator = np.maximum(x + y + self.baseline_security, 1e-12)
        return -self.defender_value * y / denominator**2 + self.defender_cost * x

    def partial_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        denominator = np.maximum(x + y + self.baseline_security, 1e-12)
        return self.defender_value * (x + self.baseline_security) / denominator**2

    def _effort_at_dual(
        self,
        c0: np.ndarray,
        dual: float,
        initial: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        h0 = self.attacker_value / np.maximum(c0, 1e-12)
        active = h0 > dual
        if initial is None:
            denominator = self.attacker_cost + 2.0 * self.attacker_value / np.maximum(c0, 1e-12) ** 2
            effort = np.maximum((h0 - dual) / denominator, 0.0)
        else:
            effort = np.maximum(np.asarray(initial, dtype=float), 0.0).copy()
        effort = np.minimum(effort, self.attacker_budget)
        effort[~active] = 0.0

        for _ in range(10):
            denominator = np.maximum(c0 + effort, 1e-12)
            equation = self.attacker_value * c0 / denominator**2 - self.attacker_cost * effort - dual
            derivative = -2.0 * self.attacker_value * c0 / denominator**3 - self.attacker_cost
            update = np.divide(
                equation,
                derivative,
                out=np.zeros_like(equation),
                where=np.abs(derivative) > 1e-14,
            )
            effort = np.clip(effort - update, 0.0, self.attacker_budget)
            effort[~active] = 0.0

        denominator = np.maximum(c0 + effort, 1e-12)
        derivative = -2.0 * self.attacker_value * c0 / denominator**3 - self.attacker_cost
        return effort, derivative

    def response(self, x: np.ndarray, warm: Any | None = None) -> tuple[np.ndarray, Any]:
        x = np.asarray(x, dtype=float)
        c0 = np.maximum(x + self.baseline_security, 1e-10)
        h0 = self.attacker_value / c0
        low = 0.0
        high = float(np.max(h0))
        dual = float(warm.get("dual", 0.5 * high)) if isinstance(warm, dict) else 0.5 * high
        dual = float(np.clip(dual, low, high))
        effort_initial = warm.get("effort") if isinstance(warm, dict) else None
        effort: np.ndarray | None = None

        for _ in range(int(self.response_max_iterations)):
            effort, derivative = self._effort_at_dual(c0, dual, effort_initial)
            total = float(np.sum(effort))
            residual = total - self.attacker_budget
            if abs(residual) <= self.response_tolerance * max(1.0, self.attacker_budget):
                break
            if residual > 0.0:
                low = dual
            else:
                high = dual

            active = effort > 1e-12
            total_derivative = float(np.sum(1.0 / derivative[active])) if np.any(active) else -np.inf
            if np.isfinite(total_derivative) and total_derivative < -1e-14:
                proposal = dual - residual / total_derivative
            else:
                proposal = np.nan
            if not np.isfinite(proposal) or proposal <= low or proposal >= high:
                proposal = 0.5 * (low + high)
            dual = float(proposal)
            effort_initial = effort

        if effort is None:
            effort, _ = self._effort_at_dual(c0, dual, effort_initial)
        return effort, {"dual": dual, "effort": effort}


def _symmetric_effort(attacker_value: np.ndarray, baseline: np.ndarray, attacker_cost: np.ndarray) -> np.ndarray:
    def equation(s: np.ndarray) -> np.ndarray:
        return attacker_value * (s + baseline) / (2.0 * s + baseline) ** 2 - attacker_cost * s

    return _bisect_positive_root(equation, attacker_value.size, np.ones(attacker_value.size))


def _initial_defense(
    defender_value: np.ndarray,
    baseline: np.ndarray,
    defender_cost: np.ndarray,
) -> np.ndarray:
    def equation(x: np.ndarray) -> np.ndarray:
        return defender_value * baseline / (x + 2.0 * baseline) ** 2 - defender_cost * x

    return _bisect_positive_root(equation, defender_value.size, np.ones(defender_value.size))


def generate_security_instances(config: dict, seed: int) -> list[SecurityInstance]:
    exp = config["experiment"]
    problem = config["problem"]
    count = int(exp["instances"])
    rng = np.random.default_rng(seed)

    dimensions = rng.integers(
        int(problem["dimension_min"]),
        int(problem["dimension_max"]) + 1,
        size=count,
    )
    instances: list[SecurityInstance] = []
    for dimension in dimensions:
        n = int(dimension)
        defender_value = rng.uniform(float(problem["value_min"]), float(problem["value_max"]), n)
        attacker_value = rng.uniform(float(problem["value_min"]), float(problem["value_max"]), n)
        baseline = rng.uniform(float(problem["baseline_min"]), float(problem["baseline_max"]), n)
        defender_cost = rng.uniform(float(problem["cost_min"]), float(problem["cost_max"]), n)
        attacker_cost = rng.uniform(float(problem["cost_min"]), float(problem["cost_max"]), n)

        symmetric = _symmetric_effort(attacker_value, baseline, attacker_cost)
        attacker_budget = float(problem["budget_scale"] * np.sum(symmetric))
        initial_x = _initial_defense(defender_value, baseline, defender_cost)

        instances.append(
            SecurityInstance(
                defender_value=defender_value,
                attacker_value=attacker_value,
                baseline_security=baseline,
                defender_cost=defender_cost,
                attacker_cost=attacker_cost,
                attacker_budget=attacker_budget,
                initial_x=initial_x,
                response_tolerance=float(problem["response_tolerance"]),
                response_max_iterations=int(problem["response_max_iterations"]),
            )
        )
    return instances
