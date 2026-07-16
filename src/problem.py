from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    products: int
    buyers: int
    initial_price: float
    outside_weight_per_product: float
    lower_inventory_factor: float
    upper_inventory_factor: float
    cost_ratio_low: float
    cost_ratio_middle: float
    cost_ratio_high: float
    rho_low: float
    rho_high: float

    @classmethod
    def from_mapping(cls, values: dict) -> "ProblemSpec":
        return cls(**values)


class PricingProblem:
    def __init__(self, theta: np.ndarray, rho: np.ndarray, spec: ProblemSpec) -> None:
        self.spec = spec
        self.n = spec.products
        self.m = spec.buyers
        self.theta = np.asarray(theta, dtype=float).reshape(self.n)
        self.rho = np.asarray(rho, dtype=float).reshape(self.n)
        self.gamma = 2.0 * np.pi / (np.sqrt(6.0) * self.theta)
        self.w = self.rho * self.theta
        self.slope_low = spec.cost_ratio_low * self.w
        self.slope_middle = spec.cost_ratio_middle * self.w
        self.slope_high = spec.cost_ratio_high * self.w
        self.lower = spec.lower_inventory_factor * self.m / self.n
        self.upper = spec.upper_inventory_factor * self.m / self.n
        self.outside_weight = spec.outside_weight_per_product * self.n

    @staticmethod
    def _week_theta(data_dir: Path, week_id: str, spec: ProblemSpec) -> np.ndarray:
        raw = np.loadtxt(
            data_dir / "prices" / f"2022_{week_id}.csv",
            encoding="utf-8-sig",
        )
        theta = raw[: spec.products].astype(float)
        theta /= theta.max()
        return theta

    @classmethod
    def from_week(
        cls,
        data_dir: Path,
        week_id: str,
        rng: np.random.RandomState,
        spec: ProblemSpec,
    ):
        theta = cls._week_theta(data_dir, week_id, spec)
        rho = spec.rho_low + (spec.rho_high - spec.rho_low) * rng.rand(spec.products)
        return cls(theta, rho, spec)

    @classmethod
    def from_week_with_rho(
        cls,
        data_dir: Path,
        week_id: str,
        rho: np.ndarray,
        spec: ProblemSpec,
    ):
        return cls(cls._week_theta(data_dir, week_id, spec), rho, spec)

    @property
    def initial_x(self) -> np.ndarray:
        return np.full(self.n, self.spec.initial_price, dtype=float)

    def sample_demands(self, x: np.ndarray, count: int, rng: np.random.RandomState) -> np.ndarray:
        utility = self.gamma * (self.theta - np.asarray(x).reshape(self.n))
        exp_value = np.exp(utility)
        cumulative = np.cumsum(exp_value / (self.outside_weight + exp_value.sum()))
        choices = np.searchsorted(cumulative, rng.rand(count, self.m))
        categories = np.arange(self.n + 1)
        return (choices[:, :, None] == categories).sum(axis=1).astype(np.int64)

    def loss_from_demands(self, x: np.ndarray, demands: np.ndarray) -> np.ndarray:
        sold = np.asarray(demands)[:, : self.n].astype(float, copy=False)
        low = self.slope_low * sold
        middle = self.slope_middle * (sold - self.lower) + self.slope_low * self.lower
        high = (
            self.slope_high * (sold - self.upper)
            + self.slope_middle * (self.upper - self.lower)
            + self.slope_low * self.lower
        )
        costs = np.where(sold < self.lower, low, np.where(sold < self.upper, middle, high))
        return -(sold @ np.asarray(x).reshape(self.n)) + costs.sum(axis=1)

    def sample_losses(self, x: np.ndarray, count: int, rng: np.random.RandomState):
        demands = self.sample_demands(x, count, rng)
        return self.loss_from_demands(x, demands), demands

    def evaluate(self, x: np.ndarray, count: int, rng: np.random.RandomState) -> float:
        losses, _ = self.sample_losses(x, count, rng)
        return float(losses.mean())

    @staticmethod
    def partial_gradients(demands: np.ndarray) -> np.ndarray:
        return -np.asarray(demands)[:, :-1].astype(float, copy=False)
