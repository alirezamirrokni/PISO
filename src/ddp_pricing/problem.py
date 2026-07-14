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
    def from_mapping(cls, data: dict) -> "ProblemSpec":
        return cls(**data)


class PricingProblem:
    def __init__(
        self,
        theta: np.ndarray,
        rho: np.ndarray,
        spec: ProblemSpec,
    ) -> None:
        self.spec = spec
        self.n = spec.products
        self.m = spec.buyers
        theta = np.asarray(theta, dtype=float).reshape(-1)
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if theta.shape != (self.n,) or rho.shape != (self.n,):
            raise ValueError("theta and rho must have one entry per product")
        if np.any(theta <= 0):
            raise ValueError("all normalized prices must be positive")
        self.theta = theta
        self.rho = rho
        self.gamma = 2.0 * np.pi / (np.sqrt(6.0) * theta)
        self.w = rho * theta
        self.slope_low = spec.cost_ratio_low * self.w
        self.slope_middle = spec.cost_ratio_middle * self.w
        self.slope_high = spec.cost_ratio_high * self.w
        self.lower = spec.lower_inventory_factor * self.m / self.n
        self.upper = spec.upper_inventory_factor * self.m / self.n
        self.outside_weight = spec.outside_weight_per_product * self.n

    @classmethod
    def from_week(
        cls,
        data_dir: Path,
        week_id: str,
        rng: np.random.RandomState,
        spec: ProblemSpec,
    ) -> "PricingProblem":
        path = data_dir / "prices" / f"2022_{week_id}.csv"
        raw = np.loadtxt(path, dtype=float, encoding="utf-8-sig")
        if raw.size < spec.products:
            raise ValueError(f"{path} contains fewer than {spec.products} values")
        theta = raw[: spec.products]
        theta = theta / np.max(theta)
        rho = spec.rho_low + (spec.rho_high - spec.rho_low) * rng.rand(spec.products)
        return cls(theta=theta, rho=rho, spec=spec)

    @property
    def initial_x(self) -> np.ndarray:
        return np.full(self.n, self.spec.initial_price, dtype=float)

    def purchase_probabilities(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.n)
        utility = self.gamma * (self.theta - x)
        exp_value = np.exp(utility)
        denominator = self.outside_weight + np.sum(exp_value)
        return exp_value / denominator

    def sample_demands(
        self,
        x: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty((0, self.n + 1), dtype=np.int64)
        probs = self.purchase_probabilities(x)
        cumulative = np.cumsum(probs)
        uniforms = rng.rand(count, self.m)
        choices = np.searchsorted(cumulative, uniforms)
        categories = np.arange(self.n + 1)
        demands = (choices[:, :, None] == categories[None, None, :]).sum(axis=1)
        return demands.astype(np.int64, copy=False)

    def loss_from_demands(self, x: np.ndarray, demands: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.n)
        demands = np.asarray(demands)
        sold = demands[:, : self.n].astype(float, copy=False)
        low_cost = self.slope_low * sold
        middle_cost = self.slope_middle * (sold - self.lower) + self.slope_low * self.lower
        high_cost = (
            self.slope_high * (sold - self.upper)
            + self.slope_middle * (self.upper - self.lower)
            + self.slope_low * self.lower
        )
        product_cost = np.where(
            sold < self.lower,
            low_cost,
            np.where(sold < self.upper, middle_cost, high_cost),
        )
        revenue = sold @ x
        return -revenue + np.sum(product_cost, axis=1)

    def sample_losses(
        self,
        x: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        demands = self.sample_demands(x=x, count=count, rng=rng)
        losses = self.loss_from_demands(x=x, demands=demands)
        return losses, demands

    def evaluate(self, x: np.ndarray, count: int, rng: np.random.RandomState) -> float:
        losses, _ = self.sample_losses(x=x, count=count, rng=rng)
        return float(np.mean(losses))

    @staticmethod
    def partial_gradients(demands: np.ndarray) -> np.ndarray:
        return -np.asarray(demands)[:, :-1].astype(float, copy=False)
