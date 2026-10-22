from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "prices"
WEEKS = {"08": "02/21-02/27", "12": "03/21-03/27", "21": "05/23-05/29",
         "25": "06/20-06/26", "29": "07/18-07/24", "32": "08/08-08/14"}


class PricingProblem:
    def __init__(self, week, rho, spec):
        self.spec = spec
        self.n = int(spec["products"])
        self.m = int(spec["buyers"])
        theta = np.loadtxt(DATA_DIR / f"2022_{week}.csv", encoding="utf-8-sig")[: self.n].astype(float)
        self.theta = theta / theta.max()
        self.rho = np.asarray(rho, dtype=float)
        self.gamma = 2.0 * np.pi / (np.sqrt(6.0) * self.theta)
        w = self.rho * self.theta
        self.slope_low = spec["cost_ratio_low"] * w
        self.slope_middle = spec["cost_ratio_middle"] * w
        self.slope_high = spec["cost_ratio_high"] * w
        self.lower = spec["lower_inventory_factor"] * self.m / self.n
        self.upper = spec["upper_inventory_factor"] * self.m / self.n
        self.outside_weight = spec["outside_weight_per_product"] * self.n

    @property
    def initial_x(self):
        return np.full(self.n, self.spec["initial_price"], dtype=float)

    def sample_demands(self, x, count, rng):
        utility = self.gamma * (self.theta - np.asarray(x).reshape(self.n))
        weights = np.exp(np.clip(utility, -700.0, 700.0))
        cumulative = np.cumsum(weights / (self.outside_weight + weights.sum()))
        choices = np.searchsorted(cumulative, rng.rand(int(count), self.m))
        return (choices[:, :, None] == np.arange(self.n + 1)).sum(axis=1).astype(np.int64)

    def loss_from_demands(self, x, demands):
        sold = np.asarray(demands)[:, : self.n].astype(float, copy=False)
        low = self.slope_low * sold
        middle = self.slope_middle * (sold - self.lower) + self.slope_low * self.lower
        high = (self.slope_high * (sold - self.upper) + self.slope_middle * (self.upper - self.lower)
                + self.slope_low * self.lower)
        costs = np.where(sold < self.lower, low, np.where(sold < self.upper, middle, high))
        return -(sold @ np.asarray(x).reshape(self.n)) + costs.sum(axis=1)

    def sample_losses(self, x, count, rng):
        demands = self.sample_demands(x, count, rng)
        return self.loss_from_demands(x, demands), demands

    def evaluate(self, x, count, rng):
        return float(self.sample_losses(x, count, rng)[0].mean())

    @staticmethod
    def partial_gradients(demands):
        return -np.asarray(demands)[:, :-1].astype(float, copy=False)
