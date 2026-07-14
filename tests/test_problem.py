from pathlib import Path

import numpy as np

from ddp_pricing.problem import PricingProblem, ProblemSpec


def spec() -> ProblemSpec:
    return ProblemSpec(
        products=10,
        buyers=40,
        initial_price=0.5,
        outside_weight_per_product=0.1,
        lower_inventory_factor=0.5,
        upper_inventory_factor=1.5,
        cost_ratio_low=2.0,
        cost_ratio_middle=1.0,
        cost_ratio_high=3.0,
        rho_low=0.25,
        rho_high=0.5,
    )


def test_demands_sum_to_number_of_buyers() -> None:
    root = Path(__file__).resolve().parents[1]
    rng = np.random.RandomState(2024)
    problem = PricingProblem.from_week(root / "data", "08", rng, spec())
    demands = problem.sample_demands(problem.initial_x, 50, rng)
    assert demands.shape == (50, 11)
    assert np.all(demands.sum(axis=1) == 40)


def test_partial_gradient_is_negative_demand() -> None:
    demands = np.array([[1, 2, 3]])
    gradient = PricingProblem.partial_gradients(demands)
    np.testing.assert_array_equal(gradient, np.array([[-1.0, -2.0]]))
