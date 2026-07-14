from pathlib import Path

import numpy as np

from ddp_pricing.config import load_yaml
from ddp_pricing.loading import load_factory
from ddp_pricing.problem import PricingProblem, ProblemSpec
from ddp_pricing.types import RunContext


def test_reference_sample_totals() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs" / "reference.yaml")
    rng = np.random.RandomState(2024)
    problem = PricingProblem.from_week(root / "data", "08", rng, ProblemSpec.from_mapping(config["problem"]))
    context = RunContext("08", "02/21-02/27", 0, 5000, 5)
    expected = {
        "GZO_NS": 5046,
        "GZO_HS": 5092,
        "ZO_TG": 5092,
        "ZO_OG": 5046,
        "ZO_OGVR": 5066,
    }
    for method_name in config["experiment"]["method_order"]:
        entry = config["methods"][method_name]
        method = load_factory(entry["factory"])(entry["params"])
        trace = method.run(problem, rng, context)
        assert trace.sample_counts[-1] == expected[method_name]
