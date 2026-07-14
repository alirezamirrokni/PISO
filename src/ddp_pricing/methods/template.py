from __future__ import annotations

import numpy as np

from ddp_pricing.methods.common import make_trace


class MethodTemplate:
    name = "METHOD_TEMPLATE"

    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self, problem, rng: np.random.RandomState, run_context):
        x = problem.initial_x
        counts = [0]
        values = [problem.evaluate(x, run_context.metric_samples, rng)]
        return make_trace(self.name, counts, values, x)
