from __future__ import annotations

import math
from typing import Any

import numpy as np

from ddp_pricing.types import MethodTrace, RunContext


def batch_size(params: dict[str, Any], iteration: int) -> int:
    return int(params["batch_initial"] + iteration * params["batch_increment"])


def update_before(value: float, decay: float, enabled: bool) -> float:
    return value * decay if enabled else value


def update_after(value: float, decay: float, enabled: bool) -> float:
    return value if enabled else value * decay


def append_metric(
    trace_counts: list[int],
    trace_values: list[float],
    sample_count: int,
    x: np.ndarray,
    problem,
    rng: np.random.RandomState,
    context: RunContext,
) -> None:
    trace_counts.append(int(sample_count))
    trace_values.append(problem.evaluate(x, context.metric_samples, rng))


def make_trace(name: str, counts: list[int], values: list[float], x: np.ndarray, **metadata) -> MethodTrace:
    trace = MethodTrace(
        method=name,
        sample_counts=counts,
        objectives=values,
        final_x=np.asarray(x, dtype=float).copy(),
        metadata=metadata,
    )
    trace.validate()
    return trace


def resolve_beta0(params: dict[str, Any], d: int) -> float:
    if "beta0" in params:
        return float(params["beta0"])
    expression = params.get("beta0_expression")
    if expression == "0.01 / sqrt(d)":
        return 0.01 / math.sqrt(d)
    raise ValueError(f"unsupported beta0 expression: {expression!r}")
