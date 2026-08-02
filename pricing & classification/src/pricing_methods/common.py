from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Trace:
    samples: list[int]
    objectives: list[float]
    final_x: np.ndarray


def batch_size(params: dict[str, Any], iteration: int) -> int:
    return int(params["batch_initial"] + iteration * params["batch_increment"])


def initial_state(problem, metric_samples: int, rng: np.random.RandomState, **values) -> dict[str, Any]:
    state = {
        "x": problem.initial_x,
        "samples": [0],
        "objectives": [problem.evaluate(problem.initial_x, metric_samples, rng)],
        "sample_count": 0,
        "iteration": 0,
    }
    state.update(values)
    return state


def restore_or_initialize(cache, rng: np.random.RandomState, initializer):
    saved = cache.load_progress()
    if saved is not None:
        rng.set_state(saved["rng_state"])
        return saved["state"], True
    state = initializer()
    cache.save_progress(state, rng.get_state())
    return state, False


def record(state: dict[str, Any], problem, metric_samples: int, rng: np.random.RandomState) -> None:
    state["samples"].append(int(state["sample_count"]))
    state["objectives"].append(problem.evaluate(state["x"], metric_samples, rng))


def save_step(cache, state: dict[str, Any], rng: np.random.RandomState, progress) -> None:
    cache.save_progress(state, rng.get_state())
    if progress is not None:
        progress.update(int(state["sample_count"]), int(state["iteration"]))


def finish(state: dict[str, Any]) -> Trace:
    return Trace(
        samples=list(state["samples"]),
        objectives=list(state["objectives"]),
        final_x=np.asarray(state["x"], dtype=float).copy(),
    )
