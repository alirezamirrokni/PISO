from __future__ import annotations

from typing import Any

import numpy as np

from src.pricing_methods.common import (
    Trace,
    batch_size,
    finish,
    initial_state,
    record,
    restore_or_initialize,
    save_step,
)


def guide_batch_size(params: dict[str, Any], iteration: int) -> int:
    return int(
        params["guide_batch_initial"]
        + iteration * params["guide_batch_increment"]
    )


def exponential(initial: float, decay: float, iteration: int) -> float:
    return float(initial) * float(decay) ** int(iteration)


def normalized(vector: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= tolerance:
        return np.zeros_like(value)
    return value / norm


def combine_observations(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "features": np.concatenate(
            (first["features"], second["features"]), axis=0
        ),
        "labels": np.concatenate((first["labels"], second["labels"]), axis=0),
    }


def known_gradient(
    problem,
    x: np.ndarray,
    observations: dict[str, np.ndarray],
) -> np.ndarray:
    return problem.partial_gradients(x, observations).mean(axis=0)


def validate_piso_params(params: dict[str, Any], method_name: str) -> None:
    for key in ("radius", "step_size", "gamma", "batch_initial"):
        value = float(params[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{method_name} {key} must be finite and positive")
    for key in ("radius_decay", "step_decay"):
        value = float(params[key])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{method_name} {key} must lie in (0, 1]")
    rho = float(params["rho"])
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"{method_name} rho must lie in [0, 1)")
    fraction = float(params["known_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"{method_name} known_fraction must lie in (0, 1)")
    if int(params["batch_increment"]) < 0:
        raise ValueError(f"{method_name} batch_increment must be nonnegative")


def allocate_batches(params: dict[str, Any], iteration_batch: int) -> tuple[int, int, int]:
    total = 3 * int(iteration_batch)
    known = int(round(total * float(params["known_fraction"])))
    remaining = total - known
    plus = remaining // 2
    minus = remaining - plus
    if min(known, plus, minus) <= 0:
        raise ValueError("PISO batch split must allocate at least one sample per term")
    return known, plus, minus


def haar_frame(
    rng: np.random.RandomState,
    dimension: int,
    columns: int,
) -> np.ndarray:
    gaussian = rng.normal(size=(dimension, columns))
    q_matrix, r_matrix = np.linalg.qr(gaussian, mode="reduced")
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return q_matrix * signs


__all__ = [
    "Trace",
    "allocate_batches",
    "batch_size",
    "combine_observations",
    "exponential",
    "finish",
    "guide_batch_size",
    "haar_frame",
    "initial_state",
    "known_gradient",
    "normalized",
    "record",
    "restore_or_initialize",
    "save_step",
    "validate_piso_params",
]
