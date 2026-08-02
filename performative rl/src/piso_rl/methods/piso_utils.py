from __future__ import annotations

from typing import Any

import numpy as np


def validate_common_params(params: dict[str, Any], method_name: str) -> None:
    for key in ("radius", "step_size", "gamma", "batch_initial"):
        value = float(params[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{method_name} {key} must be finite and positive")
    for key in ("radius_decay", "step_decay"):
        value = float(params.get(key, 1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{method_name} {key} must lie in (0, 1]")
    rho = float(params["rho"])
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"{method_name} rho must lie in [0, 1)")
    known_fraction = float(params["known_fraction"])
    if not 0.0 < known_fraction < 1.0:
        raise ValueError(f"{method_name} known_fraction must lie in (0, 1)")
    if int(params.get("batch_increment", 0)) < 0:
        raise ValueError(f"{method_name} batch_increment must be nonnegative")


def step_size(params: dict[str, Any], iteration: int) -> float:
    return float(params["step_size"]) * float(params.get("step_decay", 1.0)) ** int(
        iteration
    )


def radius(params: dict[str, Any], iteration: int) -> float:
    return float(params["radius"]) * float(params.get("radius_decay", 1.0)) ** int(
        iteration
    )


def allocate_batches(params: dict[str, Any], iteration_batch: int) -> tuple[int, int, int]:
    total = 3 * int(iteration_batch)
    known = int(round(total * float(params["known_fraction"])))
    remaining = total - known
    plus = remaining // 2
    minus = remaining - plus
    if min(known, plus, minus) <= 0:
        raise ValueError("PISO batch split must allocate at least one trajectory per query")
    return known, plus, minus


def known_gradient(problem, theta: np.ndarray, trajectories) -> np.ndarray:
    return problem.partial_gradients(theta, trajectories).mean(axis=0)


def guided_covariance_geometry(
    alpha: float, dimension: int, hint: np.ndarray
) -> tuple[float, float]:
    orthogonal = alpha / float(dimension)
    parallel = 1.0 - alpha + orthogonal if np.linalg.norm(hint) > 0.0 else orthogonal
    return parallel, orthogonal


def haar_frame(
    rng: np.random.RandomState, dimension: int, columns: int
) -> np.ndarray:
    if not 1 <= columns <= dimension:
        raise ValueError("frame columns must satisfy 1 <= columns <= dimension")
    gaussian = rng.normal(size=(dimension, columns))
    q_matrix, r_matrix = np.linalg.qr(gaussian, mode="reduced")
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return q_matrix * signs


def validate_two_level(params: dict[str, Any], method_name: str) -> None:
    mixing = float(params["lambda"])
    if not 0.0 <= mixing <= 1.0:
        raise ValueError(f"{method_name} lambda must lie in [0, 1]")


def two_level_residual(
    residual: np.ndarray,
    momentum: np.ndarray,
    rho: float,
    mixing: float,
) -> tuple[np.ndarray, np.ndarray]:
    new_momentum = rho * momentum + (1.0 - rho) * residual
    combined = mixing * residual + (1.0 - mixing) * new_momentum
    return new_momentum, combined
