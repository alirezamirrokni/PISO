from __future__ import annotations

from typing import Any

import numpy as np

from .common import scheduled_value


def validate_common_params(params: dict[str, Any], method_name: str) -> None:
    for key in ("radius", "step_size", "gamma", "batch_initial"):
        value = float(params[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{method_name} {key} must be finite and positive")
    rho = float(params["rho"])
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"{method_name} rho must lie in [0, 1)")
    known_fraction = float(params["known_fraction"])
    if not 0.0 < known_fraction < 1.0:
        raise ValueError(f"{method_name} known_fraction must lie in (0, 1)")
    if int(params.get("batch_increment", 0)) < 0:
        raise ValueError(f"{method_name} batch_increment must be nonnegative")
    if int(params.get("directions_per_update", 1)) <= 0:
        raise ValueError(f"{method_name} directions_per_update must be positive")
    rho_k = float(params.get("rho_k", 0.0))
    if not 0.0 <= rho_k < 1.0:
        raise ValueError(f"{method_name} rho_k must lie in [0, 1)")
    if int(params.get("directions_per_update", 1)) != 1:
        raise ValueError("Algorithm 1 uses exactly one direction per update")
    allowed = {
        "radius", "radius_schedule", "radius_scale", "radius_min", "radius_decay",
        "radius_decay_interval", "step_size", "step_size_schedule", "step_size_scale",
        "step_size_min", "step_size_decay", "step_size_decay_interval", "step_decay",
        "gamma", "rho", "rho_k", "lambda", "known_fraction", "batch_initial",
        "total_batch", "batch_increment", "directions_per_update", "cycle_length",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unsupported PISO parameters: {sorted(unknown)}")


def step_size(params: dict, iteration: int, sample_count: int) -> float:
    return scheduled_value(
        params,
        "step_size",
        iteration=iteration,
        sample_count=sample_count,
        legacy_decay_key="step_decay",
    )


def radius(params: dict, iteration: int, sample_count: int) -> float:
    return scheduled_value(
        params,
        "radius",
        iteration=iteration,
        sample_count=sample_count,
        legacy_decay_key="radius_decay",
    )


def allocate_batches(
    params: dict[str, Any], iteration_batch: int
) -> tuple[int, int, int]:

    total = int(params.get("total_batch", 3 * int(iteration_batch)))
    if total < 3:
        raise ValueError("PISO total_batch must be at least 3")
    known = int(round(total * float(params["known_fraction"])))
    remaining = total - known

    if remaining % 2:
        known += 1
        remaining -= 1
    plus = remaining // 2
    minus = remaining // 2
    if min(known, plus, minus) <= 0:
        raise ValueError("PISO batch split must allocate at least one trajectory per query")
    if known + plus + minus != total:
        raise RuntimeError("internal PISO batch accounting error")
    return known, plus, minus


def split_direction_batches(
    paired_count: int, directions_per_update: int
) -> list[int]:
    paired_count = int(paired_count)
    directions_per_update = int(directions_per_update)
    if paired_count < directions_per_update:
        raise ValueError(
            "plus/minus batch must provide at least one pair per direction"
        )
    quotient, remainder = divmod(paired_count, directions_per_update)
    return [quotient + (index < remainder) for index in range(directions_per_update)]


def known_gradient(problem, theta: np.ndarray, trajectories) -> np.ndarray:
    return problem.partial_gradients(theta, trajectories).mean(axis=0)


def guided_covariance_geometry(
    alpha: float, dimension: int, hint: np.ndarray
) -> tuple[float, float]:
    orthogonal = alpha / float(dimension)
    parallel = (
        1.0 - alpha + orthogonal
        if np.linalg.norm(hint) > 0.0
        else orthogonal
    )
    return parallel, orthogonal


def guided_direction(
    rng: np.random.RandomState,
    dimension: int,
    alpha: float,
    hint: np.ndarray,
) -> np.ndarray:

    norm = float(np.linalg.norm(hint))
    unit_hint = hint / norm if norm > 0.0 else np.zeros(dimension, dtype=float)
    orthogonal = alpha / float(dimension)
    return (
        np.sqrt(orthogonal) * rng.normal(size=dimension)
        + np.sqrt(1.0 - alpha) * float(rng.normal()) * unit_hint
    )


def guided_direction_and_score(
    rng: np.random.RandomState,
    dimension: int,
    alpha: float,
    hint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norm = float(np.linalg.norm(hint))
    unit_hint = hint / norm if norm > 0.0 else np.zeros(dimension, dtype=float)
    parallel, orthogonal = guided_covariance_geometry(alpha, dimension, unit_hint)
    direction = (
        np.sqrt(orthogonal) * rng.normal(size=dimension)
        + np.sqrt(1.0 - alpha) * float(rng.normal()) * unit_hint
    )
    if norm > 0.0:
        projection = float(np.dot(unit_hint, direction))
        parallel_component = projection * unit_hint
        orthogonal_component = direction - parallel_component
        score = parallel_component / parallel + orthogonal_component / orthogonal
    else:
        score = direction / orthogonal
    return direction, score


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

