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
    for key in ("rls_ridge", "residual_clip_norm", "directional_derivative_clip"):
        if key in params:
            value = float(params[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{method_name} {key} must be finite and positive")
    forgetting = float(params.get("rls_forgetting", 1.0))
    if not 0.0 < forgetting <= 1.0:
        raise ValueError(f"{method_name} rls_forgetting must lie in (0, 1]")
    if "gradient_clip_norm" in params:
        clip = float(params["gradient_clip_norm"])
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError(
                f"{method_name} gradient_clip_norm must be finite and positive"
            )


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
    """Allocate a fair per-update trajectory budget.

    New configurations set ``total_batch`` explicitly.  Older configurations
    retain the previous convention where ``batch_initial`` denoted one third of
    the total PISO cost.
    """

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
    """Sample the covariance-shaped direction used by the published GZO update."""
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


def two_level_residual(
    slow_residual: np.ndarray,
    fast_residual: np.ndarray,
    mixing: float,
) -> np.ndarray:
    return (1.0 - mixing) * slow_residual + mixing * fast_residual


def clip_vector(vector: np.ndarray, max_norm: float | None) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if max_norm is None:
        return array
    limit = float(max_norm)
    norm = float(np.linalg.norm(array))
    if norm == 0.0 or norm <= limit:
        return array
    return array * (limit / norm)


def update_residual_rls(
    state: dict[str, Any],
    directions: list[np.ndarray],
    derivatives: list[float],
    *,
    forgetting: float,
    ridge: float,
    clip_norm: float | None,
    warmup_directions: int,
) -> np.ndarray:
    """Fit the unknown environment-shift gradient from directional queries.

    Each observation follows ``y_i ≈ u_i^T g_shift``.  A regularized,
    exponentially weighted least-squares fit uses the full query history,
    instead of injecting a single extremely noisy SPSA vector into every actor
    update.  The confidence ramp suppresses early under-determined estimates.
    """

    gram = np.asarray(state["residual_gram"], dtype=float)
    rhs = np.asarray(state["residual_rhs"], dtype=float)
    gram *= float(forgetting)
    rhs *= float(forgetting)
    for direction, derivative in zip(directions, derivatives):
        u = np.asarray(direction, dtype=float)
        gram += np.outer(u, u)
        rhs += float(derivative) * u

    dimension = gram.shape[0]
    estimate = np.linalg.solve(
        gram + float(ridge) * np.eye(dimension, dtype=float), rhs
    )
    estimate = clip_vector(estimate, clip_norm)
    observations = int(state.get("residual_observations", 0)) + len(directions)
    confidence = min(1.0, observations / float(max(int(warmup_directions), 1)))
    estimate *= confidence

    state["residual_gram"] = gram
    state["residual_rhs"] = rhs
    state["residual_observations"] = observations
    state["residual_estimate"] = estimate
    return estimate
