from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Trace:
    samples: list[int]
    values: list[float]
    occupancy_changes: list[float]
    final_theta: np.ndarray
    method: str = ""
    seed: int = 0
    runtime_seconds: float = 0.0
    oracle_model_calls: int = 0
    environment_transitions: int = 0
    minimum_follower_action_gap: float = float("nan")
    last_gradient_norm: float = float("nan")
    last_policy_gradient_norm: float = float("nan")
    last_transition_gradient_norm: float = float("nan")
    last_reward_gradient_norm: float = float("nan")
    last_value_loss: float = float("nan")


def batch_size(params: dict[str, Any], iteration: int) -> int:
    value = int(params["batch_initial"]) + int(iteration) * int(
        params.get("batch_increment", 0)
    )
    if value <= 0:
        raise ValueError("batch size must be positive")
    return value


def scheduled_value(
    params: dict[str, Any],
    name: str,
    *,
    iteration: int,
    sample_count: int,
    legacy_decay_key: str | None = None,
) -> float:
    """Return a positive, budget-aware optimizer parameter.

    New configurations should use ``<name>_schedule`` with one of:
    ``constant``, ``inverse_sqrt``, or ``exponential_samples``.  When no
    schedule is supplied, the legacy per-iteration exponential behavior is
    retained for backward compatibility.
    """

    initial = float(params[name])
    if not np.isfinite(initial) or initial <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    schedule_key = f"{name}_schedule"
    schedule = params.get(schedule_key)
    if schedule is None:
        decay_key = legacy_decay_key or f"{name}_decay"
        decay = float(params.get(decay_key, 1.0))
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"{decay_key} must lie in (0, 1]")
        value = initial * decay ** int(iteration)
    else:
        schedule = str(schedule).strip().lower()
        if schedule == "constant":
            value = initial
        elif schedule == "inverse_sqrt":
            scale = float(params.get(f"{name}_scale", 100_000.0))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{name}_scale must be finite and positive")
            value = initial / np.sqrt(1.0 + float(sample_count) / scale)
        elif schedule == "exponential_samples":
            decay = float(params.get(f"{name}_decay", 1.0))
            interval = float(params.get(f"{name}_decay_interval", 1_000.0))
            if not 0.0 < decay <= 1.0:
                raise ValueError(f"{name}_decay must lie in (0, 1]")
            if not np.isfinite(interval) or interval <= 0.0:
                raise ValueError(
                    f"{name}_decay_interval must be finite and positive"
                )
            value = initial * decay ** (float(sample_count) / interval)
        else:
            raise ValueError(
                f"unknown {schedule_key}={schedule!r}; expected constant, "
                "inverse_sqrt, or exponential_samples"
            )

    minimum = float(params.get(f"{name}_min", 0.0))
    if minimum < 0.0 or not np.isfinite(minimum):
        raise ValueError(f"{name}_min must be finite and nonnegative")
    return float(max(value, minimum))


def clip_gradient(gradient: np.ndarray, max_norm: float | None) -> np.ndarray:
    array = np.asarray(gradient, dtype=float)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("gradient contains a non-finite value")
    if max_norm is None:
        return array
    limit = float(max_norm)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    norm = float(np.linalg.norm(array))
    if norm <= limit or norm == 0.0:
        return array
    return array * (limit / norm)


def normalized_occupancy_shift(current: np.ndarray, previous: np.ndarray) -> float:
    """Symmetric normalized L1 shift, guaranteed to lie in [0, 1]."""

    current_array = np.asarray(current, dtype=float)
    previous_array = np.asarray(previous, dtype=float)
    numerator = float(np.sum(np.abs(current_array - previous_array)))
    denominator = float(
        np.sum(np.abs(current_array)) + np.sum(np.abs(previous_array))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _next_interval(sample_count: int, evaluation_interval: int) -> int:
    interval = int(evaluation_interval)
    if interval <= 0:
        raise ValueError("evaluation_interval must be positive")
    return (int(sample_count) // interval + 1) * interval


def initialize_state(problem, evaluation_interval: int) -> dict[str, Any]:
    _, rewards, _, occupancy = problem.deploy_exact(problem.initial_x)
    initial_value = float(np.sum(np.asarray(occupancy) * np.asarray(rewards)))
    return {
        "theta": np.asarray(problem.initial_x, dtype=float).copy(),
        "samples": [0],
        "values": [initial_value],
        "occupancy_changes": [0.0],
        "last_occupancy": np.asarray(occupancy, dtype=float).copy(),
        "sample_count": 0,
        "iteration": 0,
        "next_evaluation_sample": int(evaluation_interval),
    }


def _normalize_state(state: dict[str, Any], evaluation_interval: int) -> dict[str, Any]:
    """Migrate progress caches created before sparse evaluation was added."""
    state.setdefault("samples", [0])
    state.setdefault("values", [])
    state.setdefault("occupancy_changes", [0.0])
    state.setdefault("sample_count", int(state["samples"][-1]))
    state.setdefault("iteration", 0)
    state["next_evaluation_sample"] = max(
        int(state.get("next_evaluation_sample", 0)),
        _next_interval(int(state["sample_count"]), evaluation_interval),
    )
    return state


def restore_or_initialize(
    cache,
    rng: np.random.RandomState,
    initializer,
    evaluation_interval: int,
):
    saved = cache.load_progress()
    if saved is None:
        state = _normalize_state(initializer(), evaluation_interval)
        cache.save_progress(state, rng.get_state())
        return state, False
    rng.set_state(saved["rng_state"])
    return _normalize_state(saved["state"], evaluation_interval), True


def record(
    state: dict[str, Any],
    problem,
    evaluation_interval: int,
    *,
    force: bool = False,
) -> bool:
    """Record an exact metric only at the configured trajectory interval."""
    sample_count = int(state["sample_count"])
    next_sample = int(
        state.get(
            "next_evaluation_sample",
            _next_interval(sample_count, evaluation_interval),
        )
    )
    if not force and sample_count < next_sample:
        return False
    if state["samples"] and int(state["samples"][-1]) == sample_count:
        state["next_evaluation_sample"] = _next_interval(
            sample_count, evaluation_interval
        )
        return False

    _, rewards, _, occupancy = problem.deploy_exact(state["theta"])
    occupancy = np.asarray(occupancy, dtype=float)
    previous = np.asarray(state["last_occupancy"], dtype=float)
    change = normalized_occupancy_shift(occupancy, previous)

    state["samples"].append(sample_count)
    state["values"].append(float(np.sum(occupancy * np.asarray(rewards))))
    state["occupancy_changes"].append(change)
    state["last_occupancy"] = occupancy.copy()
    state["next_evaluation_sample"] = _next_interval(
        sample_count, evaluation_interval
    )
    return True


def save_step(cache, state: dict[str, Any], rng: np.random.RandomState) -> None:
    cache.save_progress(state, rng.get_state())


def finish(
    state: dict[str, Any],
    problem=None,
    evaluation_interval: int | None = None,
) -> Trace:
    if problem is not None:
        if evaluation_interval is None:
            raise ValueError("evaluation_interval is required when finalizing a problem")
        record(state, problem, evaluation_interval, force=True)
    return Trace(
        samples=[int(x) for x in state["samples"]],
        values=[float(x) for x in state["values"]],
        occupancy_changes=[float(x) for x in state["occupancy_changes"]],
        final_theta=np.asarray(state["theta"], dtype=float).copy(),
    )


def can_afford(state: dict[str, Any], cost: int, max_trajectories: int) -> bool:
    return int(state["sample_count"]) + int(cost) <= int(max_trajectories)
