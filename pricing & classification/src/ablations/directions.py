from __future__ import annotations

from typing import Any

import numpy as np


def haar_frame(
    rng: np.random.RandomState,
    dimension: int,
    columns: int,
) -> np.ndarray:
    if not 1 <= int(columns) <= int(dimension):
        raise ValueError("frame columns must satisfy 1 <= columns <= dimension")
    gaussian = rng.normal(size=(dimension, columns))
    q_matrix, r_matrix = np.linalg.qr(gaussian, mode="reduced")
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return q_matrix * signs


def initialize_direction_state(
    family: str,
    dimension: int,
    cycle_length: int,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    if family == "coordinate_cycle":
        length = min(int(cycle_length), int(dimension))
        return {
            "order": rng.permutation(dimension),
            "position": 0,
            "length": length,
        }
    if family == "orthogonal_cycle":
        length = min(int(cycle_length), int(dimension))
        return {
            "frame": haar_frame(rng, dimension, length),
            "position": 0,
            "length": length,
        }
    return {}


def _single_direction(
    family: str,
    dimension: int,
    cycle_length: int,
    rng: np.random.RandomState,
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if family == "gaussian":
        direction = rng.normal(size=dimension)
        return direction, direction

    if family == "sphere":
        raw = rng.normal(size=dimension)
        norm = float(np.linalg.norm(raw))
        while norm == 0.0:
            raw = rng.normal(size=dimension)
            norm = float(np.linalg.norm(raw))
        direction = np.sqrt(dimension) * raw / norm
        return direction, direction

    if family == "rademacher":
        direction = rng.choice(np.asarray([-1.0, 1.0]), size=dimension)
        return direction, direction

    if family == "random_coordinate":
        index = int(rng.randint(0, dimension))
        direction = np.zeros(dimension, dtype=float)
        direction[index] = np.sqrt(dimension)
        return direction, direction

    if family == "coordinate_cycle":
        length = int(state.get("length", min(cycle_length, dimension)))
        position = int(state.get("position", 0))
        if "order" not in state or position >= length:
            state["order"] = rng.permutation(dimension)
            state["position"] = 0
            state["length"] = length
            position = 0
        index = int(np.asarray(state["order"])[position])
        state["position"] = position + 1
        direction = np.zeros(dimension, dtype=float)
        direction[index] = np.sqrt(dimension)
        return direction, direction

    if family == "orthogonal_cycle":
        length = int(state.get("length", min(cycle_length, dimension)))
        position = int(state.get("position", 0))
        if "frame" not in state or position >= length:
            state["frame"] = haar_frame(rng, dimension, length)
            state["position"] = 0
            state["length"] = length
            position = 0
        unit = np.asarray(state["frame"], dtype=float)[:, position]
        state["position"] = position + 1
        direction = np.sqrt(dimension) * unit
        return direction, direction

    raise ValueError(f"unknown direction family: {family}")


def sample_directions(
    family: str,
    dimension: int,
    count: int,
    cycle_length: int,
    rng: np.random.RandomState,
    state: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    directions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for _ in range(int(count)):
        direction, score = _single_direction(
            family,
            dimension,
            cycle_length,
            rng,
            state,
        )
        directions.append(np.asarray(direction, dtype=float))
        scores.append(np.asarray(score, dtype=float))
    return directions, scores
