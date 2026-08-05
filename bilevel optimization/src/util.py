from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def hash_mapping(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def pickle_load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def unit_sphere_direction(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=int(dimension))
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-15:
        direction[0] = 1.0
        return direction
    return direction / norm
