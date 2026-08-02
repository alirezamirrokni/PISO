from __future__ import annotations

from typing import Any

import numpy as np


def validate_two_level_params(params: dict[str, Any], method_name: str) -> None:
    """Validate the extra mixing coefficient used by PISO^2."""
    value = float(params["lambda"])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{method_name} lambda must lie in [0, 1]")


def two_level_residual(
    residual: np.ndarray,
    momentum: np.ndarray,
    rho: float,
    mixing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return m_(k+1) and lambda R_k + (1-lambda) m_(k+1)."""
    new_momentum = rho * momentum + (1.0 - rho) * residual
    two_level = mixing * residual + (1.0 - mixing) * new_momentum
    return new_momentum, two_level
