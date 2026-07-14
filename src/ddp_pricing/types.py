from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunContext:
    week_id: str
    week_label: str
    run_index: int
    max_samples: int
    metric_samples: int


@dataclass
class MethodTrace:
    method: str
    sample_counts: list[int]
    objectives: list[float]
    final_x: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.sample_counts) != len(self.objectives):
            raise ValueError("sample_counts and objectives must have equal length")
        if not self.sample_counts or self.sample_counts[0] != 0:
            raise ValueError("a trace must begin at sample count zero")
        if any(b < a for a, b in zip(self.sample_counts, self.sample_counts[1:])):
            raise ValueError("sample counts must be nondecreasing")
        if not np.all(np.isfinite(self.final_x)):
            raise FloatingPointError(f"{self.method} produced a non-finite iterate")
