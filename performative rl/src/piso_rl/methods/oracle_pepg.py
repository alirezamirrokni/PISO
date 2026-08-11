from __future__ import annotations

from typing import Any

import numpy as np

from ..oracle_model import exact_oracle_gradient
from .common import (
    can_afford,
    finish,
    initialize_state,
    record,
    restore_or_initialize,
    save_step,
    scheduled_value,
)


class OraclePePG:
















    name = "OraclePePG"
    seed_alias = "OraclePePG"
    deterministic_across_seeds = True

    def __init__(self, params: dict[str, Any]) -> None:
        self.p = dict(params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        for key, default in (
            ("step_size", 2.0),
            ("armijo", 1e-4),
            ("backtrack_factor", 0.5),
            ("minimum_step", 1e-8),
        ):
            value = float(self.p.get(key, default))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"OraclePePG {key} must be finite and positive")
        factor = float(self.p.get("backtrack_factor", 0.5))
        if factor >= 1.0:
            raise ValueError("OraclePePG backtrack_factor must lie in (0, 1)")
        if int(self.p.get("max_backtracks", 20)) < 0:
            raise ValueError("OraclePePG max_backtracks must be nonnegative")
        if int(self.p.get("equivalent_batch", 100)) <= 0:
            raise ValueError("OraclePePG equivalent_batch must be positive")

    def _step_size(self, iteration: int, sample_count: int) -> float:
        return scheduled_value(
            self.p,
            "step_size",
            iteration=iteration,
            sample_count=sample_count,
            legacy_decay_key="step_size_decay",
        )

    def run(self, problem, rng, context, cache):
        del rng  

        
        
        dummy_rng = np.random.RandomState(0)
        state, _ = restore_or_initialize(
            cache,
            dummy_rng,
            lambda: initialize_state(problem, context.evaluation_interval),
            context.evaluation_interval,
        )
        state.setdefault("oracle_model_calls", 0)
        state.setdefault("accepted_steps", 0)
        state.setdefault("rejected_steps", 0)
        state.setdefault("last_gradient_norm", 0.0)
        state.setdefault("last_accepted_step_size", 0.0)

        equivalent_batch = int(self.p.get("equivalent_batch", 100))
        armijo = float(self.p.get("armijo", 1e-4))
        factor = float(self.p.get("backtrack_factor", 0.5))
        minimum_step = float(self.p.get("minimum_step", 1e-8))
        max_backtracks = int(self.p.get("max_backtracks", 20))

        while can_afford(state, equivalent_batch, context.max_trajectories):
            theta = np.asarray(state["theta"], dtype=np.float64)
            current_value = float(problem.exact_value(theta))
            gradient = np.asarray(exact_oracle_gradient(problem, theta), dtype=float)
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("OraclePePG produced a non-finite gradient")

            gradient_norm_sq = float(np.dot(gradient, gradient))
            initial_step = self._step_size(
                int(state["iteration"]), int(state["sample_count"])
            )
            accepted_theta = theta.copy()
            accepted_step = 0.0

            if gradient_norm_sq > 0.0:
                trial_step = initial_step
                for _ in range(max_backtracks + 1):
                    candidate = theta + trial_step * gradient
                    candidate_value = float(problem.exact_value(candidate))
                    sufficient_increase = (
                        current_value + armijo * trial_step * gradient_norm_sq
                    )
                    if candidate_value >= sufficient_increase:
                        accepted_theta = candidate
                        accepted_step = trial_step
                        state["accepted_steps"] = int(state["accepted_steps"]) + 1
                        break
                    state["rejected_steps"] = int(state["rejected_steps"]) + 1
                    trial_step *= factor
                    if trial_step < minimum_step:
                        break

            state["theta"] = accepted_theta
            state["sample_count"] = int(state["sample_count"]) + equivalent_batch
            state["iteration"] = int(state["iteration"]) + 1
            state["oracle_model_calls"] = int(state["oracle_model_calls"]) + 1
            state["last_gradient_norm"] = float(np.linalg.norm(gradient))
            state["last_accepted_step_size"] = float(accepted_step)

            record(state, problem, context.evaluation_interval)
            save_step(cache, state, dummy_rng)

        trace = finish(state, problem, context.evaluation_interval)
        trace.oracle_model_calls = int(state["oracle_model_calls"])
        trace.last_gradient_norm = float(state["last_gradient_norm"])
        return trace
