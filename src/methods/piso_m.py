from __future__ import annotations

import numpy as np

from src.methods.common import (
    batch_size,
    finish,
    initial_state,
    record,
    restore_or_initialize,
    save_step,
)


def _normalized_hint(momentum: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(momentum))
    if norm == 0.0:
        return np.zeros_like(momentum, dtype=float)
    return np.asarray(momentum, dtype=float) / norm


def _sample_direction(
    rng: np.random.RandomState,
    dimension: int,
    shape_weight: float,
    hint: np.ndarray,
) -> np.ndarray:
    """Sample N(0, (q/d)I + (1-q)hh^T) for the current q."""
    isotropic = np.sqrt(shape_weight / dimension) * rng.normal(size=dimension)
    guided = np.sqrt(1.0 - shape_weight) * rng.normal() * hint
    return isotropic + guided


def _covariance_product(
    vector: np.ndarray,
    shape_weight: float,
    hint: np.ndarray,
) -> np.ndarray:
    dimension = vector.size
    return (
        (shape_weight / dimension) * vector
        + (1.0 - shape_weight) * hint * float(np.dot(hint, vector))
    )


class PISOM:
    name = "PISO_M"
    private_rng = True

    def __init__(self, params: dict) -> None:
        self.p = params
        alpha0 = float(params["alpha0"])
        damping = float(params["alpha_damping"])
        tau = float(params["tau"])
        if not 0.0 <= alpha0 <= 1.0:
            raise ValueError("PISO_M alpha0 must satisfy 0 <= alpha0 <= 1")
        if not 0.0 <= damping < 1.0:
            raise ValueError("PISO_M alpha_damping must satisfy 0 <= alpha_damping < 1")
        if not 0.0 <= tau < 1.0:
            raise ValueError("PISO_M tau must satisfy 0 <= tau < 1")

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        damping = float(p["alpha_damping"])
        tau = float(p["tau"])

        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initial_state(
                problem,
                context.metric_samples,
                rng,
                mu=float(p["mu0"]),
                beta=float(p["beta0"]),
                momentum=np.zeros(problem.n, dtype=float),
                residual_weight=float(p["alpha0"]),
                shape_weight=float(p["alpha0"]),
            ),
        )

        while state["sample_count"] <= context.max_samples:
            mk = batch_size(p, state["iteration"])
            state["beta"] *= float(p["beta_decay"])

            hint = _normalized_hint(state["momentum"])
            direction = _sample_direction(
                rng,
                problem.n,
                state["shape_weight"],
                hint,
            )

            plus, _ = problem.sample_losses(
                state["x"] + state["mu"] * direction,
                mk,
                rng,
            )
            minus, _ = problem.sample_losses(
                state["x"] - state["mu"] * direction,
                mk,
                rng,
            )
            known_demands = problem.sample_demands(state["x"], mk, rng)
            known_gradient = problem.partial_gradients(known_demands).mean(axis=0)

            finite_difference = (
                (plus.mean() - minus.mean()) / (2.0 * state["mu"])
            ) * direction
            residual = (
                finite_difference
                - float(np.dot(known_gradient, direction)) * direction
            )
            gradient = (
                state["residual_weight"] * residual
                + _covariance_product(
                    known_gradient,
                    state["shape_weight"],
                    hint,
                )
            )

            state["sample_count"] += 3 * mk
            # No spectral normalization is applied. The configured learning
            # rate multiplies the estimator directly.
            state["x"] = state["x"] - state["beta"] * gradient
            state["momentum"] = (
                tau * state["momentum"] + (1.0 - tau) * residual
            )
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]),
                float(p["mu_min"]),
            )
            state["residual_weight"] = (
                1.0 - damping * (1.0 - state["residual_weight"])
            )
            state["shape_weight"] = (
                1.0 - damping * (1.0 - state["shape_weight"])
            )
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
