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
    """Return h / ||h||, or the zero vector when h is zero."""
    norm = float(np.linalg.norm(momentum))
    if norm == 0.0:
        return np.zeros_like(momentum, dtype=float)
    return np.asarray(momentum, dtype=float) / norm


def _sample_direction(
    rng: np.random.RandomState,
    dimension: int,
    alpha: float,
    hint: np.ndarray,
) -> np.ndarray:
    """Sample N(0, (alpha/d)I + (1-alpha) hint hint^T)."""
    isotropic = np.sqrt(alpha / dimension) * rng.normal(size=dimension)
    guided = np.sqrt(1.0 - alpha) * rng.normal() * hint
    return isotropic + guided


def _covariance_product(
    vector: np.ndarray,
    alpha: float,
    hint: np.ndarray,
) -> np.ndarray:
    """Compute Sigma @ vector without materializing the covariance matrix."""
    dimension = vector.size
    return (
        (alpha / dimension) * vector
        + (1.0 - alpha) * hint * float(np.dot(hint, vector))
    )


class PISOM:
    """Momentum-shaped partially informed stochastic optimization.

    At each iteration the method uses an independent direct estimate of the
    known gradient component and two independent function-estimation batches at
    x +/- mu*u. The residual estimator updates the momentum buffer, which in
    turn defines the covariance used by the next perturbation.
    """

    name = "PISO_M"
    private_rng = True

    def __init__(self, params: dict) -> None:
        self.p = params
        alpha = float(params["alpha"])
        tau = float(params["tau"])
        if not 0.0 < alpha <= 1.0:
            raise ValueError("PISO_M alpha must satisfy 0 < alpha <= 1")
        if not 0.0 <= tau < 1.0:
            raise ValueError("PISO_M tau must satisfy 0 <= tau < 1")

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        alpha = float(p["alpha"])
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
            ),
        )

        while state["sample_count"] <= context.max_samples:
            mk = batch_size(p, state["iteration"])
            state["beta"] *= float(p["beta_decay"])

            hint = _normalized_hint(state["momentum"])
            direction = _sample_direction(rng, problem.n, alpha, hint)

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

            # This batch is sampled at x_k and independently of the direction
            # and the two perturbed function-estimation batches.
            known_demands = problem.sample_demands(state["x"], mk, rng)
            known_gradient = problem.partial_gradients(known_demands).mean(axis=0)

            finite_difference = (
                (plus.mean() - minus.mean()) / (2.0 * state["mu"])
            ) * direction
            residual = (
                finite_difference
                - float(np.dot(known_gradient, direction)) * direction
            )
            gradient = residual + _covariance_product(known_gradient, alpha, hint)

            state["sample_count"] += 3 * mk
            state["x"] = state["x"] - state["beta"] * gradient
            state["momentum"] = tau * state["momentum"] + (1.0 - tau) * residual
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]),
                float(p["mu_min"]),
            )
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
