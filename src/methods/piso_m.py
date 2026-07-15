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
    shape_alpha: float,
    hint: np.ndarray,
) -> np.ndarray:
    """Sample N(0, (shape_alpha/d)I + (1-shape_alpha)hh^T)."""
    isotropic = np.sqrt(shape_alpha / dimension) * rng.normal(size=dimension)
    guided = np.sqrt(1.0 - shape_alpha) * rng.normal() * hint
    return isotropic + guided


def _covariance_product(
    vector: np.ndarray,
    shape_alpha: float,
    hint: np.ndarray,
) -> np.ndarray:
    dimension = vector.size
    return (
        (shape_alpha / dimension) * vector
        + (1.0 - shape_alpha) * hint * float(np.dot(hint, vector))
    )


def _spectral_step_scale(
    dimension: int,
    shape_alpha: float,
    hint: np.ndarray,
) -> float:
    """Normalize the largest eigenvalue of the effective preconditioner to one.

    The raw estimator has expectation Sigma times the partially weighted
    gradient. Using the same numerical learning-rate range as an identity-
    covariance method therefore requires compensating for Sigma's scale.
    Dividing by lambda_max(Sigma) is the conservative normalization: it becomes
    a factor d when shape_alpha=1, while retaining the intended anisotropy for
    shaped covariances.
    """
    if float(np.linalg.norm(hint)) == 0.0:
        lambda_max = shape_alpha / dimension
    else:
        lambda_max = shape_alpha / dimension + (1.0 - shape_alpha)
    return 1.0 / lambda_max


class PISOM:
    name = "PISO_M"
    private_rng = True

    def __init__(self, params: dict) -> None:
        self.p = params
        residual_alpha = float(params["residual_alpha"])
        shape_alpha = float(params["shape_alpha"])
        tau = float(params["tau"])
        if params.get("step_normalization") != "spectral":
            raise ValueError("PISO_M step_normalization must be 'spectral'")
        if not 0.0 < residual_alpha <= 1.0:
            raise ValueError("PISO_M residual_alpha must satisfy 0 < residual_alpha <= 1")
        if not 0.0 < shape_alpha <= 1.0:
            raise ValueError("PISO_M shape_alpha must satisfy 0 < shape_alpha <= 1")
        if not 0.0 <= tau < 1.0:
            raise ValueError("PISO_M tau must satisfy 0 <= tau < 1")

    def run(self, problem, rng, context, cache, progress=None):
        p = self.p
        residual_alpha = float(p["residual_alpha"])
        shape_alpha = float(p["shape_alpha"])
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
            direction = _sample_direction(rng, problem.n, shape_alpha, hint)

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
                residual_alpha * residual
                + _covariance_product(known_gradient, shape_alpha, hint)
            )

            # E[gradient] is Sigma(g_K + residual_alpha*g_U), not the
            # unpreconditioned gradient. Spectral normalization preserves the
            # common learning-rate scale without destroying the momentum shape.
            step_scale = _spectral_step_scale(problem.n, shape_alpha, hint)

            state["sample_count"] += 3 * mk
            state["x"] = state["x"] - state["beta"] * step_scale * gradient
            # The raw residual, not its externally weighted version, is used to
            # track the unknown component's direction.
            state["momentum"] = tau * state["momentum"] + (1.0 - tau) * residual
            state["mu"] = max(
                state["mu"] * float(p["mu_decay"]),
                float(p["mu_min"]),
            )
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
