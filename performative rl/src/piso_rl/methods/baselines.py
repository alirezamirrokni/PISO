from __future__ import annotations

import numpy as np

from .common import (
    batch_size,
    can_afford,
    clip_gradient,
    finish,
    initialize_state,
    record,
    restore_or_initialize,
    save_step,
    scheduled_value,
)
from .piso_utils import guided_direction_and_score, known_gradient


def _schedule(
    params: dict,
    name: str,
    iteration: int,
    sample_count: int,
) -> float:
    legacy_key = "step_size_decay" if name == "step_size" else f"{name}_decay"
    return scheduled_value(
        params,
        name,
        iteration=iteration,
        sample_count=sample_count,
        legacy_decay_key=legacy_key,
    )


def _zo_ogvr_baseline(
    theta: np.ndarray,
    history: list[dict],
    *,
    distance_weight: float,
    window: int,
) -> float:
    """Estimate the one-point variance-reduction parameter from past samples.

    This is the finite-window version of Algorithm 1 in Hikima and Takeda
    (AAAI 2025).  In this RL benchmark the stochastic sample ``xi`` is a
    complete trajectory and ``f(theta, xi)`` is its negative discounted
    return.  Once a trajectory is sampled, that realized loss is independent
    of the argument ``theta``; therefore the past loss values can be reused
    directly without new environment trajectories or cross-deployment access.
    """

    if int(window) <= 0:
        raise ValueError("ZO_OGVR history_window must be positive")
    if float(distance_weight) < 0.0 or not np.isfinite(distance_weight):
        raise ValueError("ZO_OGVR distance_weight must be finite and nonnegative")
    recent = list(history[-int(window) :])
    if not recent:
        raise ValueError("ZO_OGVR baseline history must be nonempty")

    current = np.asarray(theta, dtype=float)
    inverse_scales: list[float] = []
    means: list[float] = []
    for entry in recent:
        query_theta = np.asarray(entry["query_theta"], dtype=float)
        losses = np.asarray(entry["losses"], dtype=float)
        if losses.ndim != 1 or losses.size == 0:
            raise ValueError("ZO_OGVR stored losses must be a nonempty vector")
        scale = (
            float(distance_weight) * float(np.sum((current - query_theta) ** 2))
            + 1.0 / float(losses.size)
        )
        if not np.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("ZO_OGVR produced an invalid history weight")
        inverse_scales.append(1.0 / scale)
        means.append(float(losses.mean()))

    normalizer = float(np.sum(inverse_scales))
    weights = np.asarray(inverse_scales, dtype=float) / normalizer
    return float(np.dot(weights, np.asarray(means, dtype=float)))


class VanillaPG:
    """Classical REINFORCE using only the known gradient component g_K."""

    name = "VanillaPG"
    seed_alias = "VanillaPG"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)

    def run(self, problem, rng, context, cache):
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initialize_state(problem, context.evaluation_interval),
            context.evaluation_interval,
        )
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            if not can_afford(state, count, context.max_trajectories):
                break
            trajectories = problem.sample_observations(state["theta"], count, rng)
            gradient = known_gradient(problem, state["theta"], trajectories)
            gradient = clip_gradient(
                gradient, self.p.get("gradient_clip_norm")
            )
            eta = _schedule(
                self.p, "step_size", iteration, int(state["sample_count"])
            )
            state["theta"] = state["theta"] - eta * gradient
            state["sample_count"] += count
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)
        return finish(state, problem, context.evaluation_interval)


class ZOTG:
    name = "ZO_TG"
    seed_alias = "ZO_TG"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)

    def run(self, problem, rng, context, cache):
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initialize_state(problem, context.evaluation_interval),
            context.evaluation_interval,
        )
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            cost = 2 * count
            if not can_afford(state, cost, context.max_trajectories):
                break
            current_samples = int(state["sample_count"])
            mu = _schedule(self.p, "radius", iteration, current_samples) / np.sqrt(
                problem.n
            )
            eta = _schedule(self.p, "step_size", iteration, current_samples)
            direction = rng.normal(size=problem.n)
            plus, _, minus, _ = problem.sample_paired_losses(
                state["theta"] + mu * direction,
                state["theta"] - mu * direction,
                count,
                rng,
            )
            gradient = (
                float((plus.mean() - minus.mean()) / (2.0 * mu)) * direction
            )
            gradient = clip_gradient(
                gradient, self.p.get("gradient_clip_norm")
            )
            state["theta"] = state["theta"] - eta * gradient
            state["sample_count"] += cost
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)
        return finish(state, problem, context.evaluation_interval)


class ZOOG:
    name = "ZO_OG"
    seed_alias = "ZO_OG"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)

    def run(self, problem, rng, context, cache):
        def initialize():
            state = initialize_state(problem, context.evaluation_interval)
            state["loss_baseline"] = None
            return state

        state, _ = restore_or_initialize(
            cache, rng, initialize, context.evaluation_interval
        )
        state.setdefault("loss_baseline", None)
        baseline_decay = float(self.p.get("baseline_decay", 0.9))
        if not 0.0 <= baseline_decay < 1.0:
            raise ValueError("ZO_OG baseline_decay must lie in [0, 1)")

        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            if not can_afford(state, count, context.max_trajectories):
                break
            current_samples = int(state["sample_count"])
            mu = _schedule(self.p, "radius", iteration, current_samples) / np.sqrt(
                problem.n
            )
            eta = _schedule(self.p, "step_size", iteration, current_samples)
            direction = rng.normal(size=problem.n)
            losses, _ = problem.sample_losses(
                state["theta"] + mu * direction, count, rng
            )
            mean_loss = float(losses.mean())
            previous_baseline = state["loss_baseline"]
            if previous_baseline is None:
                gradient = np.zeros(problem.n, dtype=float)
                state["loss_baseline"] = mean_loss
            else:
                gradient = ((mean_loss - float(previous_baseline)) / mu) * direction
                state["loss_baseline"] = (
                    baseline_decay * float(previous_baseline)
                    + (1.0 - baseline_decay) * mean_loss
                )
            gradient = clip_gradient(
                gradient, self.p.get("gradient_clip_norm")
            )
            state["theta"] = state["theta"] - eta * gradient
            state["sample_count"] += count
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)
        return finish(state, problem, context.evaluation_interval)


class ZOOGVR:
    """One-point ZO with the history-weighted variance-reduction parameter.

    The estimator is

        ((mean(loss(theta + mu*u)) - c_k) / mu) * u,

    where ``c_k`` is reconstructed from the most recent one-point deployment
    batches using the distance-aware weights of Hikima and Takeda (2025).
    """

    name = "ZO_OGVR"
    seed_alias = "ZO_OGVR"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        window = int(self.p.get("history_window", 10))
        if window <= 0:
            raise ValueError("ZO_OGVR history_window must be positive")
        distance_weight = float(self.p.get("distance_weight", 0.1))
        if not np.isfinite(distance_weight) or distance_weight < 0.0:
            raise ValueError(
                "ZO_OGVR distance_weight must be finite and nonnegative"
            )
        initial_batch = int(
            self.p.get("initial_baseline_batch", self.p["batch_initial"])
        )
        if initial_batch <= 0:
            raise ValueError("ZO_OGVR initial_baseline_batch must be positive")

    def run(self, problem, rng, context, cache):
        def initialize():
            state = initialize_state(problem, context.evaluation_interval)
            state["vr_baseline"] = None
            state["vr_history"] = []
            return state

        state, _ = restore_or_initialize(
            cache, rng, initialize, context.evaluation_interval
        )
        state.setdefault("vr_baseline", None)
        state.setdefault("vr_history", [])

        # Algorithm 1 receives c_0 as an input.  Estimate it once from an
        # ordinary deployment at theta_0 and charge those trajectories to the
        # same cumulative sample budget used by every other method.
        if state["vr_baseline"] is None:
            initial_count = int(
                self.p.get("initial_baseline_batch", self.p["batch_initial"])
            )
            if not can_afford(state, initial_count, context.max_trajectories):
                return finish(state, problem, context.evaluation_interval)
            initial_losses, _ = problem.sample_losses(
                state["theta"], initial_count, rng
            )
            state["vr_baseline"] = float(initial_losses.mean())
            state["sample_count"] += initial_count
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)

        window = int(self.p.get("history_window", 10))
        distance_weight = float(self.p.get("distance_weight", 0.1))

        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            if not can_afford(state, count, context.max_trajectories):
                break

            current_samples = int(state["sample_count"])
            mu = _schedule(
                self.p, "radius", iteration, current_samples
            ) / np.sqrt(problem.n)
            eta = _schedule(self.p, "step_size", iteration, current_samples)
            direction = rng.normal(size=problem.n)
            query_theta = np.asarray(
                state["theta"] + mu * direction, dtype=float
            )
            losses, _ = problem.sample_losses(query_theta, count, rng)
            mean_loss = float(losses.mean())
            baseline = float(state["vr_baseline"])
            gradient = ((mean_loss - baseline) / mu) * direction
            gradient = clip_gradient(
                gradient, self.p.get("gradient_clip_norm")
            )

            next_theta = np.asarray(
                state["theta"] - eta * gradient, dtype=float
            )
            state["vr_history"].append(
                {
                    "query_theta": query_theta.copy(),
                    "losses": np.asarray(losses, dtype=float).copy(),
                }
            )
            if len(state["vr_history"]) > window:
                state["vr_history"] = state["vr_history"][-window:]
            state["vr_baseline"] = _zo_ogvr_baseline(
                next_theta,
                state["vr_history"],
                distance_weight=distance_weight,
                window=window,
            )

            state["theta"] = next_theta
            state["sample_count"] += count
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)

        return finish(state, problem, context.evaluation_interval)


class GZONS:
    name = "GZO_NS"
    seed_alias = "GZO_NS"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        alpha = float(self.p["alpha"])
        if not 0.0 < alpha <= 1.0:
            raise ValueError("GZO_NS alpha must lie in (0, 1]")

    def run(self, problem, rng, context, cache):
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initialize_state(problem, context.evaluation_interval),
            context.evaluation_interval,
        )
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            total_batch = int(self.p.get("total_batch", 3 * count))
            if total_batch < 3:
                raise ValueError("GZO_NS total_batch must be at least 3")
            known_count = int(self.p.get("known_batch", total_batch - 2 * (total_batch // 3)))
            query_count = (total_batch - known_count) // 2
            if known_count + 2 * query_count != total_batch or min(known_count, query_count) <= 0:
                raise ValueError("GZO_NS batch allocation must be positive and sum to total_batch")
            cost = total_batch
            if not can_afford(state, cost, context.max_trajectories):
                break
            observations = problem.sample_observations(state["theta"], known_count, rng)
            guide = known_gradient(problem, state["theta"], observations)
            direction, score = guided_direction_and_score(
                rng,
                problem.n,
                float(self.p["alpha"]),
                np.asarray(guide, dtype=float),
            )
            current_samples = int(state["sample_count"])
            mu = _schedule(self.p, "radius", iteration, current_samples)
            eta = _schedule(self.p, "step_size", iteration, current_samples)
            plus, _, minus, _ = problem.sample_paired_losses(
                state["theta"] + mu * direction,
                state["theta"] - mu * direction,
                query_count,
                rng,
            )
            gradient = (
                float((plus.mean() - minus.mean()) / (2.0 * mu)) * score
            )
            gradient = clip_gradient(
                gradient, self.p.get("gradient_clip_norm")
            )
            state["theta"] = state["theta"] - eta * gradient
            state["sample_count"] += cost
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)
        return finish(state, problem, context.evaluation_interval)


class GZOHS:
    name = "GZO_HS"
    seed_alias = "GZO_HS"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        alpha = float(self.p["alpha"])
        if not 0.0 < alpha <= 1.0:
            raise ValueError("GZO_HS alpha must lie in (0, 1]")
        if int(self.p["window"]) <= 0:
            raise ValueError("GZO_HS window must be positive")

    def run(self, problem, rng, context, cache):
        def initialize():
            state = initialize_state(problem, context.evaluation_interval)
            state["history"] = []
            state["guide"] = np.zeros(problem.n, dtype=float)
            return state

        state, _ = restore_or_initialize(
            cache, rng, initialize, context.evaluation_interval
        )
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            cost = 2 * count
            if not can_afford(state, cost, context.max_trajectories):
                break
            guide = np.asarray(state["guide"], dtype=float)
            direction, score = guided_direction_and_score(
                rng,
                problem.n,
                float(self.p["alpha"]),
                np.asarray(guide, dtype=float),
            )
            current_samples = int(state["sample_count"])
            mu = _schedule(self.p, "radius", iteration, current_samples)
            eta = _schedule(self.p, "step_size", iteration, current_samples)
            plus_theta = state["theta"] + mu * direction
            minus_theta = state["theta"] - mu * direction
            plus, plus_trajectories, minus, minus_trajectories = (
                problem.sample_paired_losses(
                    plus_theta, minus_theta, count, rng
                )
            )
            gradient = (
                float((plus.mean() - minus.mean()) / (2.0 * mu)) * score
            )
            gradient = clip_gradient(
                gradient, self.p.get("gradient_clip_norm")
            )
            state["history"].append(
                (
                    plus_theta.copy(),
                    plus_trajectories,
                    minus_theta.copy(),
                    minus_trajectories,
                )
            )
            window = int(self.p["window"])
            if len(state["history"]) > window:
                state["history"] = state["history"][-window:]
            state["theta"] = state["theta"] - eta * gradient
            state["sample_count"] += cost

            gradients = []
            for p_theta, p_traj, m_theta, m_traj in state["history"]:
                gradients.append(known_gradient(problem, p_theta, p_traj))
                gradients.append(known_gradient(problem, m_theta, m_traj))
            state["guide"] = (
                np.mean(gradients, axis=0)
                if gradients
                else np.zeros(problem.n)
            )
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)
        return finish(state, problem, context.evaluation_interval)
