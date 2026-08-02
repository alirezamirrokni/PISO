from __future__ import annotations

import numpy as np

from .common import (
    batch_size,
    can_afford,
    finish,
    initialize_state,
    record,
    restore_or_initialize,
    save_step,
)
from .piso_utils import guided_covariance_geometry, known_gradient


def _schedule(params: dict, name: str, iteration: int, default_decay: float = 1.0) -> float:
    initial = float(params[name])
    decay = float(params.get(f"{name}_decay", default_decay))
    return initial * decay**int(iteration)


class VanillaPG:
    """Classical REINFORCE using only the known gradient component g_K."""

    name = "VanillaPG"
    seed_alias = "VanillaPG"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)

    def run(self, problem, rng, context, cache):
        state, _ = restore_or_initialize(cache, rng, lambda: initialize_state(problem, context.evaluation_interval), context.evaluation_interval)
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            if not can_afford(state, count, context.max_trajectories):
                break
            trajectories = problem.sample_observations(state["theta"], count, rng)
            gradient = known_gradient(problem, state["theta"], trajectories)
            eta = _schedule(self.p, "step_size", iteration)
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
        state, _ = restore_or_initialize(cache, rng, lambda: initialize_state(problem, context.evaluation_interval), context.evaluation_interval)
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            cost = 2 * count
            if not can_afford(state, cost, context.max_trajectories):
                break
            mu = _schedule(self.p, "radius", iteration) / np.sqrt(problem.n)
            eta = _schedule(self.p, "step_size", iteration)
            direction = rng.normal(size=problem.n)
            plus, _ = problem.sample_losses(
                state["theta"] + mu * direction, count, rng
            )
            minus, _ = problem.sample_losses(
                state["theta"] - mu * direction, count, rng
            )
            gradient = float((plus.mean() - minus.mean()) / (2.0 * mu)) * direction
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
        state, _ = restore_or_initialize(cache, rng, lambda: initialize_state(problem, context.evaluation_interval), context.evaluation_interval)
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            if not can_afford(state, count, context.max_trajectories):
                break
            mu = _schedule(self.p, "radius", iteration) / np.sqrt(problem.n)
            eta = _schedule(self.p, "step_size", iteration)
            direction = rng.normal(size=problem.n)
            losses, _ = problem.sample_losses(
                state["theta"] + mu * direction, count, rng
            )
            gradient = float(losses.mean() / mu) * direction
            state["theta"] = state["theta"] - eta * gradient
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
        state, _ = restore_or_initialize(cache, rng, lambda: initialize_state(problem, context.evaluation_interval), context.evaluation_interval)
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            cost = 3 * count
            if not can_afford(state, cost, context.max_trajectories):
                break
            observations = problem.sample_observations(state["theta"], count, rng)
            guide = known_gradient(problem, state["theta"], observations)
            norm = float(np.linalg.norm(guide))
            hint = guide / norm if norm > 0.0 else np.zeros(problem.n)
            alpha = float(self.p["alpha"])
            orthogonal = alpha / float(problem.n)
            direction = (
                np.sqrt(orthogonal) * rng.normal(size=problem.n)
                + np.sqrt(1.0 - alpha) * float(rng.normal()) * hint
            )
            mu = _schedule(self.p, "radius", iteration)
            eta = _schedule(self.p, "step_size", iteration)
            plus, _ = problem.sample_losses(
                state["theta"] + mu * direction, count, rng
            )
            minus, _ = problem.sample_losses(
                state["theta"] - mu * direction, count, rng
            )
            gradient = float((plus.mean() - minus.mean()) / (2.0 * mu)) * direction
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

        state, _ = restore_or_initialize(cache, rng, initialize, context.evaluation_interval)
        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            cost = 2 * count
            if not can_afford(state, cost, context.max_trajectories):
                break
            guide = np.asarray(state["guide"], dtype=float)
            norm = float(np.linalg.norm(guide))
            hint = guide / norm if norm > 0.0 else np.zeros(problem.n)
            alpha = float(self.p["alpha"])
            orthogonal = alpha / float(problem.n)
            direction = (
                np.sqrt(orthogonal) * rng.normal(size=problem.n)
                + np.sqrt(1.0 - alpha) * float(rng.normal()) * hint
            )
            mu = _schedule(self.p, "radius", iteration)
            eta = _schedule(self.p, "step_size", iteration)
            plus_theta = state["theta"] + mu * direction
            minus_theta = state["theta"] - mu * direction
            plus, plus_trajectories = problem.sample_losses(plus_theta, count, rng)
            minus, minus_trajectories = problem.sample_losses(minus_theta, count, rng)
            gradient = float((plus.mean() - minus.mean()) / (2.0 * mu)) * direction
            state["history"].append(
                (plus_theta.copy(), plus_trajectories, minus_theta.copy(), minus_trajectories)
            )
            window = int(self.p["window"])
            if len(state["history"]) > window:
                state["history"] = state["history"][-window:]
            state["theta"] = state["theta"] - eta * gradient
            state["sample_count"] += cost

            active = state["history"]
            gradients = []
            for p_theta, p_traj, m_theta, m_traj in active:
                gradients.append(known_gradient(problem, p_theta, p_traj))
                gradients.append(known_gradient(problem, m_theta, m_traj))
            state["guide"] = np.mean(gradients, axis=0) if gradients else np.zeros(problem.n)
            state["iteration"] = iteration + 1
            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)
        return finish(state, problem, context.evaluation_interval)
