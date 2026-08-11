from __future__ import annotations

from typing import Any

import numpy as np

from ..oracle_model import build_oracle_model
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


class PePG:
















    name = "PePG"
    seed_alias = "PePG"

    def __init__(self, params: dict[str, Any]) -> None:
        self.p = dict(params)
        if int(self.p.get("batch_initial", 100)) <= 0:
            raise ValueError("PePG batch_initial must be positive")
        for key, default in (
            ("step_size", 0.5),
            ("transition_probability_floor", 1e-12),
        ):
            value = float(self.p.get(key, default))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"PePG {key} must be finite and positive")

    def _step_size(self, iteration: int, sample_count: int) -> float:
        return scheduled_value(
            self.p,
            "step_size",
            iteration=iteration,
            sample_count=sample_count,
            legacy_decay_key="step_size_decay",
        )

    @staticmethod
    def _gradient_estimate(problem, model, trajectories, probability_floor: float):
        policy_gradient = np.zeros(problem.n, dtype=np.float64)
        transition_gradient = np.zeros(problem.n, dtype=np.float64)
        reward_gradient = np.zeros(problem.n, dtype=np.float64)
        transition_count = 0

        for trajectory in trajectories:
            for time_index, (state_id, action, next_state, _) in enumerate(
                trajectory
            ):
                discount = float(problem.discount) ** time_index
                advantage = float(model.advantage[state_id, action])
                policy_gradient += (
                    discount
                    * advantage
                    * model.policy_score[state_id, action]
                )

                probability = max(
                    float(model.transitions[state_id, action, next_state]),
                    float(probability_floor),
                )
                transition_score = (
                    model.transition_jacobian[state_id, action, next_state]
                    / probability
                )
                transition_gradient += (
                    discount
                    * float(problem.discount)
                    * float(model.value[next_state])
                    * transition_score
                )
                reward_gradient += discount * model.reward_jacobian[state_id, action]
                transition_count += 1

        scale = 1.0 / float(len(trajectories))
        policy_gradient *= scale
        transition_gradient *= scale
        reward_gradient *= scale
        return (
            policy_gradient + transition_gradient + reward_gradient,
            policy_gradient,
            transition_gradient,
            reward_gradient,
            transition_count,
        )

    def run(self, problem, rng, context, cache):
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initialize_state(problem, context.evaluation_interval),
            context.evaluation_interval,
        )
        state.setdefault("oracle_model_calls", 0)
        state.setdefault("environment_transitions", 0)
        state.setdefault("last_gradient_norm", 0.0)
        state.setdefault("last_policy_gradient_norm", 0.0)
        state.setdefault("last_transition_gradient_norm", 0.0)
        state.setdefault("last_reward_gradient_norm", 0.0)

        probability_floor = float(
            self.p.get("transition_probability_floor", 1e-12)
        )

        while True:
            iteration = int(state["iteration"])
            count = batch_size(self.p, iteration)
            if not can_afford(state, count, context.max_trajectories):
                break

            theta = np.asarray(state["theta"], dtype=np.float64)
            trajectories = problem.sample_trajectories(theta, count, rng)
            model = build_oracle_model(problem, theta)

            (
                value_gradient,
                policy_gradient,
                transition_gradient,
                reward_gradient,
                transition_count,
            ) = self._gradient_estimate(
                problem,
                model,
                trajectories,
                probability_floor,
            )
            value_gradient = clip_gradient(
                value_gradient, self.p.get("gradient_clip_norm")
            )
            eta = self._step_size(iteration, int(state["sample_count"]))

            state["theta"] = theta + eta * value_gradient
            state["sample_count"] = int(state["sample_count"]) + count
            state["iteration"] = iteration + 1
            state["oracle_model_calls"] = int(state["oracle_model_calls"]) + 1
            state["environment_transitions"] = (
                int(state["environment_transitions"]) + transition_count
            )
            state["last_gradient_norm"] = float(np.linalg.norm(value_gradient))
            state["last_policy_gradient_norm"] = float(
                np.linalg.norm(policy_gradient)
            )
            state["last_transition_gradient_norm"] = float(
                np.linalg.norm(transition_gradient)
            )
            state["last_reward_gradient_norm"] = float(
                np.linalg.norm(reward_gradient)
            )

            record(state, problem, context.evaluation_interval)
            save_step(cache, state, rng)

        trace = finish(state, problem, context.evaluation_interval)
        trace.oracle_model_calls = int(state["oracle_model_calls"])
        trace.environment_transitions = int(state["environment_transitions"])
        trace.last_gradient_norm = float(state["last_gradient_norm"])
        trace.last_policy_gradient_norm = float(
            state["last_policy_gradient_norm"]
        )
        trace.last_transition_gradient_norm = float(
            state["last_transition_gradient_norm"]
        )
        trace.last_reward_gradient_norm = float(
            state["last_reward_gradient_norm"]
        )
        return trace
