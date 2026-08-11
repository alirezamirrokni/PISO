from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleModel:








    policy: np.ndarray
    follower_policy: np.ndarray
    rewards: np.ndarray
    transitions: np.ndarray
    reward_jacobian: np.ndarray
    transition_jacobian: np.ndarray
    value: np.ndarray
    q_value: np.ndarray
    advantage: np.ndarray
    policy_score: np.ndarray
    minimum_follower_action_gap: float


def _principal_policy_and_jacobian(problem, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    policy = np.asarray(problem.policy(theta), dtype=np.float64)
    expected_features = np.einsum(
        "sa,sad->sd", policy, problem.phi, optimize=True
    )
    jacobian = policy[:, :, None] * (
        problem.phi - expected_features[:, None, :]
    )
    return policy, jacobian


def _follower_q_jacobian(
    problem,
    principal_policy: np.ndarray,
    principal_policy_jacobian: np.ndarray,
    perturbed_grid: np.ndarray,
    *,
    response_tolerance: float,
    tie_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:








    env = problem.env
    q_value = env._best_response_q_vectorized(
        principal_policy,
        perturbed_grid,
        tol=float(response_tolerance),
    )
    value = np.max(q_value, axis=1)
    terminal = int(env.terminal_state_id)
    value[terminal] = 0.0
    maximum_q = np.max(q_value, axis=1, keepdims=True)
    active_actions = q_value >= (maximum_q - float(tie_tolerance))
    active_weights = active_actions.astype(np.float64)
    active_weights /= np.sum(active_weights, axis=1, keepdims=True)

    sorted_q = np.sort(q_value, axis=1)
    gaps = sorted_q[:, -1] - sorted_q[:, -2]
    nonterminal_gaps = np.delete(gaps, terminal)
    minimum_gap = float(np.min(nonterminal_gaps)) if nonterminal_gaps.size else float("inf")

    follower_rewards = env._follower_reward_table(perturbed_grid)
    num_states = int(env.dim)
    num_actions = principal_policy.shape[1]
    num_follower_actions = q_value.shape[1]
    parameter_dim = int(problem.n)

    transition_matrix = np.zeros((num_states, num_states), dtype=np.float64)
    rhs = np.zeros((num_states, parameter_dim), dtype=np.float64)

    for state in range(num_states):
        if state == terminal:
            continue
        for follower_action in range(num_follower_actions):
            branch_weight = float(active_weights[state, follower_action])
            if branch_weight == 0.0:
                continue
            next_states = env._joint_next[state, :, follower_action]
            action_values = (
                follower_rewards[state, :, follower_action]
                + float(env.gamma) * value[next_states]
            )
            rhs[state] += branch_weight * np.einsum(
                "ad,a->d",
                principal_policy_jacobian[state],
                action_values,
                optimize=True,
            )
            for action in range(num_actions):
                transition_matrix[state, int(next_states[action])] += (
                    branch_weight
                    * float(env.gamma)
                    * principal_policy[state, action]
                )

    
    value_jacobian = np.linalg.solve(
        np.eye(num_states, dtype=np.float64) - transition_matrix,
        rhs,
    )

    q_jacobian = np.zeros(
        (num_states, num_follower_actions, parameter_dim),
        dtype=np.float64,
    )
    for state in range(num_states):
        if state == terminal:
            continue
        for follower_action in range(num_follower_actions):
            next_states = env._joint_next[state, :, follower_action]
            action_values = (
                follower_rewards[state, :, follower_action]
                + float(env.gamma) * value[next_states]
            )
            direct = np.einsum(
                "ad,a->d",
                principal_policy_jacobian[state],
                action_values,
                optimize=True,
            )
            continuation = float(env.gamma) * np.einsum(
                "a,ad->d",
                principal_policy[state],
                value_jacobian[next_states],
                optimize=True,
            )
            q_jacobian[state, follower_action] = direct + continuation

    return q_value, q_jacobian, minimum_gap


def build_oracle_model(
    problem,
    theta: np.ndarray,
    *,
    response_tolerance: float = 1e-5,
    tie_tolerance: float = 1e-10,
) -> OracleModel:


    theta = problem._validate_theta(theta)
    env = problem.env
    principal_policy, principal_policy_jacobian = _principal_policy_and_jacobian(
        problem, theta
    )

    follower_q_values: list[np.ndarray] = []
    follower_q_jacobians: list[np.ndarray] = []
    action_gaps: list[float] = []
    for perturbed_grid in env.pertrubed_grids:
        q_value, q_jacobian, action_gap = _follower_q_jacobian(
            problem,
            principal_policy,
            principal_policy_jacobian,
            perturbed_grid,
            response_tolerance=response_tolerance,
            tie_tolerance=tie_tolerance,
        )
        follower_q_values.append(q_value)
        follower_q_jacobians.append(q_jacobian)
        action_gaps.append(action_gap)

    average_q = np.mean(follower_q_values, axis=0)
    average_q_jacobian = np.mean(follower_q_jacobians, axis=0)

    logits = float(env.beta) * average_q
    logits -= np.max(logits, axis=1, keepdims=True)
    follower_policy = np.exp(logits)
    follower_policy /= np.sum(follower_policy, axis=1, keepdims=True)

    expected_q_jacobian = np.einsum(
        "sf,sfd->sd",
        follower_policy,
        average_q_jacobian,
        optimize=True,
    )
    follower_policy_jacobian = (
        float(env.beta)
        * follower_policy[:, :, None]
        * (average_q_jacobian - expected_q_jacobian[:, None, :])
    )

    rewards = np.einsum(
        "sf,saf->sa",
        follower_policy,
        env._principal_joint_rewards,
        optimize=True,
    )
    reward_jacobian = np.einsum(
        "sfd,saf->sad",
        follower_policy_jacobian,
        env._principal_joint_rewards,
        optimize=True,
    )

    num_states = int(env.dim)
    num_actions = len(env.moves)
    parameter_dim = int(problem.n)
    transitions = np.zeros(
        (num_states, num_actions, num_states), dtype=np.float64
    )
    transition_jacobian = np.zeros(
        (num_states, num_actions, num_states, parameter_dim),
        dtype=np.float64,
    )
    states = np.arange(num_states, dtype=np.int64)
    for action in env.moves:
        for follower_action in env.interventions:
            next_states = env._joint_next[:, action, follower_action]
            transitions[states, action, next_states] += follower_policy[:, follower_action]
            transition_jacobian[states, action, next_states, :] += (
                follower_policy_jacobian[:, follower_action, :]
            )

    policy_transition = np.einsum(
        "sa,san->sn", principal_policy, transitions, optimize=True
    )
    policy_reward = np.einsum(
        "sa,sa->s", principal_policy, rewards, optimize=True
    )
    value = np.linalg.solve(
        np.eye(num_states, dtype=np.float64)
        - float(problem.discount) * policy_transition,
        policy_reward,
    )
    q_value = rewards + float(problem.discount) * np.einsum(
        "san,n->sa", transitions, value, optimize=True
    )
    advantage = q_value - value[:, None]

    expected_features = np.einsum(
        "sa,sad->sd", principal_policy, problem.phi, optimize=True
    )
    policy_score = problem.phi - expected_features[:, None, :]

    return OracleModel(
        policy=principal_policy,
        follower_policy=follower_policy,
        rewards=rewards,
        transitions=transitions,
        reward_jacobian=reward_jacobian,
        transition_jacobian=transition_jacobian,
        value=value,
        q_value=q_value,
        advantage=advantage,
        policy_score=policy_score,
        minimum_follower_action_gap=float(min(action_gaps)),
    )


def exact_oracle_gradient(
    problem,
    theta: np.ndarray,
    *,
    response_tolerance: float = 1e-5,
    tie_tolerance: float = 1e-10,
) -> np.ndarray:







    model = build_oracle_model(
        problem,
        theta,
        response_tolerance=response_tolerance,
        tie_tolerance=tie_tolerance,
    )
    policy_transition = np.einsum(
        "sa,san->sn", model.policy, model.transitions, optimize=True
    )
    state_occupancy = np.linalg.solve(
        np.eye(problem.env.dim, dtype=np.float64)
        - float(problem.discount) * policy_transition.T,
        np.asarray(problem.env.rho, dtype=np.float64),
    )
    occupancy = state_occupancy[:, None] * model.policy

    policy_term = np.einsum(
        "sa,sa,sad->d",
        occupancy,
        model.q_value,
        model.policy_score,
        optimize=True,
    )
    reward_term = np.einsum(
        "sa,sad->d",
        occupancy,
        model.reward_jacobian,
        optimize=True,
    )
    transition_term = float(problem.discount) * np.einsum(
        "sa,sand,n->d",
        occupancy,
        model.transition_jacobian,
        model.value,
        optimize=True,
    )
    return policy_term + reward_term + transition_term
