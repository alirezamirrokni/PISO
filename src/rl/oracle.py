import numpy as np


def follower_q_jacobian(problem, policy, policy_jacobian, grid, tolerance=1e-5, tie_tolerance=1e-10):
    env = problem.env
    q_value = env.best_response_q(policy, grid, tol=tolerance)
    value = np.max(q_value, axis=1)
    value[env.terminal] = 0.0
    active = (q_value >= np.max(q_value, axis=1, keepdims=True) - tie_tolerance).astype(np.float64)
    active /= np.sum(active, axis=1, keepdims=True)
    rewards = env.follower_rewards(grid)
    transition = np.zeros((env.dim, env.dim), dtype=np.float64)
    rhs = np.zeros((env.dim, problem.n), dtype=np.float64)
    for state in range(env.dim):
        if state == env.terminal:
            continue
        for follower_action in range(q_value.shape[1]):
            weight = float(active[state, follower_action])
            if weight == 0.0:
                continue
            next_states = env.joint_next[state, :, follower_action]
            action_values = rewards[state, :, follower_action] + float(env.gamma) * value[next_states]
            rhs[state] += weight * np.einsum("ad,a->d", policy_jacobian[state], action_values, optimize=True)
            for action in range(policy.shape[1]):
                transition[state, int(next_states[action])] += weight * float(env.gamma) * policy[state, action]
    value_jacobian = np.linalg.solve(np.eye(env.dim, dtype=np.float64) - transition, rhs)
    q_jacobian = np.zeros((env.dim, q_value.shape[1], problem.n), dtype=np.float64)
    for state in range(env.dim):
        if state == env.terminal:
            continue
        for follower_action in range(q_value.shape[1]):
            next_states = env.joint_next[state, :, follower_action]
            action_values = rewards[state, :, follower_action] + float(env.gamma) * value[next_states]
            direct = np.einsum("ad,a->d", policy_jacobian[state], action_values, optimize=True)
            continuation = float(env.gamma) * np.einsum("a,ad->d", policy[state], value_jacobian[next_states], optimize=True)
            q_jacobian[state, follower_action] = direct + continuation
    return q_value, q_jacobian


def oracle_model(problem, theta):
    env = problem.env
    policy = np.asarray(problem.policy(theta), dtype=np.float64)
    expected_features = np.einsum("sa,sad->sd", policy, problem.phi, optimize=True)
    policy_jacobian = policy[:, :, None] * (problem.phi - expected_features[:, None, :])
    pairs = [follower_q_jacobian(problem, policy, policy_jacobian, grid) for grid in env.perturbed_grids]
    average_q = np.mean([q for q, _ in pairs], axis=0)
    average_q_jacobian = np.mean([j for _, j in pairs], axis=0)
    logits = float(env.beta) * average_q
    logits -= np.max(logits, axis=1, keepdims=True)
    follower = np.exp(logits)
    follower /= np.sum(follower, axis=1, keepdims=True)
    expected_q_jacobian = np.einsum("sf,sfd->sd", follower, average_q_jacobian, optimize=True)
    follower_jacobian = float(env.beta) * follower[:, :, None] * (average_q_jacobian - expected_q_jacobian[:, None, :])
    rewards = np.einsum("sf,saf->sa", follower, env.principal_rewards, optimize=True)
    reward_jacobian = np.einsum("sfd,saf->sad", follower_jacobian, env.principal_rewards, optimize=True)
    transitions = np.zeros((env.dim, 4, env.dim), dtype=np.float64)
    transition_jacobian = np.zeros((env.dim, 4, env.dim, problem.n), dtype=np.float64)
    states = np.arange(env.dim, dtype=np.int64)
    for action in env.moves:
        for follower_action in env.interventions:
            next_states = env.joint_next[:, action, follower_action]
            transitions[states, action, next_states] += follower[:, follower_action]
            transition_jacobian[states, action, next_states, :] += follower_jacobian[:, follower_action, :]
    value = np.linalg.solve(np.eye(env.dim, dtype=np.float64) - float(problem.discount) * np.einsum("sa,san->sn", policy, transitions, optimize=True),
                            np.einsum("sa,sa->s", policy, rewards, optimize=True))
    q_value = rewards + float(problem.discount) * np.einsum("san,n->sa", transitions, value, optimize=True)
    return {"advantage": q_value - value[:, None], "policy_score": problem.phi - expected_features[:, None, :],
            "transitions": transitions, "transition_jacobian": transition_jacobian,
            "reward_jacobian": reward_jacobian, "value": value}
