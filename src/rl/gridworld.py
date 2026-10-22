import itertools

import numpy as np

H, F = -0.5, -0.02
GRID = np.array([
    [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
    [-0.01, -0.01, F, -0.01, H, -0.01, -0.01, -0.01],
    [-0.01, -0.01, -0.01, H, -0.01, -0.01, F, -0.01],
    [-0.01, F, -0.01, -0.01, -0.01, H, -0.01, F],
    [-0.01, -0.01, -0.01, H, -0.01, -0.01, F, -0.01],
    [-0.01, H, H, -0.01, F, -0.01, H, -0.01],
    [-0.01, H, -0.01, -0.01, H, -0.01, H, -0.01],
    [-0.01, -0.01, -0.01, H, -0.01, F, -0.01, +1],
])
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]
INTERVENTION_COST = -0.05


class Gridworld:
    def __init__(self, beta, eps, gamma, num_followers):
        self.beta, self.eps, self.gamma = beta, eps, gamma
        self.size = GRID.shape[0]
        self.dim = GRID.size
        self.terminal = self.dim - 1
        self.moves, self.interventions = list(range(4)), list(range(5))
        self.initial_states = [s for s in range(self.dim) if s // self.size == 0 or s % self.size == 0]
        self.rho = np.zeros(self.dim)
        self.rho[self.initial_states] = 1 / len(self.initial_states)
        self.perturbed_grids = [self.perturb(np.random.default_rng(seed)) for seed in range(num_followers)]
        move_next = np.array([[self.next_state(s, move) for move in self.moves] for s in range(self.dim)], dtype=np.int64)
        self.joint_next = np.empty((self.dim, 4, 5), dtype=np.int64)
        self.joint_next[:, :, 4] = move_next
        for follower_action in range(4):
            self.joint_next[:, :, follower_action] = move_next[:, follower_action, None]
        self.principal_rewards = GRID.reshape(-1)[self.joint_next].astype(np.float64)
        self.principal_rewards[self.terminal] = 0.0

    def next_state(self, state, move):
        if state == self.terminal:
            return state
        row, column = state // self.size + MOVES[move][0], state % self.size + MOVES[move][1]
        if not (0 <= row < self.size and 0 <= column < self.size):
            return state
        return row * self.size + column

    def perturb(self, rng):
        values = np.unique(GRID)
        values = np.delete(values, np.where(values == 1))
        grid = np.zeros(GRID.shape)
        draws = rng.random(GRID.shape)
        for i, j in itertools.product(range(self.size), range(self.size)):
            keep = i * self.size + j in self.initial_states or draws[i, j] > self.eps
            grid[i, j] = GRID[i, j] if keep else rng.choice(values)
        grid[-1, -1] = GRID[-1, -1]
        return grid

    def follower_rewards(self, perturbed_grid):
        rewards = np.asarray(perturbed_grid, dtype=np.float64).reshape(-1)[self.joint_next].copy()
        rewards[:, :, :4] += INTERVENTION_COST
        rewards[self.terminal] = 0.0
        return rewards

    def best_response_q(self, principal_probabilities, perturbed_grid, tol=1e-5):
        rewards = self.follower_rewards(perturbed_grid)
        values = np.zeros(self.dim, dtype=np.float64)
        while True:
            q_values = np.einsum("sa,saf->sf", principal_probabilities, rewards + float(self.gamma) * values[self.joint_next], optimize=True)
            q_values[self.terminal, :] = 0.0
            updated = np.max(q_values, axis=1)
            updated[self.terminal] = 0.0
            converged = float(np.max(np.abs(updated - values))) < float(tol)
            values = updated
            if converged:
                break
        q_values = np.einsum("sa,saf->sf", principal_probabilities, rewards + float(self.gamma) * values[self.joint_next], optimize=True)
        q_values[self.terminal, :] = 0.0
        return q_values
