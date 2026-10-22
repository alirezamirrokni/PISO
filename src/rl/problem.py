import numpy as np

from .gridworld import Gridworld


class PerformativeRL:
    def __init__(self, environment, policy, seed):
        self.env = Gridworld(float(environment["beta"]), float(environment["eps"]), float(environment["discount"]),
                             int(environment["num_followers"]))
        self.discount, self.horizon = float(environment["discount"]), int(environment["horizon"])
        self.n = int(policy["feature_dim"])
        init_rng = np.random.RandomState(int(seed))
        self.initial_x = init_rng.normal(loc=0.0, scale=float(policy["theta_std"]), size=self.n)
        self.phi = init_rng.normal(loc=0.0, scale=float(policy["phi_std"]), size=(self.env.dim, 4, self.n))
        self.states = np.arange(self.env.dim)
        self._policy = (None, None)
        self._deployed = (None, None, None)

    def policy(self, theta):
        if self._policy[0] is None or not np.array_equal(self._policy[0], theta):
            logits = np.einsum("sad,d->sa", self.phi, theta, optimize=True)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            self._policy = (np.array(theta, dtype=float), probabilities)
        return self._policy[1]

    def follower_policy(self, principal):
        principal = principal / principal.sum(axis=1, keepdims=True)
        average_q = np.mean([self.env.best_response_q(principal, grid) for grid in self.env.perturbed_grids], axis=0)
        logits = float(self.env.beta) * average_q
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def deploy(self, theta):
        if self._deployed[0] is None or not np.array_equal(self._deployed[0], theta):
            principal = np.asarray(self.policy(theta), dtype=float).copy()
            self._deployed = (np.array(theta, dtype=float), principal, self.follower_policy(self.policy(theta)))
        return self._deployed[1], self._deployed[2]

    def model(self, actor, follower):
        env = self.env
        rewards = np.einsum("sf,saf->sa", follower, env.principal_rewards, optimize=True)
        transitions = np.zeros((env.dim, 4, env.dim), dtype=float)
        for action in env.moves:
            for follower_action in env.interventions:
                transitions[self.states, action, env.joint_next[:, action, follower_action]] += follower[:, follower_action]
        nonterminal = np.asarray([s for s in range(env.dim) if s != env.terminal], dtype=int)
        restricted = np.einsum("sa,san->sn", actor, transitions, optimize=True)[np.ix_(nonterminal, nonterminal)]
        mass = np.linalg.solve(np.eye(nonterminal.size) - self.discount * restricted.T, env.rho[nonterminal])
        occupancy = np.zeros_like(actor, dtype=float)
        occupancy[nonterminal] = mass[:, None] * actor[nonterminal]
        return rewards, transitions, occupancy

    def evaluate(self, theta):
        rewards, _, occupancy = self.model(*self.deploy(theta))
        return float(np.sum(occupancy * rewards)), occupancy

    def trajectory(self, principal, follower, rng):
        env = self.env
        state, path = int(rng.choice(self.states, p=env.rho)), []
        for _ in range(self.horizon):
            if state == env.terminal:
                break
            action = int(rng.choice(env.moves, p=principal[state]))
            follower_action = int(rng.choice(env.interventions, p=follower[state]))
            next_state = int(env.joint_next[state, action, follower_action])
            path.append((state, action, next_state, float(env.principal_rewards[state, action, follower_action])))
            state = next_state
        return path

    def sample_trajectories(self, theta, count, rng):
        principal, follower = self.deploy(theta)
        return [self.trajectory(principal, follower, rng) for _ in range(int(count))]

    def discounted_return(self, path):
        value, discount = 0.0, 1.0
        for _, _, _, reward in path:
            value += discount * reward
            discount *= self.discount
        return float(value)

    def sample_losses(self, theta, count, rng):
        paths = self.sample_trajectories(theta, count, rng)
        return -np.fromiter((self.discounted_return(path) for path in paths), dtype=float, count=len(paths)), paths

    def sample_paired_losses(self, theta_plus, theta_minus, count, rng):
        seeds = rng.randint(0, 2 ** 31 - 1, size=int(count), dtype=np.int64)
        plus_policies = tuple(np.asarray(p, dtype=float).copy() for p in self.deploy(theta_plus))
        minus_policies = tuple(np.asarray(p, dtype=float).copy() for p in self.deploy(theta_minus))
        plus = [self.trajectory(*plus_policies, np.random.RandomState(int(s))) for s in seeds]
        minus = [self.trajectory(*minus_policies, np.random.RandomState(int(s))) for s in seeds]
        returns = lambda paths: -np.fromiter((self.discounted_return(p) for p in paths), dtype=float, count=len(paths))
        return returns(plus), plus, returns(minus), minus

    def known_gradient(self, theta, paths):
        probabilities = self.policy(theta)
        gradients = []
        for path in paths:
            to_go, running = np.zeros(len(path)), 0.0
            for index in range(len(path) - 1, -1, -1):
                running = path[index][3] + self.discount * running
                to_go[index] = running
            gradient = np.zeros(self.n, dtype=float)
            for t, (state, action, _, _) in enumerate(path):
                gradient += self.discount ** t * to_go[t] * (self.phi[state, action] - probabilities[state] @ self.phi[state])
            gradients.append(-gradient)
        return np.stack(gradients, axis=0).mean(axis=0)
