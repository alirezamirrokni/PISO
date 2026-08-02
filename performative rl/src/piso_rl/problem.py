from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.envs.gridworld import Gridworld
from src.policies.tabular import Tabular


Trajectory = list[tuple[int, int, int, float]]


@dataclass(frozen=True)
class ProblemMetrics:
    performative_value: float
    mean_episode_length: float | None = None
    success_rate: float | None = None
    hazard_rate: float | None = None
    intervention_rate: float | None = None


class PerformativeRLProblem:
    """Expose the PePG gridworld as a trajectory-based optimization problem.

    The optimizer minimizes ``F(theta) = -J(theta)`` while reports display the
    positive performative return ``J(theta)``.  Sampling deployment is kept
    separate from exact occupancy evaluation so stochastic-gradient iterations
    do not repeatedly construct reward/transition tensors they never use.
    """

    def __init__(
        self,
        *,
        beta: float,
        eps: float,
        discount: float,
        num_followers: int,
        seed: int,
        feature_dim: int = 32,
        theta_std: float = 0.10,
        phi_std: float = 0.20,
        horizon: int = 100,
    ) -> None:
        if not 0.0 < discount < 1.0:
            raise ValueError("discount must lie in (0, 1)")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_followers <= 0:
            raise ValueError("num_followers must be positive")

        self.seed = int(seed)
        self.horizon = int(horizon)
        self.feature_dim = int(feature_dim)
        self.n = self.feature_dim

        self.env = Gridworld(
            beta=float(beta),
            eps=float(eps),
            gamma=float(discount),
            num_followers=int(num_followers),
            sampling=False,
            n_sample=1,
            seed=self.seed,
            max_sample_steps=self.horizon,
        )

        init_rng = np.random.RandomState(self.seed)
        self.initial_x = init_rng.normal(
            loc=0.0,
            scale=float(theta_std),
            size=self.feature_dim,
        )
        self.phi = init_rng.normal(
            loc=0.0,
            scale=float(phi_std),
            size=(self.env.dim, len(self.env.agents[1].actions), self.feature_dim),
        )

        self._policy_theta: np.ndarray | None = None
        self._policy_probabilities: np.ndarray | None = None
        self._deployed_theta: np.ndarray | None = None
        self._deployed_principal: np.ndarray | None = None
        self._deployed_follower: np.ndarray | None = None
        self._principal_actions = np.asarray(self.env.agents[1].actions, dtype=int)
        self._follower_actions = np.asarray(self.env.agents[2].actions, dtype=int)
        self._states = np.arange(self.env.dim, dtype=int)

    @property
    def discount(self) -> float:
        return float(self.env.gamma)

    def signature(self) -> dict:
        return {
            "seed": self.seed,
            "beta": float(self.env.beta),
            "eps": float(self.env.eps),
            "discount": self.discount,
            "num_followers": int(self.env.num_followers),
            "feature_dim": self.feature_dim,
            "horizon": self.horizon,
            "initial_x": np.asarray(self.initial_x).round(12).tolist(),
            "phi_checksum": float(np.sum(self.phi)),
        }

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        array = np.asarray(theta, dtype=float)
        if array.shape != (self.feature_dim,):
            raise ValueError(
                f"theta must have shape ({self.feature_dim},), got {array.shape}"
            )
        return array

    @staticmethod
    def _same_theta(left: np.ndarray | None, right: np.ndarray) -> bool:
        return left is not None and np.array_equal(left, right)

    def policy(self, theta: np.ndarray) -> np.ndarray:
        theta = self._validate_theta(theta)
        if self._same_theta(self._policy_theta, theta):
            assert self._policy_probabilities is not None
            return self._policy_probabilities

        logits = np.einsum("sad,d->sa", self.phi, theta, optimize=True)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        self._policy_theta = theta.copy()
        self._policy_probabilities = probabilities
        return probabilities

    def deploy_sampling(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deploy only the principal and follower policies needed for rollouts."""
        theta = self._validate_theta(theta)
        if self._same_theta(self._deployed_theta, theta):
            assert self._deployed_principal is not None
            assert self._deployed_follower is not None
            return self._deployed_principal, self._deployed_follower

        principal = self.env.agents[1]
        follower = self.env.agents[2]
        principal_probabilities = self.policy(theta)
        principal.policy = Tabular(principal.actions, principal_probabilities)
        follower.policy = self.env.response_model(self.env.agents)
        follower_probabilities = self.env._policy_matrix(follower)

        self._deployed_theta = theta.copy()
        self._deployed_principal = principal_probabilities
        self._deployed_follower = follower_probabilities
        return principal_probabilities, follower_probabilities

    def deploy_exact(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deploy policies and construct exact rewards, transitions, occupancy."""
        probabilities, _ = self.deploy_sampling(theta)
        principal = self.env.agents[1]
        rewards, transitions = self.env._get_exact_RT()
        occupancy = self.env._get_d(transitions, principal)
        return probabilities, rewards, transitions, occupancy

    # Backward-compatible alias used by the PePG adapter and older caches.
    def deploy(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.deploy_exact(theta)

    def exact_value(self, theta: np.ndarray) -> float:
        _, rewards, _, occupancy = self.deploy_exact(theta)
        return float(np.sum(occupancy * rewards))

    def exact_metrics(self, theta: np.ndarray) -> ProblemMetrics:
        return ProblemMetrics(performative_value=self.exact_value(theta))

    def evaluate(
        self,
        theta: np.ndarray,
        metric_samples: int,
        rng: np.random.RandomState,
    ) -> float:
        del metric_samples, rng
        return -self.exact_value(theta)

    def sample_trajectory(
        self,
        theta: np.ndarray,
        rng: np.random.RandomState,
        *,
        already_deployed: bool = False,
    ) -> Trajectory:
        if already_deployed:
            principal_probabilities = self._deployed_principal
            follower_probabilities = self._deployed_follower
            if principal_probabilities is None or follower_probabilities is None:
                raise RuntimeError("already_deployed=True without a deployed policy")
        else:
            principal_probabilities, follower_probabilities = self.deploy_sampling(theta)

        state = int(rng.choice(self._states, p=self.env.rho))
        trajectory: Trajectory = []
        terminal = int(self.env.terminal_state_id)
        joint_next = self.env._joint_next
        joint_rewards = self.env._principal_joint_rewards

        for _ in range(self.horizon):
            if state == terminal:
                break
            principal_action = int(
                rng.choice(
                    self._principal_actions,
                    p=principal_probabilities[state],
                )
            )
            follower_action = int(
                rng.choice(
                    self._follower_actions,
                    p=follower_probabilities[state],
                )
            )
            next_state = int(joint_next[state, principal_action, follower_action])
            reward = float(joint_rewards[state, principal_action, follower_action])
            trajectory.append((state, principal_action, next_state, reward))
            state = next_state

        return trajectory

    def sample_trajectories(
        self,
        theta: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> list[Trajectory]:
        count = int(count)
        if count <= 0:
            raise ValueError("trajectory count must be positive")
        self.deploy_sampling(theta)
        return [
            self.sample_trajectory(theta, rng, already_deployed=True)
            for _ in range(count)
        ]

    def discounted_return(self, trajectory: Trajectory) -> float:
        value = 0.0
        discount = 1.0
        for _, _, _, reward in trajectory:
            value += discount * reward
            discount *= self.discount
        return float(value)

    def sample_losses(
        self,
        theta: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, list[Trajectory]]:
        trajectories = self.sample_trajectories(theta, count, rng)
        returns = np.fromiter(
            (self.discounted_return(trajectory) for trajectory in trajectories),
            dtype=float,
            count=len(trajectories),
        )
        return -returns, trajectories

    def sample_observations(
        self,
        theta: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> list[Trajectory]:
        return self.sample_trajectories(theta, count, rng)

    def _reward_to_go(self, trajectory: Trajectory) -> np.ndarray:
        returns = np.zeros(len(trajectory), dtype=float)
        running = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            running = trajectory[index][3] + self.discount * running
            returns[index] = running
        return returns

    def _partial_gradient_with_policy(
        self,
        trajectory: Trajectory,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        reward_to_go = self._reward_to_go(trajectory)
        gradient = np.zeros(self.feature_dim, dtype=float)

        for time_index, (state, action, _, _) in enumerate(trajectory):
            expected_feature = probabilities[state] @ self.phi[state]
            score = self.phi[state, action] - expected_feature
            gradient += (
                self.discount**time_index * reward_to_go[time_index] * score
            )
        return -gradient

    def partial_gradient_for_trajectory(
        self,
        theta: np.ndarray,
        trajectory: Trajectory,
    ) -> np.ndarray:
        probabilities = self.policy(theta)
        return self._partial_gradient_with_policy(trajectory, probabilities)

    def partial_gradients(
        self,
        theta: np.ndarray,
        trajectories: Iterable[Trajectory],
    ) -> np.ndarray:
        trajectories = list(trajectories)
        if not trajectories:
            raise ValueError("at least one trajectory is required")

        # Compute policy probabilities exactly once for the entire gradient batch.
        probabilities = self.policy(theta)
        return np.stack(
            [
                self._partial_gradient_with_policy(trajectory, probabilities)
                for trajectory in trajectories
            ],
            axis=0,
        )

    def rollout_metrics(
        self,
        theta: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> ProblemMetrics:
        trajectories = self.sample_trajectories(theta, count, rng)
        returns = [self.discounted_return(t) for t in trajectories]
        lengths = [len(t) for t in trajectories]
        terminal = self.env.terminal_state_id
        success = [bool(t and t[-1][2] == terminal) for t in trajectories]
        hazard = [
            sum(1 for _, _, _, reward in t if reward == -0.5) / max(len(t), 1)
            for t in trajectories
        ]
        return ProblemMetrics(
            performative_value=float(np.mean(returns)),
            mean_episode_length=float(np.mean(lengths)),
            success_rate=float(np.mean(success)),
            hazard_rate=float(np.mean(hazard)),
            intervention_rate=None,
        )
