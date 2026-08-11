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


    def validate_policy_probabilities(self, probabilities: np.ndarray) -> np.ndarray:





        array = np.asarray(probabilities, dtype=float)
        expected_shape = (self.env.dim, len(self.env.agents[1].actions))
        if array.shape != expected_shape:
            raise ValueError(
                f"policy probabilities must have shape {expected_shape}, "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("policy probabilities must be finite and nonnegative")
        row_sums = array.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("every policy row must have positive mass")
        return array / row_sums

    def follower_policy_from_probabilities(
        self, principal_probabilities: np.ndarray
    ) -> np.ndarray:

        inducing_probabilities = self.validate_policy_probabilities(
            principal_probabilities
        )
        q_values = [
            self.env._best_response_q_vectorized(
                inducing_probabilities, perturbed_grid
            )
            for perturbed_grid in self.env.pertrubed_grids
        ]
        average_q = np.mean(q_values, axis=0)
        logits = float(self.env.beta) * average_q
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return np.asarray(probabilities, dtype=float)

    def follower_policy(self, environment_theta: np.ndarray) -> np.ndarray:









        environment_theta = self._validate_theta(environment_theta)
        return self.follower_policy_from_probabilities(
            self.policy(environment_theta)
        )

    def cross_deployment(
        self,
        policy_theta: np.ndarray,
        environment_theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        actor = np.asarray(self.policy(policy_theta), dtype=float).copy()
        follower = self.follower_policy(environment_theta)
        return actor, follower

    def _model_from_probabilities(
        self,
        actor: np.ndarray,
        follower: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        actor = np.asarray(actor, dtype=float)
        follower = np.asarray(follower, dtype=float)
        rewards = np.einsum(
            "sf,saf->sa",
            follower,
            self.env._principal_joint_rewards,
            optimize=True,
        )
        transitions = np.zeros(
            (self.env.dim, len(self.env.moves), self.env.dim), dtype=float
        )
        states = np.arange(self.env.dim, dtype=int)
        for action in self.env.moves:
            for follower_action in self.env.interventions:
                next_states = self.env._joint_next[:, action, follower_action]
                transitions[states, action, next_states] += follower[:, follower_action]

        
        
        
        
        terminal = int(self.env.terminal_state_id)
        nonterminal = np.asarray(
            [state for state in range(self.env.dim) if state != terminal],
            dtype=int,
        )
        policy_transition = np.einsum(
            "sa,san->sn", actor, transitions, optimize=True
        )
        restricted = policy_transition[np.ix_(nonterminal, nonterminal)]
        state_mass = np.linalg.solve(
            np.eye(nonterminal.size, dtype=float)
            - self.discount * restricted.T,
            np.asarray(self.env.rho, dtype=float)[nonterminal],
        )
        occupancy = np.zeros_like(actor, dtype=float)
        occupancy[nonterminal] = state_mass[:, None] * actor[nonterminal]
        return rewards, transitions, occupancy

    def deploy_policy_exact(
        self, principal_probabilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        actor = self.validate_policy_probabilities(principal_probabilities)
        follower = self.follower_policy_from_probabilities(actor)
        rewards, transitions, occupancy = self._model_from_probabilities(
            actor, follower
        )
        return actor.copy(), rewards, transitions, occupancy

    def sample_policy_trajectories(
        self,
        principal_probabilities: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> list[Trajectory]:

        count = int(count)
        if count <= 0:
            raise ValueError("trajectory count must be positive")
        actor = self.validate_policy_probabilities(principal_probabilities)
        follower = self.follower_policy_from_probabilities(actor)
        return [
            self._sample_trajectory_from_probabilities(actor, follower, rng)
            for _ in range(count)
        ]

    def exact_cross_value(
        self,
        policy_theta: np.ndarray,
        environment_theta: np.ndarray,
    ) -> float:

        actor, follower = self.cross_deployment(policy_theta, environment_theta)
        rewards, _, occupancy = self._model_from_probabilities(actor, follower)
        return float(np.sum(occupancy * rewards))

    def deploy_sampling(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        theta = self._validate_theta(theta)
        if self._same_theta(self._deployed_theta, theta):
            assert self._deployed_principal is not None
            assert self._deployed_follower is not None
            return self._deployed_principal, self._deployed_follower

        principal_probabilities = np.asarray(self.policy(theta), dtype=float).copy()
        follower_probabilities = self.follower_policy(theta)
        self._deployed_theta = theta.copy()
        self._deployed_principal = principal_probabilities
        self._deployed_follower = follower_probabilities
        return principal_probabilities, follower_probabilities

    def deploy_exact(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        probabilities, follower = self.deploy_sampling(theta)
        rewards, transitions, occupancy = self._model_from_probabilities(
            probabilities, follower
        )
        return probabilities, rewards, transitions, occupancy

    
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

    def _sample_trajectory_from_probabilities(
        self,
        principal_probabilities: np.ndarray,
        follower_probabilities: np.ndarray,
        rng: np.random.RandomState,
    ) -> Trajectory:
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
        return self._sample_trajectory_from_probabilities(
            principal_probabilities, follower_probabilities, rng
        )

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

    def sample_paired_losses(
        self,
        theta_plus: np.ndarray,
        theta_minus: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, list[Trajectory], np.ndarray, list[Trajectory]]:









        count = int(count)
        if count <= 0:
            raise ValueError("trajectory count must be positive")
        seeds = rng.randint(0, 2**31 - 1, size=count, dtype=np.int64)
        plus_principal, plus_follower = self.deploy_sampling(theta_plus)
        plus_principal = np.asarray(plus_principal, dtype=float).copy()
        plus_follower = np.asarray(plus_follower, dtype=float).copy()
        minus_principal, minus_follower = self.deploy_sampling(theta_minus)
        minus_principal = np.asarray(minus_principal, dtype=float).copy()
        minus_follower = np.asarray(minus_follower, dtype=float).copy()

        plus_trajectories: list[Trajectory] = []
        minus_trajectories: list[Trajectory] = []
        for child_seed in seeds:
            seed_value = int(child_seed)
            plus_rng = np.random.RandomState(seed_value)
            minus_rng = np.random.RandomState(seed_value)
            plus_trajectories.append(
                self._sample_trajectory_from_probabilities(
                    plus_principal, plus_follower, plus_rng
                )
            )
            minus_trajectories.append(
                self._sample_trajectory_from_probabilities(
                    minus_principal, minus_follower, minus_rng
                )
            )

        plus_returns = np.fromiter(
            (self.discounted_return(t) for t in plus_trajectories),
            dtype=float,
            count=count,
        )
        minus_returns = np.fromiter(
            (self.discounted_return(t) for t in minus_trajectories),
            dtype=float,
            count=count,
        )
        return (
            -plus_returns,
            plus_trajectories,
            -minus_returns,
            minus_trajectories,
        )


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
