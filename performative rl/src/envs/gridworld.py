import copy
import itertools
import json

import numpy as np
from numpy.testing import assert_almost_equal

from src.agents.gw_agent import Agent
from src.policies.policies import *


class Gridworld:
    def __init__(
        self,
        beta,
        eps,
        gamma,
        num_followers,
        sampling=False,
        n_sample=500,
        seed=1,
        max_sample_steps=100,
    ):
        # grid
        h = -0.5
        f = -0.02
        self.grid = np.array(
            [
                [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
                [-0.01, -0.01, f, -0.01, h, -0.01, -0.01, -0.01],
                [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
                [-0.01, f, -0.01, -0.01, -0.01, h, -0.01, f],
                [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
                [-0.01, h, h, -0.01, f, -0.01, h, -0.01],
                [-0.01, h, -0.01, -0.01, h, -0.01, h, -0.01],
                [-0.01, -0.01, -0.01, h, -0.01, f, -0.01, +1],
            ]
        )

        self.dim = np.prod(self.grid.shape)
        self.state_ids = range(self.dim)
        self.terminal_state_id = self.dim - 1
        self.reward_space = [-0.01, f, h, 1]
        # initial states
        self.initial_states = [
            s
            for s in self.state_ids
            if (state := self._get_state(s)) and (state[0] == 0 or state[1] == 0)
        ]
        self.rho = self._get_initial_state_distribution()
        # move: left, right, up, down
        self.moves = [0, 1, 2, 3]
        self.move_mapping = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self.move_meaning = {0: "left", 1: "right", 2: "up", 3: "down"}
        # intervene: left, right, up, down
        self.interventions = [0, 1, 2, 3, 4]
        self.interventions_mapping = {
            move: self.move_mapping[move] for move in self.moves
        }
        self.interventions_mapping[4] = False
        self.interventions_meaning = {
            move: self.move_meaning[move] for move in self.moves
        }
        self.interventions_meaning[4] = "nope"
        # cost of intervention
        self.C = -0.05
        # agents
        self.num_of_agents = 2
        self.agents = {}
        self.action_space = {}
        # agent 1 is the main agent (controls actor movement based on the correct gird)
        self.action_space[1] = self.moves
        self.agents[1] = Agent(1, self.action_space[1])
        # agent 2 is the secondary agent (makes interventions based on one or more pertrubed grids) and it is considered as part of the environment
        # followers are basically integrated in agent 2
        self.action_space[2] = self.interventions
        self.agents[2] = Agent(2, self.action_space[2])
        # pertrub grids for agent 2/followers -- the pertrubed grids stay the same for all experiments
        self.num_followers = num_followers
        self.eps = eps
        self.pertrubed_grids = []
        for pertrubation_seed in range(num_followers):
            pertrubation_rng = np.random.default_rng(pertrubation_seed)
            self.pertrubed_grids.append(self.pertrub_grid(pertrubation_rng))
        # pertrubed grid that computational functions solve for
        self.active_pertrubed_grid = self.pertrubed_grids[0]
        # parameters
        self.gamma = gamma
        self.beta = beta
        # sampling
        self.sampling = sampling
        self.n_sample = n_sample
        self.max_sample_steps = max_sample_steps
        # set random generator (for trajectory sampling)
        self.rng = np.random.default_rng(seed)

        # Precompute the deterministic grid mechanics once.  The original
        # implementation rebuilt these transitions through nested Python
        # loops for every follower response and every exact evaluation.
        self._build_fast_tables()

        self.reset()


    def _build_fast_tables(self):
        """Precompute joint transitions and rewards for vectorized solvers."""
        num_states = int(self.dim)
        num_moves = len(self.moves)
        num_interventions = len(self.interventions)

        move_next = np.empty((num_states, num_moves), dtype=np.int64)
        for state in self.state_ids:
            for move in self.moves:
                move_next[state, move] = self.get_next_state(state, move)

        joint_next = np.empty(
            (num_states, num_moves, num_interventions), dtype=np.int64
        )
        for follower_action in self.interventions:
            if follower_action == 4:
                joint_next[:, :, follower_action] = move_next
            else:
                joint_next[:, :, follower_action] = move_next[:, follower_action, None]

        true_rewards = self.grid.reshape(-1)[joint_next].astype(np.float64)
        true_rewards[self.terminal_state_id, :, :] = 0.0

        self._move_next = move_next
        self._joint_next = joint_next
        self._principal_joint_rewards = true_rewards
        self._state_index = np.arange(num_states, dtype=np.int64)

    def _policy_matrix(self, agent):
        policy = getattr(agent, "policy", None)
        pi = getattr(policy, "pi", None)
        if pi is not None:
            array = np.asarray(pi, dtype=np.float64)
            expected = (int(self.dim), len(agent.actions))
            if array.shape == expected:
                return array
        return self._get_policy_array(agent)

    def _follower_reward_table(self, perturbed_grid):
        rewards = np.asarray(perturbed_grid, dtype=np.float64).reshape(-1)[
            self._joint_next
        ].copy()
        rewards[:, :, :4] += float(self.C)
        rewards[self.terminal_state_id, :, :] = 0.0
        return rewards

    def _best_response_q_vectorized(self, principal_probabilities, perturbed_grid, tol=1e-5):
        rewards = self._follower_reward_table(perturbed_grid)
        values = np.zeros(int(self.dim), dtype=np.float64)

        while True:
            continuation = values[self._joint_next]
            q_values = np.einsum(
                "sa,saf->sf",
                principal_probabilities,
                rewards + float(self.gamma) * continuation,
                optimize=True,
            )
            q_values[self.terminal_state_id, :] = 0.0
            updated = np.max(q_values, axis=1)
            updated[self.terminal_state_id] = 0.0
            if float(np.max(np.abs(updated - values))) < float(tol):
                values = updated
                break
            values = updated

        continuation = values[self._joint_next]
        q_values = np.einsum(
            "sa,saf->sf",
            principal_probabilities,
            rewards + float(self.gamma) * continuation,
            optimize=True,
        )
        q_values[self.terminal_state_id, :] = 0.0
        return q_values

    def reset(self):
        """ """
        self.initialize_policies()

        return

    def get_next_state(self, state_id, move):
        """ """
        assert move in self.moves, f"Illegal move {move} was given."

        state = self._get_state(state_id)

        if self.is_terminal(state_id):
            return state_id

        next_state = self.vector_add(state, self.move_mapping[move])

        if not self.is_valid(next_state):
            next_state = state

        next_state_id = self._get_state_id(next_state)

        return next_state_id

    def get_rewards(self, state_id, next_state_id):
        """ """
        rewards = {}

        if self.is_terminal(state_id):
            rewards[1] = 0
            rewards[2] = 0
            return rewards

        state = self._get_state(next_state_id)
        # agent 1
        rewards[1] = self.grid[state[0], state[1]]
        # agent 2
        rewards[2] = self.active_pertrubed_grid[state[0], state[1]]

        return rewards

    def _get_initial_state_distribution(self):
        """ """
        rho = np.zeros(self.dim, dtype="float64")
        initial_states = self.initial_states
        for s in initial_states:
            rho[s] = 1 / len(initial_states)

        return rho

    def value_iteration(self, agent, tol=1e-5):
        """ """
        gamma = self.gamma

        # initialize all state values to zero
        U = np.zeros(self.dim, dtype="float64")
        while True:
            U_old = copy.deepcopy(U)
            delta = 0
            for s in self.state_ids:
                if self.is_terminal(s):
                    U[s] = 0
                else:
                    bell = np.zeros_like(self.moves, dtype="float64")
                    for move in self.moves:
                        s_pr = self.get_next_state(s, move)
                        r = self.get_rewards(s, s_pr)
                        bell[move] = r[agent.id] + gamma * U_old[s_pr]
                    U[s] = np.max(bell)
                delta = max(delta, abs(U[s] - U_old[s]))

            if delta < tol:
                break

        return U

    def _get_Q(self, U, agent):
        """ """
        gamma = self.gamma

        # initialize all state-move values to zero
        Q = np.zeros(shape=(self.dim, len(self.moves)), dtype="float64")
        for s, move in itertools.product(self.state_ids, self.moves):
            if self.is_terminal(s):
                Q[s, move] = 0
            else:
                s_pr = self.get_next_state(s, move)
                r = self.get_rewards(s, s_pr)
                Q[s, move] = r[agent.id] + gamma * U[s_pr]

        return Q

    def get_mnext_state(self, state_id, actions):
        """
        actions: dict {agent.id: action}
        """
        for agent in self.agents.values():
            assert actions[agent.id] in agent.actions, (
                f"Illegal action {actions[agent.id]} was given."
            )

        if not self.intervention_happened(actions):
            next_state_id = self.get_next_state(state_id, actions[1])
        else:
            next_state_id = self.get_next_state(state_id, actions[2])

        return next_state_id

    def get_mrewards(self, state_id, actions, next_state_id):
        """ """
        rewards = self.get_rewards(state_id, next_state_id)
        if self.intervention_happened(actions):
            rewards[2] += self.C

        return rewards

    def best_response_value_iteration(self, agent, fixed_agent, tol=1e-5):
        """ """
        gamma = self.gamma

        # initialize all state values to zero
        U = np.zeros(self.dim, dtype="float64")
        while True:
            U_old = copy.deepcopy(U)
            delta = 0
            for s in self.state_ids:
                if self.is_terminal(s):
                    U[s] = 0
                else:
                    p = fixed_agent._get_probs(s)
                    bell = np.zeros_like(agent.actions, dtype="float64")
                    for a in agent.actions:
                        for fa in fixed_agent.actions:
                            actions = {fixed_agent.id: fa, agent.id: a}
                            s_pr = self.get_mnext_state(s, actions)
                            r = self.get_mrewards(s, actions, s_pr)
                            bell[a] += p[fa] * (r[agent.id] + gamma * U_old[s_pr])
                    U[s] = np.max(bell)
                delta = max(delta, abs(U[s] - U_old[s]))
            if delta < tol:
                break

        return U

    def _get_mU(self, agent, fixed_agent, tol=1e-5):
        """ """
        gamma = self.gamma

        # initialize all state values to zero
        U = np.zeros(self.dim, dtype="float64")
        while True:
            U_old = copy.deepcopy(U)
            delta = 0
            for s in self.state_ids:
                if self.is_terminal(s):
                    U[s] = 0
                else:
                    p = fixed_agent._get_probs(s)
                    bell = np.zeros_like(agent.actions, dtype="float64")
                    for a in agent.actions:
                        for fa in fixed_agent.actions:
                            actions = {fixed_agent.id: fa, agent.id: a}
                            s_pr = self.get_mnext_state(s, actions)
                            r = self.get_mrewards(s, actions, s_pr)
                            bell[a] += p[fa] * (r[agent.id] + gamma * U_old[s_pr])
                    p = agent._get_probs(s)
                    U[s] = np.sum(p * bell)
                delta = max(delta, abs(U[s] - U_old[s]))
            if delta < tol:
                break

        return U

    def _get_mQ(self, U, agent, fixed_agent):
        """ """
        gamma = self.gamma

        # initialize all state-move values to zero
        Q = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        for s, a in itertools.product(self.state_ids, agent.actions):
            if self.is_terminal(s):
                Q[s, a] = 0
            else:
                p = fixed_agent._get_probs(s)
                for fa in fixed_agent.actions:
                    actions = {fixed_agent.id: fa, agent.id: a}
                    s_pr = self.get_mnext_state(s, actions)
                    r = self.get_mrewards(s, actions, s_pr)
                    Q[s, a] += p[fa] * (r[agent.id] + gamma * U[s_pr])

        return Q

    def _get_RT(self):
        """
        Get R and T functions for agent 1, given the fixed policy of agent 2.
        R(s,a) is the expected reward that agent 1 will receive after taking action a in state s
        T(s, a, s_pr) is the probability of the actor transitioning from state s to state s_pr after agent 1 takes action a
        """
        if not self.sampling:
            return self._get_exact_RT()
        else:
            return self._get_approximate_RT()

    def _get_exact_RT(self):
        """Return exact principal rewards/transitions for the deployed follower."""
        agent = self.agents[1]
        fixed_agent = self.agents[2]
        follower_probabilities = self._policy_matrix(fixed_agent)

        num_states = int(self.dim)
        num_actions = len(agent.actions)
        T = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        R = np.zeros((num_states, num_actions), dtype=np.float64)

        states = self._state_index
        for action in agent.actions:
            for follower_action in fixed_agent.actions:
                probability = follower_probabilities[:, follower_action]
                next_states = self._joint_next[:, action, follower_action]
                T[states, action, next_states] += probability
                R[:, action] += (
                    probability
                    * self._principal_joint_rewards[:, action, follower_action]
                )

        return R, T

    def _get_approximate_RT(self):
        """ """
        agent = self.agents[1]
        n_sample = self.n_sample

        # total values
        T_tot = np.zeros(
            shape=(self.dim, len(agent.actions), self.dim), dtype="float64"
        )
        R_tot = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        # visitattion count
        V = np.zeros(shape=(self.dim, len(agent.actions)), dtype="int")
        # compute empirical data
        for _ in range(n_sample):
            trajectory = self.sample_trajectory()
            for s, a, s_pr, r in trajectory:
                # update visitation count
                V[s, a] += 1
                # update total values
                T_tot[s, a, s_pr] += 1
                R_tot[s, a] += r

        # approximated values
        T_hat = np.zeros(
            shape=(self.dim, len(agent.actions), self.dim), dtype="float64"
        )
        R_hat = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        # remove +1 from reward space
        reward_space = [r for r in self.reward_space if r != 1]
        for s in self.state_ids:
            r_states = self._get_reachable_states(s)
            for a in agent.actions:
                if self.is_terminal(s):
                    R_hat[s, a] = 0
                    T_hat[s, a, self.terminal_state_id] = 1
                elif V[s, a]:
                    R_hat[s, a] = R_tot[s, a] / V[s, a]
                    for s_pr in r_states:
                        T_hat[s, a, s_pr] = T_tot[s, a, s_pr] / V[s, a]
                else:
                    # if s, a not seen during sampling assign to R[s,a] the mean of all negative rewards
                    R_hat[s, a] = np.mean(reward_space)
                    # if if s, a not seen during sampling assign to T[s, a] the uniform distiribution over all reachable states
                    for s_pr in r_states:
                        T_hat[s, a, s_pr] = 1 / len(r_states)

                assert_almost_equal(np.sum(T_hat[s, a, :]), 1)

        return R_hat, T_hat

    def _get_d(self, T, agent, tol=1e-5):
        """Compute the discounted state-action occupancy measure."""
        transition = np.asarray(T, dtype=np.float64)
        probabilities = self._policy_matrix(agent)
        occupancy = np.zeros_like(probabilities, dtype=np.float64)

        while True:
            previous = occupancy
            incoming = np.einsum(
                "sa,san->n", previous, transition, optimize=True
            )
            state_mass = self.rho + float(self.gamma) * incoming
            occupancy = probabilities * state_mass[:, None]
            occupancy[self.terminal_state_id, :] = 0.0
            if float(np.max(np.abs(occupancy - previous))) < float(tol):
                break

        return occupancy

    def is_valid(self, state):
        """ """
        grid_shape = self.grid.shape

        return bool(0 <= state[0] < grid_shape[0] and 0 <= state[1] < grid_shape[1])

    def is_terminal(self, state_id):
        """ """

        return bool(state_id == self.terminal_state_id)

    def intervention_happened(self, actions):
        """ """

        return bool(self.interventions_mapping[actions[2]])

    def _get_state(self, state_id):
        """ """
        grid_shape = self.grid.shape
        state = (state_id // grid_shape[0], state_id % grid_shape[1])

        return state

    def _get_state_id(self, state):
        """ """
        grid_shape = self.grid.shape
        state_id = state[0] * grid_shape[0] + state[1]

        return state_id

    def _get_reachable_states(self, state_id):
        """ """
        r_states = []

        for move in self.moves:
            next_state_id = self.get_next_state(state_id, move)
            if next_state_id not in r_states:
                r_states.append(next_state_id)

        return r_states

    def pertrub_grid(self, pertrubation_rng):
        """ """
        rng = pertrubation_rng
        grid_shape = self.grid.shape

        grid_values = np.unique(self.grid)
        grid_values = np.delete(grid_values, np.where(grid_values == 1))

        new_grid = np.zeros(shape=grid_shape)
        r = rng.random(grid_shape)
        for i, j in itertools.product(range(grid_shape[0]), range(grid_shape[1])):
            # initial states stay the same
            if self._get_state_id([i, j]) in self.initial_states or r[i, j] > self.eps:
                new_grid[i, j] = self.grid[i, j]
            else:
                new_grid[i, j] = rng.choice(grid_values)

        # goal value remains the same
        new_grid[-1, -1] = self.grid[-1, -1]

        return new_grid

    def sample_trajectory(self):
        """ """
        agent = self.agents[1]
        fixed_agent = self.agents[2]
        rho = self.rho
        rng = self.rng

        # trajectory quartets (state, action, next state, reward)
        trajectory = []
        n_steps = 0
        # initial state
        s = rng.choice(np.arange(self.dim), p=rho)
        while not self.is_terminal(s) and n_steps < self.max_sample_steps:
            # actions
            actions = {}
            actions[agent.id] = agent.take_action(s, rng)
            actions[fixed_agent.id] = fixed_agent.take_action(s, rng)
            a = actions[agent.id]
            # next state
            s_pr = self.get_mnext_state(s, actions)
            # rewards
            r = self.get_mrewards(s, actions, s_pr)
            # update trajectory
            trajectory.append((s, a, s_pr, r[agent.id]))
            # prepare for next time-step
            s = s_pr
            n_steps += 1

        return trajectory

    def _get_policy_array(self, agent):
        """
        Given the (initial) agent's policy return the probability array
        """
        pi = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")

        for s in self.state_ids:
            p = agent._get_probs(s)
            for a in agent.actions:
                pi[s, a] = p[a]

        return pi

    def initialize_policies(self):
        """ """
        agents = self.agents

        # agent 1
        agent = agents[1]

        U_star = self.value_iteration(agent)
        Q_star = self._get_Q(U_star, agent)
        agent.policy = Eps_Greedy_Policy(agent.actions, Q_star)

        # agent 2
        agent = agents[2]

        agent.policy = self.response_model(agents)

        return

    def response_model(self, agents):
        """Compute the follower response with a vectorized Bellman solver."""
        fixed_agent = agents[1]
        agent = agents[2]
        principal_probabilities = self._policy_matrix(fixed_agent)

        q_values = []
        for perturbed_grid in self.pertrubed_grids:
            self.active_pertrubed_grid = perturbed_grid
            q_values.append(
                self._best_response_q_vectorized(
                    principal_probabilities, perturbed_grid
                )
            )

        average_q = np.mean(q_values, axis=0)
        logits = float(self.beta) * average_q
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return Tabular(agent.actions, probabilities)

    def _get_env_vis(self):
        """
        Returns a visualization of the environment/policy of agent 2
        """
        agent = self.agents[2]
        grid_shape = self.grid.shape

        vis = []
        for i in range(grid_shape[0]):
            vis_row = []
            for j in range(grid_shape[1]):
                s = self._get_state_id([i, j])
                p = agent._get_probs(s)
                a = np.argmax(p)
                # a should have probability higher than nope
                if p[a] == p[4]:
                    a = 4
                vis_row.append(self.interventions_meaning[a])
            vis.append(vis_row)

        return vis

    @staticmethod
    def vector_add(x, y):
        return (x[0] + y[0], x[1] + y[1])
