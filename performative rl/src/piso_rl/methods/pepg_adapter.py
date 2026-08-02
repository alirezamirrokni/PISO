from __future__ import annotations

import os

import numpy as np

from .common import can_afford, finish, initialize_state, record


class PePGBaseline:
    """Checkpointable adapter around the repository's official PePG class.

    PePG uses learned reward/transition models after warm-up.  Its training cost
    is counted as ``num_trajectories`` per update, matching the trajectories
    collected by ``PePG._collect_trajectories``.
    """

    name = "PePG"
    seed_alias = "PePG"

    def __init__(self, params: dict) -> None:
        self.p = dict(params)
        if int(self.p.get("num_trajectories", 30)) <= 0:
            raise ValueError("PePG num_trajectories must be positive")

    def _construct(self, problem, seed: int):
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_SILENT", "true")
        import torch
        torch.manual_seed(int(seed))

        from src.envs.gridworld import Gridworld
        from src.performative_pepg import PePG
        from src.policies.tabular import Tabular

        env = Gridworld(
            beta=float(problem.env.beta),
            eps=float(problem.env.eps),
            gamma=float(problem.env.gamma),
            num_followers=int(problem.env.num_followers),
            sampling=False,
            n_sample=1,
            seed=int(seed),
            max_sample_steps=int(problem.horizon),
        )
        algorithm = PePG(
            env=env,
            max_iterations=1,
            lamda=float(self.p.get("regularization", 0.0)),
            reg=str(self.p.get("regularizer", "L2")),
            gradient=False,
            eta=float(self.p["step_size"]),
            sampling=False,
            n_sample=None,
            policy_gradient=True,
            nu=float(self.p.get("nu", 0.10)),
            unregularized_obj=True,
            lagrangian=False,
            N=1,
            delta=0.10,
            B=1,
            feature_dim=int(problem.feature_dim),
            nn_lr=float(self.p.get("model_learning_rate", 0.001)),
            warmup=int(self.p.get("warmup", 1)),
            num_trajectories=int(self.p.get("num_trajectories", 30)),
            value_lr=float(self.p.get("value_learning_rate", 0.10)),
            trajectory_horizon=int(problem.horizon),
            seed=int(seed),
            track_internal_metrics=False,
        )
        algorithm.theta = np.asarray(problem.initial_x, dtype=float).copy()
        algorithm.phi = np.asarray(problem.phi, dtype=float).copy()
        algorithm.pi_theta = algorithm._compute_policy()
        algorithm.agents[1].policy = Tabular(
            algorithm.agents[1].actions, algorithm.pi_theta
        )
        algorithm.agents[2].policy = algorithm.env.response_model(algorithm.agents)
        algorithm.R, algorithm.T = algorithm.env._get_RT()
        algorithm.d_last = algorithm.env._get_d(algorithm.T, algorithm.agents[1])
        return algorithm, torch

    @staticmethod
    def _snapshot(algorithm, torch, common_state: dict) -> dict:
        return {
            "common": common_state,
            "theta": algorithm.theta,
            "phi": algorithm.phi,
            "pi_theta": algorithm.pi_theta,
            "R": algorithm.R,
            "T": algorithm.T,
            "d_last": algorithm.d_last,
            "d_diff": algorithm.d_diff,
            "sub_gap": algorithm.sub_gap,
            "v_values": algorithm.v_values,
            "iteration": algorithm.iteration,
            "warmup_data": [
                {
                    "theta": np.asarray(item["theta"], dtype=float).copy(),
                    "rewards": np.asarray(item["rewards"], dtype=float).copy(),
                }
                for item in algorithm.warmup_data
            ],
            "f_r": algorithm.f_r.state_dict(),
            "f_p": algorithm.f_p.state_dict(),
            "value_network": algorithm.value_network.state_dict(),
            "f_r_optimizer": algorithm.f_r_optimizer.state_dict(),
            "f_p_optimizer": algorithm.f_p_optimizer.state_dict(),
            "value_optimizer": algorithm.value_optimizer.state_dict(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
        }

    @staticmethod
    def _restore(algorithm, torch, snapshot: dict) -> dict:
        from src.policies.tabular import Tabular

        algorithm.theta = np.asarray(snapshot["theta"], dtype=float)
        algorithm.phi = np.asarray(snapshot["phi"], dtype=float)
        algorithm.pi_theta = np.asarray(snapshot["pi_theta"], dtype=float)
        algorithm.R = np.asarray(snapshot["R"], dtype=float)
        algorithm.T = np.asarray(snapshot["T"], dtype=float)
        algorithm.d_last = np.asarray(snapshot["d_last"], dtype=float)
        algorithm.d_diff = list(snapshot["d_diff"])
        algorithm.sub_gap = list(snapshot["sub_gap"])
        algorithm.v_values = list(snapshot["v_values"])
        algorithm.iteration = int(snapshot["iteration"])
        algorithm.warmup_data = list(snapshot["warmup_data"])
        algorithm.f_r.load_state_dict(snapshot["f_r"])
        algorithm.f_p.load_state_dict(snapshot["f_p"])
        algorithm.value_network.load_state_dict(snapshot["value_network"])
        algorithm.f_r_optimizer.load_state_dict(snapshot["f_r_optimizer"])
        algorithm.f_p_optimizer.load_state_dict(snapshot["f_p_optimizer"])
        algorithm.value_optimizer.load_state_dict(snapshot["value_optimizer"])
        algorithm.agents[1].policy = Tabular(
            algorithm.agents[1].actions, algorithm.pi_theta
        )
        algorithm.agents[2].policy = algorithm.env.response_model(algorithm.agents)
        np.random.set_state(snapshot["numpy_state"])
        torch.set_rng_state(snapshot["torch_state"])
        return snapshot["common"]

    def run(self, problem, rng, context, cache):
        seed = int(rng.randint(0, 2**31 - 1))
        np.random.seed(seed)
        algorithm, torch = self._construct(problem, seed)
        saved = cache.load_progress()
        if saved is None:
            common_state = initialize_state(problem, context.evaluation_interval)
            cache.save_progress(
                self._snapshot(algorithm, torch, common_state), rng.get_state()
            )
        else:
            rng.set_state(saved["rng_state"])
            common_state = self._restore(algorithm, torch, saved["state"])

        cost = int(self.p.get("num_trajectories", 30))
        while can_afford(common_state, cost, context.max_trajectories):
            algorithm.retrain1()
            algorithm.retrain2()
            if algorithm.iteration >= algorithm.warmup:
                algorithm.R = algorithm._get_performative_rewards()
                algorithm.T = algorithm._get_performative_transitions()

            common_state["theta"] = np.asarray(algorithm.theta, dtype=float).copy()
            common_state["sample_count"] += cost
            common_state["iteration"] += 1

            record(
                common_state,
                problem,
                context.evaluation_interval,
            )
            snapshot = self._snapshot(algorithm, torch, common_state)
            cache.save_progress(snapshot, rng.get_state())

        return finish(
            common_state,
            problem,
            context.evaluation_interval,
        )
