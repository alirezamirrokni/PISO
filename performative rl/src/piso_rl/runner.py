from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .cache import RunCache
from .config import load_config
from .methods import METHODS
from .problem import PerformativeRLProblem
from .report import build_reports


@dataclass(frozen=True)
class RunContext:
    max_trajectories: int
    evaluation_interval: int
    checkpoint_interval: int


def _method_seed(master_seed: int, experiment_seed: int, alias: str) -> int:
    payload = f"{int(master_seed)}|{int(experiment_seed)}|{alias}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _build_problem(config: dict, experiment_seed: int) -> PerformativeRLProblem:
    environment = config["environment"]
    policy = config["policy"]
    experiment = config.get("experiment", {})
    vary_problem = bool(experiment.get("vary_problem_by_seed", False))
    problem_seed = (
        int(experiment_seed)
        if vary_problem
        else int(experiment.get("problem_seed", 0))
    )
    return PerformativeRLProblem(
        beta=float(environment["beta"]),
        eps=float(environment["eps"]),
        discount=float(environment["discount"]),
        num_followers=int(environment["num_followers"]),
        seed=problem_seed,
        feature_dim=int(policy.get("feature_dim", 32)),
        theta_std=float(policy.get("theta_std", 0.10)),
        phi_std=float(policy.get("phi_std", 0.20)),
        horizon=int(environment.get("horizon", 100)),
    )


def _run_single(
    task_index: int,
    config: dict,
    cache_root: str,
    method_name: str,
    seed: int,
    context: RunContext,
    master_seed: int,
    reset_cache: bool,
):
    problem = _build_problem(config, seed)
    method_config = config["methods"].get(method_name)
    if not isinstance(method_config, dict):
        raise ValueError(f"missing parameter mapping for {method_name}")

    method = METHODS[method_name](method_config)
    alias = getattr(method, "seed_alias", method_name)
    rng_seed = _method_seed(master_seed, seed, alias)
    rng = np.random.RandomState(rng_seed)
    fingerprint_payload = {
        "method": method_name,
        "method_params": method_config,
        "problem": problem.signature(),
        "max_trajectories": context.max_trajectories,
        "evaluation_interval": context.evaluation_interval,
        "checkpoint_interval": context.checkpoint_interval,
        "rng_seed": rng_seed,
    }
    cache = RunCache(cache_root, method_name, seed, fingerprint_payload)
    cache.checkpoint_interval = int(context.checkpoint_interval)
    if reset_cache:
        cache.reset()

    trace = cache.load_final()
    status = "cached"
    if trace is None:
        status = "computed/resumed"
        cache.start_timing()
        trace = method.run(problem, rng, context, cache)
        trace.runtime_seconds = cache.stop_timing()
        cache.save_final(trace)

    trace.method = method_name
    trace.seed = seed
    return task_index, trace, status, method_name, seed


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    selected_methods: list[str] | None = None,
    reset_cache: bool = False,
    jobs: int | None = None,
    seeds: list[int] | None = None,
    max_trajectories: int | None = None,
    evaluation_interval: int | None = None,
    checkpoint_interval: int | None = None,
) -> list:
    config = load_config(config_path)
    experiment = config["experiment"]
    output_candidate = Path(output_dir or experiment.get("output_dir", "results/piso_v4_heldout")).expanduser()
    if output_candidate.is_absolute():
        output = output_candidate.resolve()
    else:
        output = (Path(__file__).resolve().parents[3] / output_candidate).resolve()
    cache_root = output / "cache"
    output.mkdir(parents=True, exist_ok=True)

    enabled = list(config["methods"]["enabled"])
    if selected_methods:
        enabled = selected_methods
    unknown = [name for name in enabled if name not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; available: {sorted(METHODS)}")

    run_seeds = [int(seed) for seed in (seeds if seeds is not None else experiment["seeds"])]
    if not run_seeds:
        raise ValueError("at least one seed is required")
    run_evaluation_interval = int(
        evaluation_interval
        if evaluation_interval is not None
        else experiment.get("evaluation_interval", 1000)
    )
    run_max_trajectories = int(
        max_trajectories
        if max_trajectories is not None
        else experiment["max_trajectories"]
    )
    run_checkpoint_interval = int(
        checkpoint_interval
        if checkpoint_interval is not None
        else experiment.get("checkpoint_interval", 1)
    )
    if (
        run_max_trajectories <= 0
        or run_evaluation_interval <= 0
        or run_checkpoint_interval <= 0
    ):
        raise ValueError("trajectory, evaluation, and checkpoint intervals must be positive")
    context = RunContext(
        max_trajectories=run_max_trajectories,
        evaluation_interval=run_evaluation_interval,
        checkpoint_interval=run_checkpoint_interval,
    )
    master_seed = int(experiment.get("master_seed", 2026))
    n_jobs = int(jobs if jobs is not None else experiment.get("n_jobs", 1))

    
    experiment["seeds"] = list(run_seeds)
    experiment.setdefault("problem_seed", 0)
    experiment.setdefault("vary_problem_by_seed", False)
    experiment["max_trajectories"] = run_max_trajectories
    experiment["evaluation_interval"] = run_evaluation_interval
    experiment["checkpoint_interval"] = run_checkpoint_interval
    experiment["n_jobs"] = n_jobs
    experiment["output_dir"] = str(output)
    config["methods"]["enabled"] = list(enabled)
    if n_jobs <= 0:
        raise ValueError("number of jobs must be positive")
    if n_jobs > 1:
        
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(variable, "1")

    tasks = [
        (index, method_name, seed)
        for index, (seed, method_name) in enumerate(
            (seed, method_name) for seed in run_seeds for method_name in enabled
        )
    ]
    traces_by_index = {}
    progress = tqdm(total=len(tasks), desc="PISO RL runs")

    if n_jobs == 1:
        for index, method_name, seed in tasks:
            result = _run_single(
                index,
                config,
                str(cache_root),
                method_name,
                seed,
                context,
                master_seed,
                reset_cache,
            )
            task_index, trace, status, completed_method, completed_seed = result
            traces_by_index[task_index] = trace
            progress.set_postfix(
                method=completed_method,
                seed=completed_seed,
                status=status,
            )
            progress.update(1)
    else:
        
        process_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=process_context,
        ) as executor:
            futures = {
                executor.submit(
                    _run_single,
                    index,
                    config,
                    str(cache_root),
                    method_name,
                    seed,
                    context,
                    master_seed,
                    reset_cache,
                ): (method_name, seed)
                for index, method_name, seed in tasks
            }
            try:
                for future in as_completed(futures):
                    task_index, trace, status, method_name, seed = future.result()
                    traces_by_index[task_index] = trace
                    progress.set_postfix(
                        method=method_name,
                        seed=seed,
                        status=status,
                    )
                    progress.update(1)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    progress.close()

    traces = [traces_by_index[index] for index, _, _ in tasks]
    build_reports(traces, config, output)
    print(
        f"Results and plots written to {output} "
        f"(jobs={n_jobs}, evaluation_interval={run_evaluation_interval}, "
        f"checkpoint_interval={run_checkpoint_interval}, max_trajectories={run_max_trajectories})"
    )
    return traces
