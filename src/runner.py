from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cache import CacheManager, build_fingerprint
from src.config import load_config, save_config
from src.methods import METHODS
from src.problem import PricingProblem, ProblemSpec
from src.progress import ExperimentProgress
from src.report import write_outputs


@dataclass(frozen=True)
class Context:
    metric_samples: int
    max_samples: int


def run_experiment(config_path: Path, output_dir: Path, reset_cache: bool = False) -> None:
    config_path = config_path.resolve()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    experiment = config["experiment"]
    method_names = list(config["methods"])
    unknown = [name for name in method_names if name not in METHODS]
    if unknown:
        raise KeyError(f"Methods not registered in src/methods/__init__.py: {unknown}")

    required_outputs = [
        output_dir / "summary.csv",
        output_dir / "final_scores.csv",
        output_dir / "figure_data.csv",
        output_dir / "figure.png",
        output_dir / "figure.pdf",
        output_dir / "config.yaml",
    ]
    fingerprint = build_fingerprint(project_root, config)
    cache = CacheManager(output_dir, fingerprint, config, reset=reset_cache)
    if cache.is_complete(required_outputs):
        print(f"Complete cache found in {output_dir}. No experiment was rerun.")
        return

    weeks = load_config(project_root / "data" / "weeks.yaml")
    spec = ProblemSpec.from_mapping(config["problem"])
    context = Context(
        metric_samples=int(experiment["metric_samples"]),
        max_samples=int(experiment["max_samples"]),
    )
    rng = np.random.RandomState(int(experiment["seed"]))
    simulations = int(experiment["simulations"])
    total_jobs = len(experiment["weeks"]) * simulations * len(method_names)
    cache_files = len(experiment["weeks"]) * len(method_names)
    results: list[dict] = []

    print(
        f"Running {total_jobs} jobs. Checkpoints use {cache_files} CSV files "
        f"(one per dataset-method pair) in {cache.root}."
    )
    progress = ExperimentProgress(total_jobs, context.max_samples)
    job_number = 0
    try:
        for week_value in experiment["weeks"]:
            week_id = str(week_value).zfill(2)
            for run_index in range(simulations):
                problem = PricingProblem.from_week(project_root / "data", week_id, rng, spec)
                for method_name in method_names:
                    job_number += 1
                    label = (
                        f"{weeks[week_id]} | run {run_index + 1}/{simulations} | {method_name}"
                    )
                    job_cache = cache.job(week_id, run_index, method_name)
                    saved = job_cache.load_final()
                    if saved is not None:
                        trace = saved["trace"]
                        rng.set_state(saved["rng_state"])
                        iteration = len(trace.samples) - 1
                        progress.start_job(
                            job_number,
                            label,
                            context.max_samples,
                            iteration,
                            "loading cache",
                        )
                        status = "cached"
                    else:
                        partial = job_cache.load_progress()
                        initial_samples = 0 if partial is None else int(
                            partial["state"]["sample_count"]
                        )
                        initial_iteration = 0 if partial is None else int(
                            partial["state"]["iteration"]
                        )
                        handle = progress.start_job(
                            job_number,
                            label,
                            initial_samples,
                            initial_iteration,
                            "resuming" if partial is not None else "starting",
                        )
                        trace = METHODS[method_name](config["methods"][method_name]).run(
                            problem,
                            rng,
                            context,
                            job_cache,
                            handle,
                        )
                        job_cache.save_final(trace, rng.get_state())
                        status = "resumed" if partial is not None else "computed"

                    results.append(
                        {
                            "week_id": week_id,
                            "week_label": weeks[week_id],
                            "run": run_index,
                            "method": method_name,
                            "trace": trace,
                        }
                    )
                    progress.finish_job(
                        status,
                        int(trace.samples[-1]),
                        len(trace.samples) - 1,
                    )
    finally:
        summary = progress.close()

    write_outputs(results, method_names, int(experiment["figure_run"]), output_dir)
    save_config(config, output_dir / "config.yaml")
    cache.mark_complete()
    print(
        f"Finished: {summary.computed} computed, {summary.resumed} resumed, "
        f"{summary.cached} loaded from cache. Results are in {output_dir}"
    )
