from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from src.cache import CacheManager
from src.config import load_config, save_config
from src.methods import METHODS
from src.problem import ProblemSpec
from src.progress import ExperimentProgress
from src.report import write_outputs


def _private_method_seed(
    base_seed: int,
    week_id: str,
    run_index: int,
    method: str,
) -> int:
    payload = f"{base_seed}:{week_id}:{run_index}:{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


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
    cache = CacheManager(
        output_dir,
        project_root,
        config,
        reset=reset_cache,
    )
    if cache.is_complete(required_outputs):
        print(f"Complete cache found in {output_dir}. No experiment was rerun.")
        return

    weeks = load_config(project_root / "data" / "weeks.yaml")
    spec = ProblemSpec.from_mapping(config["problem"])
    context = Context(
        metric_samples=int(experiment["metric_samples"]),
        max_samples=int(experiment["max_samples"]),
    )
    base_seed = int(experiment["seed"])
    rng = np.random.RandomState(base_seed)
    simulations = int(experiment["simulations"])
    total_pairs = len(experiment["weeks"]) * len(method_names)
    total_runs = total_pairs * simulations
    results: list[dict] = []

    print(
        f"Running {total_pairs} dataset-method pairs across {simulations} simulations "
        f"({total_runs} method runs). Checkpoints use one CSV per dataset and method "
        f"in {cache.root}."
    )
    progress = ExperimentProgress(total_pairs, simulations, context.max_samples)
    try:
        # The released baseline order is preserved: week -> simulation -> method.
        # Stable problem-instance checkpoints prevent a changed method from
        # altering later problem instances. PISO variants use private RNGs.
        for week_value in experiment["weeks"]:
            week_id = str(week_value).zfill(2)
            for run_index in range(simulations):
                problem = cache.problem(
                    project_root / "data",
                    week_id,
                    run_index,
                    rng,
                    spec,
                )
                for method_name in method_names:
                    label = (
                        f"{weeks[week_id]} | run {run_index + 1}/{simulations} | "
                        f"{method_name}"
                    )
                    method_class = METHODS[method_name]
                    if getattr(method_class, "private_rng", False):
                        job_rng = np.random.RandomState(
                            _private_method_seed(
                                base_seed,
                                week_id,
                                run_index,
                                method_name,
                            )
                        )
                    else:
                        job_rng = rng

                    job_cache = cache.job(week_id, run_index, method_name)
                    saved = job_cache.load_final()
                    if saved is not None:
                        trace = saved["trace"]
                        job_rng.set_state(saved["rng_state"])
                        progress.start_run(
                            label,
                            context.max_samples,
                            len(trace.samples) - 1,
                            "cached",
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
                        handle = progress.start_run(
                            label,
                            initial_samples,
                            initial_iteration,
                            "resuming" if partial is not None else "starting",
                        )
                        trace = method_class(config["methods"][method_name]).run(
                            problem,
                            job_rng,
                            context,
                            job_cache,
                            handle,
                        )
                        job_cache.save_final(trace, job_rng.get_state())
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
                    progress.finish_run(
                        status,
                        int(trace.samples[-1]),
                        len(trace.samples) - 1,
                    )
    finally:
        progress_summary = progress.close()

    write_outputs(results, method_names, int(experiment["figure_run"]), output_dir)
    save_config(config, output_dir / "config.yaml")
    cache.mark_complete()
    print(
        f"Finished: {progress_summary.computed} computed, "
        f"{progress_summary.resumed} resumed, "
        f"{progress_summary.cached} loaded from cache. Results are in {output_dir}"
    )
