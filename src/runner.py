from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cache import CacheManager, build_fingerprint
from src.config import load_config, save_config
from src.methods import METHODS
from src.problem import PricingProblem, ProblemSpec
from src.progress import progress_bar, reset_bar
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
    results: list[dict] = []

    overall = progress_bar(total=total_jobs, desc="Experiments", unit="job", leave=True)
    samples = progress_bar(
        total=context.max_samples,
        desc="Current method",
        unit="sample",
        leave=False,
    )
    try:
        for week_value in experiment["weeks"]:
            week_id = str(week_value).zfill(2)
            for run_index in range(simulations):
                problem = PricingProblem.from_week(project_root / "data", week_id, rng, spec)
                for method_name in method_names:
                    job_label = f"{weeks[week_id]} | run {run_index + 1}/{simulations} | {method_name}"
                    overall.set_description(job_label, refresh=False)
                    job_cache = cache.job(week_id, run_index, method_name)
                    saved = job_cache.load_final()
                    if saved is not None:
                        trace = saved["trace"]
                        rng.set_state(saved["rng_state"])
                        reset_bar(samples, context.max_samples, context.max_samples, "Current method")
                        status = "cached"
                    else:
                        progress_saved = job_cache.load_progress()
                        initial = 0 if progress_saved is None else min(
                            int(progress_saved["state"]["sample_count"]), context.max_samples
                        )
                        reset_bar(samples, context.max_samples, initial, "Current method")
                        trace = METHODS[method_name](config["methods"][method_name]).run(
                            problem, rng, context, job_cache, samples
                        )
                        job_cache.save_final(trace, rng.get_state())
                        status = "resumed" if progress_saved else "computed"

                    results.append(
                        {
                            "week_id": week_id,
                            "week_label": weeks[week_id],
                            "run": run_index,
                            "method": method_name,
                            "trace": trace,
                        }
                    )
                    overall.set_postfix(status=status, refresh=False)
                    overall.update(1)
    finally:
        samples.close()
        overall.close()

    write_outputs(results, method_names, int(experiment["figure_run"]), output_dir)
    save_config(config, output_dir / "config.yaml")
    cache.mark_complete()
    print(f"Finished. Results are in {output_dir}")
