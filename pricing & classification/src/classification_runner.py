from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from src.classification_cache import ClassificationCacheManager
from src.classification_methods import METHODS
from src.classification_problem import (
    ClassificationSpec,
    StrategicClassificationProblem,
    load_classification_dataset,
)
from src.classification_report import write_classification_outputs
from src.config import load_config, save_config
from src.progress import ExperimentProgress


def _private_method_seed(
    base_seed: int,
    tau: float,
    run_index: int,
    method: str,
) -> int:
    payload = f"classification:{base_seed}:{float(tau):.8g}:{run_index}:{method}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@dataclass(frozen=True)
class Context:
    metric_samples: int
    max_samples: int


def run_experiment(
    config_path: Path,
    output_dir: Path,
    reset_cache: bool = False,
) -> None:
    config_path = config_path.resolve()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    experiment = config["experiment"]
    methods = list(config["methods"])
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise KeyError(
            f"Methods not registered in src/classification_methods/__init__.py: {unknown}"
        )
    taus = [float(value) for value in experiment["taus"]]
    if not taus or any(value <= 0.0 for value in taus):
        raise ValueError("experiment.taus must contain positive values")
    plotting = config.get("plotting", {})
    plot_methods = list(plotting.get("methods", methods))
    missing_plot = [name for name in plot_methods if name not in methods]
    if missing_plot:
        raise KeyError(
            "Plotting methods must also appear under config['methods']: "
            f"{missing_plot}"
        )

    spec = ClassificationSpec.from_mapping(config["problem"])
    dataset = load_classification_dataset(project_root, spec)
    cache = ClassificationCacheManager(
        output_dir,
        project_root,
        config,
        reset=reset_cache,
    )
    required_outputs = [
        output_dir / "summary.csv",
        output_dir / "final_scores.csv",
        output_dir / "figure_data.csv",
        output_dir / "figure_data_raw.csv",
        output_dir / "figure.png",
        output_dir / "figure.pdf",
        output_dir / "config.yaml",
    ]
    if cache.is_complete(required_outputs):
        print(
            f"Complete classification cache found in {output_dir}. "
            "All method runs will be loaded from cache and the reports and plots "
            "will be regenerated."
        )

    context = Context(
        metric_samples=int(experiment.get("metric_samples", 0)),
        max_samples=int(experiment["max_samples"]),
    )
    simulations = int(experiment["simulations"])
    base_seed = int(experiment["seed"])
    shared_rng = np.random.RandomState(base_seed)
    pair_count = len(taus) * len(methods)
    run_count = pair_count * simulations
    results: list[dict] = []
    print(
        f"Running {pair_count} tau-method pairs across {simulations} simulations "
        f"({run_count} method runs). Checkpoints are stored in {cache.root}."
    )
    progress = ExperimentProgress(pair_count, simulations, context.max_samples)
    try:
        
        
        for tau in taus:
            for run_index in range(simulations):
                problem = StrategicClassificationProblem(dataset, tau, spec)
                for method_name in methods:
                    method_class = METHODS[method_name]
                    label = (
                        f"tau={tau:g} | run {run_index + 1}/{simulations} | "
                        f"{method_name}"
                    )
                    if getattr(method_class, "private_rng", False):
                        alias = getattr(method_class, "seed_alias", method_name)
                        job_rng = np.random.RandomState(
                            _private_method_seed(
                                base_seed,
                                tau,
                                run_index,
                                alias,
                            )
                        )
                    else:
                        job_rng = shared_rng

                    job_cache = cache.job(tau, run_index, method_name)
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
                        initial_samples = (
                            0
                            if partial is None
                            else int(partial["state"]["sample_count"])
                        )
                        initial_iteration = (
                            0
                            if partial is None
                            else int(partial["state"]["iteration"])
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
                            "tau": tau,
                            "run": run_index,
                            "method": method_name,
                            "trace": trace,
                            "metrics": problem.metrics(trace.final_x),
                        }
                    )
                    progress.finish_run(
                        status,
                        int(trace.samples[-1]),
                        len(trace.samples) - 1,
                    )
    finally:
        progress_summary = progress.close()

    write_classification_outputs(
        results,
        methods,
        simulations,
        int(experiment.get("figure_run", 0)),
        output_dir,
        plot_methods=plot_methods,
    )
    save_config(config, output_dir / "config.yaml")
    cache.mark_complete()
    print(
        f"Finished: {progress_summary.computed} computed, "
        f"{progress_summary.resumed} resumed, "
        f"{progress_summary.cached} loaded from cache. Results are in {output_dir}"
    )
