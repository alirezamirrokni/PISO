from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.cache import CacheManager, build_fingerprint
from src.config import load_config, save_config
from src.methods import METHODS
from src.problem import PricingProblem, ProblemSpec
from src.progress import ExperimentProgress
from src.report import write_outputs


def _private_method_seed(
    base_seed: int,
    week_id: str,
    run_index: int,
    method: str,
    variant: str,
) -> int:
    payload = f"{base_seed}:{week_id}:{run_index}:{method}:{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _method_variants(
    method: str,
    params: dict[str, Any],
) -> list[tuple[str, str, float | None, dict[str, Any]]]:
    if method not in {"PISO", "PISO_M"}:
        return [("", "default", None, dict(params))]

    values = [float(value) for value in params["residual_alphas"]]
    if not values:
        raise ValueError(f"{method} residual_alphas cannot be empty")
    variants = []
    for alpha in values:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"{method} residual alpha must satisfy 0 < alpha <= 1")
        variant_params = dict(params)
        variant_params.pop("residual_alphas", None)
        variant_params["residual_alpha"] = alpha
        variant_key = f"alpha={alpha:.12g}"
        variants.append((variant_key, f"alpha={alpha:g}", alpha, variant_params))
    return variants


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
        output_dir / "selected_alphas.csv",
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
    base_seed = int(experiment["seed"])
    rng = np.random.RandomState(base_seed)
    simulations = int(experiment["simulations"])
    total_pairs = len(experiment["weeks"]) * len(method_names)
    total_groups = total_pairs * simulations
    cache_files = total_pairs
    results: list[dict[str, Any]] = []

    print(
        f"Running {total_pairs} dataset-method pairs across {simulations} simulations "
        f"({total_groups} method runs). Checkpoints use {cache_files} CSV files "
        f"in {cache.root}."
    )
    progress = ExperimentProgress(total_pairs, simulations, context.max_samples)
    group_number = 0
    try:
        # This order intentionally preserves the released baseline random-number
        # stream: week -> simulation -> method. PISO variants use private RNGs,
        # so their sweep does not perturb any baseline method.
        for week_value in experiment["weeks"]:
            week_id = str(week_value).zfill(2)
            for run_index in range(simulations):
                problem = PricingProblem.from_week(project_root / "data", week_id, rng, spec)
                for method_name in method_names:
                    group_number += 1
                    variants = _method_variants(
                        method_name,
                        config["methods"][method_name],
                    )
                    group_label = (
                        f"{weeks[week_id]} | run {run_index + 1}/{simulations} | "
                        f"{method_name}"
                    )
                    progress.start_group(group_number, group_label, len(variants))

                    for variant_index, (
                        variant_key,
                        variant_label,
                        residual_alpha,
                        method_params,
                    ) in enumerate(variants, start=1):
                        method_class = METHODS[method_name]
                        if getattr(method_class, "private_rng", False):
                            job_rng = np.random.RandomState(
                                _private_method_seed(
                                    base_seed,
                                    week_id,
                                    run_index,
                                    method_name,
                                    variant_key,
                                )
                            )
                        else:
                            job_rng = rng

                        job_cache = cache.job(
                            week_id,
                            run_index,
                            method_name,
                            variant_key,
                        )
                        saved = job_cache.load_final()
                        if saved is not None:
                            trace = saved["trace"]
                            job_rng.set_state(saved["rng_state"])
                            progress.start_variant(
                                variant_index,
                                variant_label,
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
                            handle = progress.start_variant(
                                variant_index,
                                variant_label,
                                initial_samples,
                                initial_iteration,
                                "resuming" if partial is not None else "starting",
                            )
                            trace = method_class(method_params).run(
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
                                "residual_alpha": residual_alpha,
                                "trace": trace,
                            }
                        )
                        progress.finish_variant(
                            status,
                            int(trace.samples[-1]),
                            len(trace.samples) - 1,
                        )
                    progress.finish_group()
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
