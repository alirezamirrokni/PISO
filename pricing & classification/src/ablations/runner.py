from __future__ import annotations

import json
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .algorithm import run_piso_variant, stable_seed
from .cache import JobCache, canonical_hash
from src.problem import PricingProblem, ProblemSpec
from .report import build_suite_report
from .suites import VariantSpec, build_suite, suite_simulations


@dataclass(frozen=True)
class Task:
    project_root: str
    output_root: str
    suite_name: str
    variant: VariantSpec
    reference_id: str
    week_id: str
    run_index: int
    base_seed: int
    max_samples: int
    metric_samples: int
    checkpoint_every: int
    problem_values: dict[str, Any]
    source_fingerprint: str
    data_fingerprint: str
    compatible_source_fingerprints: tuple[str, ...] = ()


def _fingerprint_source_files(project_root: Path, names: list[str]) -> str:
    entries: list[tuple[str, str]] = []
    for name in names:
        path = project_root / "src" / name
        entries.append((name, canonical_hash(path.read_text(encoding="utf-8"))))
    return canonical_hash(entries)


def _source_fingerprint(project_root: Path) -> str:
    
    
    
    
    
    return _fingerprint_source_files(
        project_root,
        ["ablations/algorithm.py", "ablations/directions.py", "problem.py"],
    )


def _legacy_source_fingerprint(project_root: Path) -> str:
    
    
    
    return _fingerprint_source_files(
        project_root,
        ["ablations/algorithm.py", "ablations/directions.py", "problem.py", "ablations/config.py", "ablations/suites.py"],
    )


def _task_fingerprint(
    task: Task,
    source_fingerprint: str | None = None,
    *,
    include_data: bool = True,
) -> str:
    payload = {
        "source": (
            task.source_fingerprint
            if source_fingerprint is None
            else source_fingerprint
        ),
        "suite": task.suite_name,
        "variant_id": task.variant.variant_id,
        "params": task.variant.params,
        "week": task.week_id,
        "run": task.run_index,
        "base_seed": task.base_seed,
        "max_samples": task.max_samples,
        "metric_samples": task.metric_samples,
        "problem": task.problem_values,
    }
    if include_data:
        payload["data"] = task.data_fingerprint
    return canonical_hash(payload)


def _job_cache(task: Task) -> JobCache:
    root = (
        Path(task.output_root)
        / task.suite_name
        / "cache"
        / task.variant.variant_id
        / task.week_id
        / f"run_{task.run_index:04d}"
    )
    
    
    
    compatible = [
        _task_fingerprint(
            task,
            source_fingerprint=value,
            include_data=False,
        )
        for value in task.compatible_source_fingerprints
    ]
    return JobCache(root, _task_fingerprint(task), compatible)


def _load_task_final(task: Task) -> dict[str, Any] | None:
    saved = _job_cache(task).load_final()
    if not isinstance(saved, dict):
        return None
    trace = saved.get("trace")
    variant = saved.get("variant")
    if not isinstance(trace, dict) or "failed" not in trace:
        return None
    if not isinstance(variant, dict):
        return None
    if variant.get("variant_id") != task.variant.variant_id:
        return None
    if str(saved.get("week_id")) != task.week_id:
        return None
    if saved.get("run_index") != task.run_index:
        return None
    return saved


def _run_task(task: Task) -> dict[str, Any]:
    cache = _job_cache(task)
    saved = _load_task_final(task)
    if saved is not None:
        return {
            "suite": task.suite_name,
            "variant_id": task.variant.variant_id,
            "week_id": task.week_id,
            "run_index": task.run_index,
            "status": "cached",
            "failed": bool(saved["trace"]["failed"]),
        }

    spec = ProblemSpec.from_mapping(task.problem_values)
    instance_rng = np.random.RandomState(
        stable_seed(task.base_seed, "instance", task.week_id, task.run_index)
    )
    rho = spec.rho_low + (spec.rho_high - spec.rho_low) * instance_rng.rand(spec.products)
    problem = PricingProblem.from_week_with_rho(
        Path(task.project_root) / "data",
        task.week_id,
        rho,
        spec,
    )

    trace = run_piso_variant(
        problem,
        task.variant.params,
        base_seed=task.base_seed,
        week_id=task.week_id,
        run_index=task.run_index,
        
        
        variant_seed_key=task.suite_name,
        max_samples=task.max_samples,
        metric_samples=task.metric_samples,
        checkpoint_every=task.checkpoint_every,
        load_checkpoint=cache.load_progress,
        save_checkpoint=cache.save_progress,
        clear_checkpoint=cache.clear_progress,
    )
    cache.save_final(
        {
            "trace": trace.to_dict(),
            "variant": {
                "suite": task.variant.suite,
                "variant_id": task.variant.variant_id,
                "label": task.variant.label,
                "params": task.variant.params,
                "metadata": task.variant.metadata,
            },
            "week_id": task.week_id,
            "run_index": task.run_index,
            "reference_id": task.reference_id,
        }
    )
    return {
        "suite": task.suite_name,
        "variant_id": task.variant.variant_id,
        "week_id": task.week_id,
        "run_index": task.run_index,
        "status": "computed",
        "failed": trace.failed,
    }


def _collect_suite_records(
    output_root: Path,
    suite_name: str,
    variants: list[VariantSpec],
    weeks: list[str],
    simulations: int,
    tasks: list[Task],
) -> list[dict[str, Any]]:
    by_key = {
        (task.variant.variant_id, task.week_id, task.run_index): task for task in tasks
    }
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for variant in variants:
        for week_id in weeks:
            for run_index in range(simulations):
                task = by_key[(variant.variant_id, week_id, run_index)]
                saved = _load_task_final(task)
                if saved is None:
                    missing.append(f"{variant.variant_id}/{week_id}/{run_index}")
                else:
                    records.append(saved)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"suite {suite_name} is incomplete; {len(missing)} final caches are missing "
            f"(first: {preview})"
        )
    return records


def run_ablation(
    config: dict[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    jobs: int,
    reset_cache: bool = False,
    plot_only: bool = False,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if reset_cache and not plot_only:
        for suite_name in config["active_suites"]:
            shutil.rmtree(output_root / suite_name / "cache", ignore_errors=True)

    source_hash = _source_fingerprint(project_root)
    legacy_source_hash = _legacy_source_fingerprint(project_root)
    compatible_source_hashes = (legacy_source_hash,)
    experiment = config["experiment"]
    weeks = [str(value).zfill(2) for value in experiment["weeks"]]
    week_data_hashes = {
        week_id: canonical_hash(
            (project_root / "data" / "prices" / f"2022_{week_id}.csv").read_text(
                encoding="utf-8-sig"
            )
        )
        for week_id in weeks
    }

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "source_fingerprint": source_hash,
        "active_suites": list(config["active_suites"]),
        "experiment": experiment,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    completed_summary_paths: list[Path] = []
    for suite_name in config["active_suites"]:
        variants, reference_id = build_suite(config, suite_name)
        simulations = suite_simulations(config, suite_name)
        tasks = [
            Task(
                project_root=str(project_root),
                output_root=str(output_root),
                suite_name=suite_name,
                variant=variant,
                reference_id=reference_id,
                week_id=week_id,
                run_index=run_index,
                base_seed=int(experiment["seed"]),
                max_samples=int(experiment["max_samples"]),
                metric_samples=int(experiment["metric_samples"]),
                checkpoint_every=int(experiment.get("checkpoint_every", 1)),
                problem_values=dict(config["problem"]),
                source_fingerprint=source_hash,
                data_fingerprint=week_data_hashes[week_id],
                compatible_source_fingerprints=compatible_source_hashes,
            )
            for variant in variants
            for week_id in weeks
            for run_index in range(simulations)
        ]

        if not plot_only:
            
            
            
            pending_tasks: list[Task] = []
            cached = 0
            failures = 0
            for task in tasks:
                saved = _load_task_final(task)
                if saved is None:
                    pending_tasks.append(task)
                else:
                    cached += 1
                    failures += bool(saved["trace"]["failed"])

            computed = 0
            description = f"{suite_name} ablation"
            if pending_tasks:
                if jobs <= 1:
                    iterator = (_run_task(task) for task in pending_tasks)
                    with tqdm(
                        total=len(pending_tasks),
                        desc=description,
                        unit="run",
                    ) as bar:
                        for result in iterator:
                            computed += result["status"] == "computed"
                            cached += result["status"] == "cached"
                            failures += bool(result["failed"])
                            bar.set_postfix(
                                computed=computed,
                                cached=cached,
                                failed=failures,
                            )
                            bar.update(1)
                else:
                    with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
                        future_to_task = {
                            executor.submit(_run_task, task): task
                            for task in pending_tasks
                        }
                        with tqdm(
                            total=len(pending_tasks),
                            desc=description,
                            unit="run",
                        ) as bar:
                            for future in as_completed(future_to_task):
                                result = future.result()
                                computed += result["status"] == "computed"
                                cached += result["status"] == "cached"
                                failures += bool(result["failed"])
                                bar.set_postfix(
                                    computed=computed,
                                    cached=cached,
                                    failed=failures,
                                )
                                bar.update(1)
            else:
                print(
                    f"{suite_name}: all {cached} runs already cached; "
                    "no optimization runs needed"
                )

            if pending_tasks:
                print(
                    f"{suite_name}: {computed} computed, {cached} cached, "
                    f"{failures} failed runs"
                )

        records = _collect_suite_records(
            output_root,
            suite_name,
            variants,
            weeks,
            simulations,
            tasks,
        )
        suite_report_config = dict(config["suites"][suite_name])
        plotting = dict(config.get("plotting", {}))
        plotting.update(dict(suite_report_config.get("plotting", {})))
        suite_report_config["plotting"] = plotting
        build_suite_report(
            suite_name=suite_name,
            suite_config=suite_report_config,
            records=records,
            variants=variants,
            reference_id=reference_id,
            output_dir=output_root / suite_name,
            max_samples=int(experiment["max_samples"]),
            grid_points=int(experiment["report_grid_points"]),
            bootstrap_resamples=int(experiment.get("bootstrap_resamples", 0)),
            permutation_resamples=int(experiment.get("permutation_resamples", 0)),
            base_seed=int(experiment["seed"]),
        )
        completed_summary_paths.append(output_root / suite_name / "summary.csv")

    if completed_summary_paths:
        combined = pd.concat(
            [pd.read_csv(path) for path in completed_summary_paths],
            ignore_index=True,
            sort=False,
        )
        combined.to_csv(output_root / "all_suites_summary.csv", index=False)
