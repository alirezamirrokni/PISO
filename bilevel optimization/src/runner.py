from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator

from src.cache import CacheManager
from src.config import load_config, save_config
from src.methods import METHODS
from src.methods.common import MethodTrace
from src.problems import (
    RoutingInstance,
    SecurityInstance,
    generate_routing_instances,
    generate_security_instances,
)
from src.report import write_outputs

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  
    _tqdm = None


class _FallbackProgress:


    def __init__(self, total: int, desc: str, unit: str = "run") -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.completed = 0
        self.postfix = ""
        self.started_at = time.monotonic()
        self._last_render_length = 0
        self._render()

    def __enter__(self) -> "_FallbackProgress":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        self.postfix = text
        if refresh:
            self._render()

    def update(self, amount: int = 1) -> None:
        self.completed = min(self.total, self.completed + int(amount))
        self._render()

    def close(self) -> None:
        self._render(final=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, final: bool = False) -> None:
        if self.total <= 0:
            fraction = 1.0
        else:
            fraction = self.completed / self.total

        width = 28
        filled = min(width, int(round(width * fraction)))
        bar = "█" * filled + "-" * (width - filled)
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        rate = self.completed / elapsed
        percent = 100.0 * fraction
        suffix = f" | {self.postfix}" if self.postfix else ""
        line = (
            f"\r{self.desc}: |{bar}| {self.completed}/{self.total} "
            f"{self.unit}s [{percent:6.2f}%] {rate:5.2f} {self.unit}/s{suffix}"
        )
        if final and self.completed < self.total:
            line += " (stopped)"
        padding = " " * max(0, self._last_render_length - len(line))
        sys.stdout.write(line + padding)
        sys.stdout.flush()
        self._last_render_length = len(line)


def _progress_bar(total: int, desc: str):
    if _tqdm is not None:
        return _tqdm(
            total=total,
            desc=desc,
            unit="run",
            dynamic_ncols=True,
            smoothing=0.1,
            leave=True,
        )
    return _FallbackProgress(total=total, desc=desc, unit="run")


def _family_components(family: str):
    if family == "routing":
        return generate_routing_instances, RoutingInstance
    if family == "security":
        return generate_security_instances, SecurityInstance
    raise ValueError("family must be 'routing' or 'security'")


def _run_one_method(
    task: tuple[str, dict[str, Any], str, int, str],
) -> tuple[int, str, bool]:







    family, config, output_text, instance_id, method_name = task
    output_dir = Path(output_text)
    generator, problem_class = _family_components(family)
    cache = CacheManager(output_dir, family, config)
    fingerprint = cache.instance_fingerprint(generator)
    instances = cache.load_instances(fingerprint)
    if instances is None:
        raise RuntimeError("instance cache disappeared while workers were running")
    problem = instances[instance_id]

    experiment = config["experiment"]
    iterations = int(experiment["iterations"])
    checkpoint_interval = int(experiment.get("checkpoint_interval", 5))
    base_seed = int(experiment["seed"])

    method_class = METHODS[method_name]
    method_fingerprint = cache.method_fingerprint(
        method_name,
        method_class,
        problem_class,
        instance_id,
    )
    saved = cache.load_method(method_name, instance_id, method_fingerprint)
    if saved is not None and saved.get("complete", False):
        return instance_id, method_name, True

    restored = None if saved is None else saved["state"]
    method = method_class(config["methods"][method_name])

    def checkpoint(state: dict[str, Any], complete: bool) -> None:
        cache.save_method(
            method_name,
            instance_id,
            method_fingerprint,
            state,
            complete=complete,
        )

    method.run(
        problem,
        instance_id,
        base_seed,
        iterations,
        checkpoint_interval,
        restored,
        checkpoint,
    )
    return instance_id, method_name, False


def _load_final_traces(
    family: str,
    config: dict[str, Any],
    output_dir: Path,
    instances: list[Any],
    method_names: list[str],
) -> list[MethodTrace]:
    _, problem_class = _family_components(family)
    cache = CacheManager(output_dir, family, config)
    traces: list[MethodTrace] = []
    for instance_id, problem in enumerate(instances):
        for method_name in method_names:
            method_class = METHODS[method_name]
            fingerprint = cache.method_fingerprint(
                method_name,
                method_class,
                problem_class,
                instance_id,
            )
            payload = cache.load_method(method_name, instance_id, fingerprint)
            if payload is None or not payload.get("complete", False):
                raise RuntimeError(
                    f"missing complete cache for {method_name}, instance {instance_id}"
                )
            state = payload["state"]
            traces.append(
                MethodTrace(
                    method=method_name,
                    instance_id=instance_id,
                    dimension=int(problem.dimension),
                    objectives=__import__("numpy").asarray(
                        state["objectives"], dtype=float
                    ),
                    oracle_calls=__import__("numpy").asarray(
                        state["oracle_calls"], dtype=int
                    ),
                )
            )
    return traces


def _execute_tasks(
    tasks: list[tuple[str, dict[str, Any], str, int, str]],
    jobs: int,
    family: str,
) -> None:
    total = len(tasks)
    cache_hits = 0
    computed = 0
    description = f"{family.capitalize()} experiments"

    with _progress_bar(total=total, desc=description) as progress:
        if jobs <= 1:
            for task in tasks:
                instance_id, method_name, from_cache = _run_one_method(task)
                cache_hits += int(from_cache)
                computed += int(not from_cache)
                progress.set_postfix_str(
                    f"{method_name} #{instance_id:04d} | "
                    f"computed {computed}, cached {cache_hits}",
                    refresh=False,
                )
                progress.update(1)
            return

        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_task = {
                executor.submit(_run_one_method, task): task for task in tasks
            }
            try:
                for future in as_completed(future_to_task):
                    instance_id, method_name, from_cache = future.result()
                    cache_hits += int(from_cache)
                    computed += int(not from_cache)
                    progress.set_postfix_str(
                        f"{method_name} #{instance_id:04d} | "
                        f"computed {computed}, cached {cache_hits}",
                        refresh=False,
                    )
                    progress.update(1)
            except BaseException:
                for future in future_to_task:
                    future.cancel()
                raise


def run_experiment(
    family: str,
    config_path: Path,
    output_override: Path | None = None,
    *,
    reset_cache: bool = False,
    plot_only: bool = False,
    jobs_override: int | None = None,
    methods_override: list[str] | None = None,
) -> None:
    family = family.lower().strip()
    config = load_config(config_path)
    configured_family = str(config.get("family", family)).lower()
    if configured_family != family:
        raise ValueError(
            f"config family is {configured_family!r}, but --problem is {family!r}"
        )

    output_candidate = (output_override or Path(config["output_dir"])).expanduser()
    if output_candidate.is_absolute():
        output_dir = output_candidate.resolve()
    else:
        output_dir = (Path(__file__).resolve().parents[2] / output_candidate).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = CacheManager(output_dir, family, config)
    if reset_cache:
        cache.clear_all_methods()
        for name in [
            "raw_runs.csv",
            "normalized_runs.csv",
            "trajectory_summary.csv",
            "final_scores.csv",
        ]:
            (output_dir / name).unlink(missing_ok=True)
        for pattern in (
            "routing_*.png",
            "routing_*.pdf",
            "routing_*.csv",
            "security_*.png",
            "security_*.pdf",
            "security_*.csv",
        ):
            for path in output_dir.glob(pattern):
                path.unlink(missing_ok=True)

    method_names = list(methods_override or config["methods"].keys())
    unknown = [name for name in method_names if name not in METHODS]
    if unknown:
        raise KeyError(
            f"unknown methods {unknown}; available methods are {sorted(METHODS)}"
        )

    generator, _ = _family_components(family)
    instance_fingerprint = cache.instance_fingerprint(generator)
    instances = cache.load_instances(instance_fingerprint)
    if instances is None:
        print(
            f"Generating {config['experiment']['instances']} deterministic "
            f"{family} instances..."
        )
        instances = generator(config, int(config["experiment"]["seed"]))
        cache.save_instances(instance_fingerprint, instances)
    else:
        print(f"Loaded {len(instances)} cached {family} instances.")

    if family == "security":
        dimension_min = int(
            config.get("plotting", {}).get("dimension_plot_min", 100)
        )
        high_dimensional = sum(
            int(instance.dimension) >= dimension_min for instance in instances
        )
        print(f"High-dimensional security instances used by the dimension plot: {high_dimensional}")

    if not plot_only:
        jobs = max(1, int(jobs_override or config["experiment"].get("jobs", 1)))
        tasks = [
            (family, config, str(output_dir), instance_id, method_name)
            for instance_id in range(len(instances))
            for method_name in method_names
        ]
        if jobs > 1:
            print(
                f"Running {len(tasks)} method-instance runs with {jobs} worker "
                "processes. Ctrl+C is safe; in-flight checkpoints remain usable."
            )
        _execute_tasks(tasks, jobs, family)

    traces = _load_final_traces(
        family,
        config,
        output_dir,
        instances,
        method_names,
    )
    write_outputs(family, traces, config, output_dir)
    save_config(config, output_dir / "config.yaml")
    print(f"Finished. Results are in: {output_dir}")