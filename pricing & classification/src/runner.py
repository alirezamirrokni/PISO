from __future__ import annotations

from pathlib import Path

import yaml


def run_experiment(
    problem: str,
    config_path: Path,
    output_dir: Path,
    reset_cache: bool = False,
    suites: list[str] | None = None,
    jobs: int = 1,
    simulations: int | None = None,
    plot_only: bool = False,
    list_suites: bool = False,
) -> None:
    problem_key = str(problem).strip().lower()
    if problem_key == "pricing":
        from src.pricing_runner import run_experiment as run_pricing
        run_pricing(config_path, output_dir, reset_cache=reset_cache)
        return
    if problem_key == "classification":
        from src.classification_runner import run_experiment as run_classification
        run_classification(config_path, output_dir, reset_cache=reset_cache)
        return
    if problem_key == "ablation":
        from src.ablations.config import load_config, resolve_config
        from src.ablations.runner import run_ablation
        config = load_config(config_path)
        if list_suites:
            for name, values in config["suites"].items():
                print(f"{name:24s} {values['kind']:16s}")
            return
        resolved = resolve_config(config, suites_override=suites, simulations_override=simulations)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)
        run_ablation(
            resolved,
            Path(__file__).resolve().parents[1],
            output_dir,
            jobs=max(int(jobs), 1),
            reset_cache=bool(reset_cache),
            plot_only=bool(plot_only),
        )
        return
    raise ValueError("--problem must be 'pricing', 'classification', or 'ablation'")
