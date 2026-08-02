from __future__ import annotations

from pathlib import Path


def run_experiment(
    problem: str,
    config_path: Path,
    output_dir: Path,
    reset_cache: bool = False,
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
    raise ValueError("--problem must be either 'pricing' or 'classification'")
