from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from ddp_pricing.config import load_yaml, save_yaml
from ddp_pricing.loading import load_factory
from ddp_pricing.problem import PricingProblem, ProblemSpec
from ddp_pricing.reporting import summarize, write_figure, write_latex
from ddp_pricing.types import RunContext


PACKAGE_NAMES = ["numpy", "scipy", "pandas", "matplotlib", "PyYAML"]


def _environment() -> dict:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def run_experiment(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    data_dir = root / "data"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    week_labels = load_yaml(data_dir / "weeks.yaml")
    experiment = config["experiment"]
    problem_spec = ProblemSpec.from_mapping(config["problem"])
    method_order = list(experiment["method_order"])
    methods = {}
    for method_name in method_order:
        method_config = config["methods"][method_name]
        factory = load_factory(method_config["factory"])
        methods[method_name] = factory(dict(method_config["params"]))

    rng = np.random.RandomState(int(experiment["seed"]))
    trajectory_rows: list[dict] = []
    final_rows: list[dict] = []

    for week_id_value in experiment["weeks"]:
        week_id = str(week_id_value).zfill(2)
        week_label = str(week_labels[week_id])
        for run_index in range(int(experiment["simulations"])):
            problem = PricingProblem.from_week(data_dir, week_id, rng, problem_spec)
            context = RunContext(
                week_id=week_id,
                week_label=week_label,
                run_index=run_index,
                max_samples=int(experiment["max_samples"]),
                metric_samples=int(experiment["metric_samples"]),
            )
            for method_name in method_order:
                trace = methods[method_name].run(problem=problem, rng=rng, run_context=context)
                for step_index, (sample_count, objective) in enumerate(
                    zip(trace.sample_counts, trace.objectives, strict=True)
                ):
                    trajectory_rows.append(
                        {
                            "week_id": week_id,
                            "week_label": week_label,
                            "run_index": run_index,
                            "method": method_name,
                            "step_index": step_index,
                            "sample_count": sample_count,
                            "objective": objective,
                        }
                    )
                final_rows.append(
                    {
                        "week_id": week_id,
                        "week_label": week_label,
                        "run_index": run_index,
                        "method": method_name,
                        "objective": trace.objectives[-1],
                        "final_samples": trace.sample_counts[-1],
                        "iterations": trace.metadata.get("iterations"),
                        "final_x": json.dumps(trace.final_x.tolist()),
                    }
                )

    trajectories = pd.DataFrame(trajectory_rows)
    final_runs = pd.DataFrame(final_rows)
    summary = summarize(final_runs, method_order)

    trajectories_path = output_dir / "trajectories.csv"
    final_runs_path = output_dir / "final_runs.csv"
    summary_path = output_dir / "summary.csv"
    latex_path = output_dir / "summary.tex"
    png_path = output_dir / "figure.png"
    pdf_path = output_dir / "figure.pdf"
    resolved_path = output_dir / "resolved_config.yaml"
    environment_path = output_dir / "environment.json"

    trajectories.to_csv(trajectories_path, index=False)
    final_runs.to_csv(final_runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_latex(summary, method_order, latex_path)
    write_figure(
        trajectories=trajectories,
        method_order=method_order,
        figure_run_index=int(experiment["figure_run_index"]),
        output_png=png_path,
        output_pdf=pdf_path,
    )
    save_yaml(config, resolved_path)
    environment_path.write_text(json.dumps(_environment(), indent=2), encoding="utf-8")

    return {
        "summary": summary_path,
        "latex": latex_path,
        "trajectories": trajectories_path,
        "final_runs": final_runs_path,
        "figure_png": png_path,
        "figure_pdf": pdf_path,
        "config": resolved_path,
        "environment": environment_path,
    }
