from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from src.piso_rl.report import _plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate PISO RL figures from figure_data.csv"
    )
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "results" / "piso_v4_heldout"))
    parser.add_argument(
        "--config",
        default=None,
        help="Plot configuration. Defaults to <output>/config.yaml when present.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Override report.smoothing_window from the configuration.",
    )
    parser.add_argument(
        "--show-raw-curves",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Overlay thin unsmoothed means for debugging. Disabled by default "
            "because they can look like hollow duplicate curves."
        ),
    )
    args = parser.parse_args()

    output = Path(args.output)
    data_path = output / "figure_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"missing {data_path}")

    config_path = Path(args.config) if args.config else output / "config.yaml"
    report_config: dict = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        report_config = config.get("report", {})
        if not isinstance(report_config, dict):
            raise ValueError(f"report must be a mapping in {config_path}")

    smoothing_window = (
        int(args.smoothing_window)
        if args.smoothing_window is not None
        else int(report_config.get("smoothing_window", 5))
    )
    plotting_methods = report_config.get("plotting_methods")
    total_iterations = int(report_config.get("total_iterations", 1000))
    show_raw_curves = (
        bool(args.show_raw_curves)
        if args.show_raw_curves is not None
        else bool(report_config.get("show_raw_curves", False))
    )
    if plotting_methods is not None and not isinstance(plotting_methods, list):
        raise ValueError(
            f"report.plotting_methods must be a list in {config_path}"
        )

    rows: list[dict] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            converted["samples"] = int(row["samples"])
            for key, value in list(row.items()):
                if key in {"method", "samples", "n_runs"}:
                    continue
                converted[key] = float(value)
            converted["n_runs"] = int(row["n_runs"])
            rows.append(converted)

    _plot(
        rows,
        output,
        smoothing_window=max(1, smoothing_window),
        plotting_methods=plotting_methods,
        show_raw_curves=show_raw_curves,
        total_iterations=total_iterations,
    )
    print(f"Regenerated figures in {output}")


if __name__ == "__main__":
    main()
