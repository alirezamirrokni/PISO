from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from src.piso_rl.methods.common import Trace
from src.piso_rl.report import build_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate PISO RL plots from raw_runs.csv")
    parser.add_argument("--output", default="results/piso_rl_reference")
    args = parser.parse_args()
    output = Path(args.output)
    raw_path = output / "raw_runs.csv"
    config_path = output / "config.yaml"
    if not raw_path.exists() or not config_path.exists():
        raise FileNotFoundError("raw_runs.csv and config.yaml are required")

    grouped = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["seed"]))
            grouped.setdefault(key, {"samples": [], "values": [], "changes": []})
            grouped[key]["samples"].append(int(row["samples"]))
            grouped[key]["values"].append(float(row["performative_value"]))
            grouped[key]["changes"].append(float(row["occupancy_change"]))

    traces = []
    for (method, seed), values in grouped.items():
        traces.append(
            Trace(
                samples=values["samples"],
                values=values["values"],
                occupancy_changes=values["changes"],
                final_theta=[],
                method=method,
                seed=seed,
            )
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    build_reports(traces, config, output)
    print(f"Regenerated reports in {output}")


if __name__ == "__main__":
    main()
