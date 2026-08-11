from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.config import apply_profile, load_config
from src.runner import run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistically paired PISO ablations on the pricing benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/piso_pricing_ablation.yaml"),
    )
    parser.add_argument(
        "--profile",
        default="smoke",
        help="smoke or paper",
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Run one suite; repeat this option to select several suites.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--list-suites", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    if args.list_suites:
        for name, values in config["suites"].items():
            print(f"{name:24s} {values['kind']:8s} {values['protocol']}")
        return

    resolved = apply_profile(
        config,
        args.profile,
        suites_override=args.suites,
        simulations_override=args.simulations,
    )
    output = (
        args.output
        if args.output is not None
        else project_root / "results" / str(args.profile)
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)

    run_ablation(
        resolved,
        project_root,
        output,
        jobs=max(int(args.jobs), 1),
        reset_cache=bool(args.reset_cache),
        plot_only=bool(args.plot_only),
    )


if __name__ == "__main__":
    main()
