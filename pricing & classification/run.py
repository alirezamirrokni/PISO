from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import run_experiment


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pricing, classification, or pricing-ablation experiments.")
    parser.add_argument("--problem", choices=("pricing", "classification", "ablation"), required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--suite", action="append", dest="suites")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--list-suites", action="store_true")
    args = parser.parse_args()

    default_configs = {
        "pricing": PROJECT_DIR / "configs" / "pricing.yaml",
        "classification": PROJECT_DIR / "configs" / "classification.yaml",
        "ablation": PROJECT_DIR / "configs" / "ablations.yaml",
    }
    default_outputs = {
        "pricing": REPO_ROOT / "results" / "pricing",
        "classification": REPO_ROOT / "results" / "classification",
        "ablation": REPO_ROOT / "results" / "ablations",
    }
    config = args.config or default_configs[args.problem]
    output = args.output or default_outputs[args.problem]
    if args.problem != "ablation" and (args.suites or args.simulations is not None or args.plot_only or args.list_suites):
        parser.error("--suite, --simulations, --plot-only, and --list-suites are only valid with --problem ablation")

    run_experiment(
        args.problem,
        config,
        output,
        reset_cache=args.reset_cache,
        suites=args.suites,
        jobs=args.jobs,
        simulations=args.simulations,
        plot_only=args.plot_only,
        list_suites=args.list_suites,
    )


if __name__ == "__main__":
    main()
