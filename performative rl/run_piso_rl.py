from __future__ import annotations

import argparse

from src.piso_rl.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PISO/PISO² and trajectory-based baselines on PePG Gridworld."
    )
    parser.add_argument(
        "--config",
        default="configs/piso_rl.yaml",
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override experiment.output_dir.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset of configured methods.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel method/seed workers (overrides experiment.n_jobs).",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete matching run caches before execution.",
    )
    args = parser.parse_args()
    selected = None
    if args.methods:
        selected = [item.strip() for item in args.methods.split(",") if item.strip()]
    run_experiment(
        args.config,
        output_dir=args.output,
        selected_methods=selected,
        reset_cache=args.reset_cache,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
