from __future__ import annotations

import argparse

from src.piso_rl.runner import run_experiment


def _parse_seeds(value: str | None) -> list[int] | None:
    if value is None:
        return None
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PISO/PISO², access-matched model-free baselines, sampled "
            "PePG, and the privileged exact oracle on the performative-RL gridworld."
        )
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
        "--seeds",
        default=None,
        help="Comma-separated seed IDs, for example 0,1,2.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Override the real environment trajectory budget per method/seed.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=None,
        help="Override exact-evaluation interval in real trajectories.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save resumable progress every N optimizer updates.",
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
        seeds=_parse_seeds(args.seeds),
        max_trajectories=args.max_trajectories,
        evaluation_interval=args.evaluation_interval,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
