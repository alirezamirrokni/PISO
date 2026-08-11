from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run black-box-follower bilevel experiments with ZO, GZO, PZOS, PISO, and PISO².")
    parser.add_argument("--problem", choices=("routing", "security"), required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    config = args.config or PROJECT_ROOT / "configs" / f"{args.problem}.yaml"
    methods = None if args.methods is None else [part.strip() for part in args.methods.split(",") if part.strip()]
    run_experiment(
        args.problem,
        config,
        args.output,
        reset_cache=args.reset_cache,
        plot_only=args.plot_only,
        jobs_override=args.jobs,
        methods_override=methods,
    )


if __name__ == "__main__":
    main()
