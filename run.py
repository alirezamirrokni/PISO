from __future__ import annotations

import argparse
from pathlib import Path

from piso.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pricing experiments.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="discard checkpoints in the output directory before running",
    )
    args = parser.parse_args()
    run_experiment(args.config, args.output, reset_cache=args.reset_cache)


if __name__ == "__main__":
    main()
