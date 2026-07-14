from __future__ import annotations

import argparse
from pathlib import Path

from ddp_pricing.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddp-pricing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the benchmark")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        paths = run_experiment(args.config, args.output)
        for name, path in paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
