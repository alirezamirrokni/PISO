import argparse
import importlib
from pathlib import Path

import yaml

from src.common import load_config, save_config

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "pricing": "src.pricing.experiment",
    "classification": "src.classification.experiment",
    "ablations": "src.pricing.ablations",
    "routing": "src.bilevel.experiment",
    "security": "src.bilevel.experiment",
    "rl": "src.rl.experiment",
}


def main():
    parser = argparse.ArgumentParser(description="Run one PISO experiment and write its tables and figures.")
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--config", type=Path, help="Default: configs/<experiment>.yaml")
    parser.add_argument("--output", type=Path, help="Default: results/<experiment>")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (ablations, routing, security, rl)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config entries, e.g. experiment.simulations=5")
    args = parser.parse_args()
    config = load_config(args.config or ROOT / "configs" / f"{args.experiment}.yaml")
    for item in args.set:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = config
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)
    output = (args.output or ROOT / "results" / args.experiment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    importlib.import_module(EXPERIMENTS[args.experiment]).run(config, output, jobs=args.jobs)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
