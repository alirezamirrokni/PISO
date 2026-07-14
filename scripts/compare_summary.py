from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "benchmark_reference.csv",
    )
    args = parser.parse_args()

    actual = pd.read_csv(args.summary, dtype={"week_id": str})
    expected = pd.read_csv(args.reference, dtype={"week_id": str})
    joined = expected.merge(
        actual[["week_id", "method", "obj", "sd"]],
        on=["week_id", "method"],
        suffixes=("_reference", "_actual"),
        validate="one_to_one",
    )
    joined["obj_abs_error"] = (joined["obj_actual"] - joined["obj_reference"]).abs()
    joined["sd_abs_error"] = (joined["sd_actual"] - joined["sd_reference"]).abs()
    print(joined.to_string(index=False))
    print("\nmaximum objective error:", joined["obj_abs_error"].max())
    print("maximum standard-deviation error:", joined["sd_abs_error"].max())


if __name__ == "__main__":
    main()
