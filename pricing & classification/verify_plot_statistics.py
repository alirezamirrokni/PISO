from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ATOL = 1.0e-12


def _assert_close(label: str, actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.allclose(actual, expected, rtol=0.0, atol=ATOL, equal_nan=True):
        raise AssertionError(f"{label}: maximum absolute error is {error:.3e}")
    return error


def verify(output_dir: Path) -> None:
    raw_path = output_dir / "figure_data_raw.csv"
    aggregate_path = output_dir / "figure_data.csv"
    selected_path = output_dir / "figure_data_selected_run.csv"

    raw = pd.read_csv(raw_path)
    aggregate = pd.read_csv(aggregate_path)
    selected = pd.read_csv(selected_path)

    required_raw = {"week_id", "week", "run", "method", "samples", "objective"}
    required_aggregate = {
        "week_id",
        "week",
        "method",
        "samples",
        "objective_mean",
        "objective_sd",
        "objective_se",
        "objective_lower",
        "objective_upper",
        "n_runs",
    }
    if missing := required_raw.difference(raw.columns):
        raise AssertionError(f"Missing raw columns: {sorted(missing)}")
    if missing := required_aggregate.difference(aggregate.columns):
        raise AssertionError(f"Missing aggregate columns: {sorted(missing)}")

    expected_runs = np.arange(1, int(raw["run"].max()) + 1, dtype=int)
    max_mean_error = 0.0
    max_sd_error = 0.0
    max_se_error = 0.0
    comparison_rows: list[dict[str, object]] = []

    groups = raw.groupby(["week_id", "week", "method"], sort=False)
    for (week_id, week, method), frame in groups:
        actual_runs = np.sort(frame["run"].unique().astype(int))
        if not np.array_equal(actual_runs, expected_runs):
            raise AssertionError(
                f"Incomplete runs for week={week_id}, method={method}: "
                f"expected {expected_runs.tolist()}, found {actual_runs.tolist()}"
            )

        sample_reference: np.ndarray | None = None
        objective_rows: list[np.ndarray] = []
        for run_number in expected_runs:
            run_frame = frame[frame["run"] == run_number].sort_values("samples")
            if run_frame["samples"].duplicated().any():
                raise AssertionError(
                    f"Duplicate samples for week={week_id}, method={method}, "
                    f"run={run_number}"
                )
            samples = run_frame["samples"].to_numpy(dtype=int)
            objectives = run_frame["objective"].to_numpy(dtype=float)
            if sample_reference is None:
                sample_reference = samples
            elif not np.array_equal(samples, sample_reference):
                raise AssertionError(
                    f"Sample-grid mismatch for week={week_id}, method={method}"
                )
            objective_rows.append(objectives)

        assert sample_reference is not None
        matrix = np.vstack(objective_rows)
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
        se = sd / np.sqrt(matrix.shape[0])

        plotted = aggregate[
            (aggregate["week_id"] == week_id)
            & (aggregate["week"] == week)
            & (aggregate["method"] == method)
        ].sort_values("samples")

        if not np.array_equal(
            plotted["samples"].to_numpy(dtype=int),
            sample_reference,
        ):
            raise AssertionError(
                f"Aggregate sample-grid mismatch for week={week_id}, method={method}"
            )
        if not np.all(plotted["n_runs"].to_numpy(dtype=int) == matrix.shape[0]):
            raise AssertionError(
                f"Incorrect run count for week={week_id}, method={method}"
            )

        max_mean_error = max(
            max_mean_error,
            _assert_close(
                f"mean: week={week_id}, method={method}",
                plotted["objective_mean"].to_numpy(dtype=float),
                mean,
            ),
        )
        max_sd_error = max(
            max_sd_error,
            _assert_close(
                f"SD: week={week_id}, method={method}",
                plotted["objective_sd"].to_numpy(dtype=float),
                sd,
            ),
        )
        max_se_error = max(
            max_se_error,
            _assert_close(
                f"SE: week={week_id}, method={method}",
                plotted["objective_se"].to_numpy(dtype=float),
                se,
            ),
        )
        _assert_close(
            f"lower bound: week={week_id}, method={method}",
            plotted["objective_lower"].to_numpy(dtype=float),
            mean - se,
        )
        _assert_close(
            f"upper bound: week={week_id}, method={method}",
            plotted["objective_upper"].to_numpy(dtype=float),
            mean + se,
        )

        selected_group = selected[
            (selected["week_id"] == week_id)
            & (selected["week"] == week)
            & (selected["method"] == method)
        ].sort_values("samples")
        if not selected_group.empty:
            selected_objective = selected_group["objective"].to_numpy(dtype=float)
            if selected_objective.shape == mean.shape:
                comparison_rows.append(
                    {
                        "week": week,
                        "method": method,
                        "max_abs_selected_run_minus_mean": float(
                            np.max(np.abs(selected_objective - mean))
                        ),
                        "final_selected_run": float(selected_objective[-1]),
                        "final_mean": float(mean[-1]),
                    }
                )

    print("Plot-statistics verification passed.")
    print(f"Runs per week/method: {len(expected_runs)}")
    print(f"Maximum mean error: {max_mean_error:.3e}")
    print(f"Maximum SD error:   {max_sd_error:.3e}")
    print(f"Maximum SE error:   {max_se_error:.3e}")

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(
            output_dir / "selected_run_vs_mean.csv",
            index=False,
        )
        largest = comparison.sort_values(
            "max_abs_selected_run_minus_mean",
            ascending=False,
        ).head(10)
        print("\nLargest changes from the old selected-run curve to the mean curve:")
        print(largest.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that plotted means and standard errors exactly match raw runs."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output.resolve())


if __name__ == "__main__":
    main()
