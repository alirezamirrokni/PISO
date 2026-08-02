from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.report import DISPLAY_NAMES, STYLES

_METRICS = (
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_auc",
    "test_accuracy",
)


def _mean_sd_se(values: np.ndarray) -> tuple[float, float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(finite.mean())
    if count == 1:
        return mean, 0.0, 0.0, 1
    sd = float(finite.std(ddof=1))
    return mean, sd, float(sd / np.sqrt(count)), count


def _final_scores(results: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in results:
        row: dict[str, object] = {
            "tau": float(item["tau"]),
            "run": int(item["run"]) + 1,
            "method": item["method"],
            "samples": int(item["trace"].samples[-1]),
        }
        row.update({key: float(value) for key, value in item["metrics"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(final_scores: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tau in sorted(final_scores["tau"].unique()):
        frame = final_scores[final_scores["tau"] == tau]
        for method in methods:
            selected = frame[frame["method"] == method]
            row: dict[str, object] = {"tau": float(tau), "method": method}
            for metric in _METRICS:
                mean, sd, se, count = _mean_sd_se(
                    selected[metric].to_numpy(dtype=float)
                )
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_se"] = se
                row[f"{metric}_n"] = count
            rows.append(row)
    return pd.DataFrame(rows)


def _raw_trajectories(results: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in results:
        trace = item["trace"]
        for step, (samples, objective) in enumerate(
            zip(trace.samples, trace.objectives, strict=True)
        ):
            rows.append(
                {
                    "tau": float(item["tau"]),
                    "run": int(item["run"]) + 1,
                    "method": item["method"],
                    "step": step,
                    "samples": int(samples),
                    "train_loss": float(objective),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_trajectories(
    raw: pd.DataFrame,
    simulations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_runs = np.arange(1, int(simulations) + 1, dtype=int)
    for (tau, method), frame in raw.groupby(["tau", "method"], sort=False):
        actual = np.sort(frame["run"].unique().astype(int))
        if not np.array_equal(actual, expected_runs):
            raise ValueError(
                f"Incomplete classification trajectories for tau={tau}, "
                f"method={method}: expected {expected_runs.tolist()}, "
                f"found {actual.tolist()}"
            )
        sample_grid: np.ndarray | None = None
        values: list[np.ndarray] = []
        for run in expected_runs:
            run_frame = frame[frame["run"] == run].sort_values("step")
            samples = run_frame["samples"].to_numpy(dtype=int)
            losses = run_frame["train_loss"].to_numpy(dtype=float)
            if samples.size == 0 or np.any(np.diff(samples) <= 0):
                raise ValueError(
                    f"Invalid trajectory grid for tau={tau}, method={method}, run={run}"
                )
            if sample_grid is None:
                sample_grid = samples
            elif not np.array_equal(sample_grid, samples):
                raise ValueError(
                    "Classification trajectory sample grids differ across runs "
                    f"for tau={tau}, method={method}. No interpolation is performed."
                )
            if not np.all(np.isfinite(losses)):
                raise ValueError(
                    f"Non-finite train loss for tau={tau}, method={method}, run={run}"
                )
            values.append(losses)
        assert sample_grid is not None
        matrix = np.vstack(values)
        means = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if simulations > 1 else np.zeros_like(means)
        se = sd / np.sqrt(simulations)
        for sample, mean, sd_value, se_value in zip(
            sample_grid, means, sd, se, strict=True
        ):
            rows.append(
                {
                    "tau": float(tau),
                    "method": method,
                    "samples": int(sample),
                    "train_loss": float(mean),
                    "train_loss_mean": float(mean),
                    "train_loss_sd": float(sd_value),
                    "train_loss_se": float(se_value),
                    "train_loss_lower": float(mean - se_value),
                    "train_loss_upper": float(mean + se_value),
                    "n_runs": int(simulations),
                }
            )
    return pd.DataFrame(rows)


def _selected_run(raw: pd.DataFrame, figure_run: int) -> pd.DataFrame:
    selected = raw[raw["run"] == int(figure_run) + 1].copy()
    return selected.reset_index(drop=True)


def _plot(
    aggregate: pd.DataFrame,
    methods: list[str],
    output_dir: Path,
) -> None:
    taus = sorted(aggregate["tau"].unique())
    if len(taus) == 5:
        # Three equally sized panels on the first row and two centered panels on
        # the second row. A six-column GridSpec gives every panel width two while
        # leaving one column of symmetric whitespace on each side of row two.
        figure = plt.figure(figsize=(15.6, 7.8))
        grid = figure.add_gridspec(2, 6)
        axes_list = [
            figure.add_subplot(grid[0, 0:2]),
            figure.add_subplot(grid[0, 2:4]),
            figure.add_subplot(grid[0, 4:6]),
            figure.add_subplot(grid[1, 1:3]),
            figure.add_subplot(grid[1, 3:5]),
        ]
    else:
        columns = min(3, max(1, len(taus)))
        rows = int(math.ceil(len(taus) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.2 * columns, 3.9 * rows),
            squeeze=False,
        )
        axes_list = list(axes.ravel())

    handles: dict[str, object] = {}
    for axis, tau in zip(axes_list, taus, strict=False):
        frame = aggregate[aggregate["tau"] == tau]
        # Draw uncertainty first so every mean line remains visible.
        for method in methods:
            values = frame[frame["method"] == method].sort_values("samples")
            if values.empty:
                continue
            style = STYLES.get(method, {})
            color = style.get("color")
            axis.fill_between(
                values["samples"].to_numpy(dtype=float),
                values["train_loss_lower"].to_numpy(dtype=float),
                values["train_loss_upper"].to_numpy(dtype=float),
                color=color,
                alpha=0.18,
                linewidth=0.0,
                zorder=1,
            )
        for method in methods:
            values = frame[frame["method"] == method].sort_values("samples")
            if values.empty:
                continue
            style = STYLES.get(method, {})
            line, = axis.plot(
                values["samples"],
                values["train_loss_mean"],
                label=DISPLAY_NAMES.get(method, method),
                linewidth=1.55,
                markersize=3.4,
                markevery=max(1, len(values) // 10),
                zorder=3,
                **style,
            )
            handles.setdefault(method, line)
        axis.set_title(rf"$\tau={tau:g}$")
        axis.set_xlabel("sample number")
        axis.set_ylabel("train loss")
        axis.grid(True, alpha=0.35)

    for axis in axes_list[len(taus):]:
        axis.axis("off")
    ordered = [handles[name] for name in methods if name in handles]
    labels = [DISPLAY_NAMES.get(name, name) for name in methods if name in handles]
    if ordered:
        figure.legend(
            ordered,
            labels,
            loc="lower center",
            ncol=min(6, len(ordered)),
            bbox_to_anchor=(0.5, -0.01),
            frameon=False,
        )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    figure.savefig(output_dir / "figure.png", dpi=240, bbox_inches="tight")
    figure.savefig(output_dir / "figure.pdf", bbox_inches="tight")
    plt.close(figure)


def _print_final_metric_summary(summary: pd.DataFrame) -> None:
    columns = [
        "tau",
        "method",
        "train_loss_mean",
        "train_accuracy_mean",
        "test_loss_mean",
        "test_accuracy_mean",
    ]
    report = summary.loc[:, columns].copy()
    report = report.rename(
        columns={
            "train_loss_mean": "train loss",
            "train_accuracy_mean": "train accuracy",
            "test_loss_mean": "test loss",
            "test_accuracy_mean": "test accuracy",
        }
    )
    print("\nFinal strategic-classification metrics "
          "(means across completed simulations):")
    print(
        report.to_string(
            index=False,
            formatters={
                "tau": lambda value: f"{float(value):.2f}",
                "train loss": lambda value: f"{float(value):.6f}",
                "train accuracy": lambda value: f"{float(value):.6f}",
                "test loss": lambda value: f"{float(value):.6f}",
                "test accuracy": lambda value: f"{float(value):.6f}",
            },
        )
    )
    print(
        "Per-run values are cached in final_scores.csv; aggregate means, SDs, "
        "and SEs are cached in summary.csv."
    )


def write_classification_outputs(
    results: list[dict],
    methods: list[str],
    simulations: int,
    figure_run: int,
    output_dir: Path,
    plot_methods: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_scores = _final_scores(results)
    summary = _summary(final_scores, methods)
    raw = _raw_trajectories(results)
    aggregate = _aggregate_trajectories(raw, simulations)
    selected = _selected_run(raw, figure_run)

    final_scores.to_csv(output_dir / "final_scores.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    raw.to_csv(output_dir / "figure_data_raw.csv", index=False)
    aggregate.to_csv(output_dir / "figure_data.csv", index=False)
    selected.to_csv(output_dir / "figure_data_selected_run.csv", index=False)
    _plot(aggregate, list(plot_methods or methods), output_dir)
    _print_final_metric_summary(summary)
