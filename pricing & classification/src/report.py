from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 14.0,
        "font.weight": "bold",
        "axes.labelsize": 17.0,
        "axes.labelweight": "bold",
        "axes.titlesize": 17.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13.0,
        "ytick.labelsize": 13.0,
        "legend.fontsize": 12.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


STYLES = {
    
    "GZO_NS": {"color": "blue", "marker": "o", "linestyle": "-"},
    "GZO_HS": {"color": "cyan", "marker": "s", "linestyle": "-"},
    "ZO_TG": {"color": "green", "marker": "^", "linestyle": "-"},
    "ZO_OG": {"color": "red", "marker": "D", "linestyle": "-"},
    "ZO_OGVR": {"color": "black", "marker": "*", "linestyle": "-"},
    "GaussianPISO": {"color": "purple", "marker": "X", "linestyle": "-"},
    "CyclePISO": {"color": "magenta", "marker": "v", "linestyle": "-"},
    "GaussianPISO2": {"color": "tab:brown", "marker": "h", "linestyle": "-"},
    "CyclePISO2": {"color": "tab:pink", "marker": "<", "linestyle": "-"},
}


DISPLAY_NAMES: dict[str, str] = {
    "GaussianPISO2": r"GaussianPISO$^2$",
    "CyclePISO2": r"CyclePISO$^2$",
}






def write_outputs(
    results: list[dict],
    methods: list[str],
    figure_run: int,
    output_dir: Path,
    *,
    plot_methods: list[str] | None = None,
    y_margin_fraction: float = 0.05,
    y_limit_mode: str = "post_drop",
    post_drop_remaining_fraction: float = 0.20,
    post_drop_tail_fraction: float = 0.25,
    final_window_fraction: float = 0.30,
) -> None:







    selected_run_number = int(figure_run) + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_alphas.csv").unlink(missing_ok=True)

    final_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    for item in results:
        trace = item["trace"]
        run_number = int(item["run"]) + 1

        final_rows.append(
            {
                "week_id": item["week_id"],
                "week": item["week_label"],
                "run": run_number,
                "method": item["method"],
                "objective": float(trace.objectives[-1]),
            }
        )

        for samples, objective in zip(
            trace.samples,
            trace.objectives,
            strict=True,
        ):
            trace_rows.append(
                {
                    "week_id": item["week_id"],
                    "week": item["week_label"],
                    "run": run_number,
                    "method": item["method"],
                    "samples": int(samples),
                    "objective": float(objective),
                }
            )

    final = pd.DataFrame(final_rows)
    raw_trace_data = pd.DataFrame(trace_rows)
    figure_data = _aggregate_trace_statistics(raw_trace_data)
    summary = _build_final_summary(final, methods)

    final.to_csv(output_dir / "final_scores.csv", index=False)
    raw_trace_data.to_csv(output_dir / "figure_data_raw.csv", index=False)
    selected_run_data = raw_trace_data[
        raw_trace_data["run"] == selected_run_number
    ].copy()
    selected_run_data.to_csv(
        output_dir / "figure_data_selected_run.csv",
        index=False,
    )
    figure_data.to_csv(output_dir / "figure_data.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    _plot(
        figure_data,
        methods if plot_methods is None else plot_methods,
        output_dir / "figure.png",
        output_dir / "figure.pdf",
        y_margin_fraction=y_margin_fraction,
        y_limit_mode=y_limit_mode,
        post_drop_remaining_fraction=post_drop_remaining_fraction,
        post_drop_tail_fraction=post_drop_tail_fraction,
        final_window_fraction=final_window_fraction,
    )






def _build_final_summary(final: pd.DataFrame, methods: list[str]) -> pd.DataFrame:

    summary_rows: list[dict[str, object]] = []

    for (week_id, week), frame in final.groupby(
        ["week_id", "week"],
        sort=False,
    ):
        row: dict[str, object] = {"week_id": week_id, "week": week}

        for method in methods:
            values = frame.loc[
                frame["method"] == method,
                "objective",
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            count = int(values.size)

            if count == 0:
                mean = np.nan
                sd = np.nan
                se = np.nan
            elif count == 1:
                mean = float(values[0])
                sd = 0.0
                se = 0.0
            else:
                mean = float(values.mean())
                sd = float(values.std(ddof=1))
                se = float(sd / np.sqrt(count))

            row[f"{method}_obj"] = mean
            row[f"{method}_sd"] = sd
            row[f"{method}_se"] = se
            row[f"{method}_n"] = count

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _aggregate_trace_statistics(raw: pd.DataFrame) -> pd.DataFrame:







    output_columns = [
        "week_id",
        "week",
        "method",
        "samples",
        "objective",
        "objective_mean",
        "objective_sd",
        "objective_se",
        "objective_lower",
        "objective_upper",
        "n_runs",
    ]
    if raw.empty:
        return pd.DataFrame(columns=output_columns)

    required_columns = {
        "week_id",
        "week",
        "run",
        "method",
        "samples",
        "objective",
    }
    missing = required_columns.difference(raw.columns)
    if missing:
        raise ValueError(
            "Raw trajectory table is missing required columns: "
            f"{sorted(missing)}"
        )

    run_values = np.sort(raw["run"].unique().astype(int))
    expected_run_values = np.arange(1, int(run_values.max()) + 1, dtype=int)
    if not np.array_equal(run_values, expected_run_values):
        raise ValueError(
            "Run identifiers must be consecutive and start at 1; found "
            f"{run_values.tolist()}"
        )

    rows: list[dict[str, object]] = []
    grouped = raw.groupby(
        ["week_id", "week", "method"],
        sort=False,
    )

    for (week_id, week, method), method_frame in grouped:
        actual_runs = np.sort(method_frame["run"].unique().astype(int))
        if not np.array_equal(actual_runs, expected_run_values):
            raise ValueError(
                f"Incomplete runs for week={week_id}, method={method}: "
                f"expected {expected_run_values.tolist()}, "
                f"found {actual_runs.tolist()}"
            )

        reference_samples: np.ndarray | None = None
        objective_rows: list[np.ndarray] = []

        for run_number in expected_run_values:
            run_frame = (
                method_frame[method_frame["run"] == run_number]
                .sort_values("samples")
                .reset_index(drop=True)
            )
            if run_frame.empty:
                raise ValueError(
                    f"Missing trajectory for week={week_id}, method={method}, "
                    f"run={run_number}"
                )
            if run_frame["samples"].duplicated().any():
                duplicates = (
                    run_frame.loc[
                        run_frame["samples"].duplicated(keep=False),
                        "samples",
                    ]
                    .astype(int)
                    .tolist()
                )
                raise ValueError(
                    f"Duplicate sample counts for week={week_id}, "
                    f"method={method}, run={run_number}: {duplicates}"
                )

            samples = run_frame["samples"].to_numpy(dtype=int)
            objectives = run_frame["objective"].to_numpy(dtype=float)

            if samples.size == 0:
                raise ValueError(
                    f"Empty trajectory for week={week_id}, method={method}, "
                    f"run={run_number}"
                )
            if np.any(np.diff(samples) <= 0):
                raise ValueError(
                    f"Sample counts are not strictly increasing for "
                    f"week={week_id}, method={method}, run={run_number}"
                )
            if not np.all(np.isfinite(objectives)):
                raise ValueError(
                    f"Non-finite objective encountered for week={week_id}, "
                    f"method={method}, run={run_number}"
                )

            if reference_samples is None:
                reference_samples = samples
            elif not np.array_equal(samples, reference_samples):
                raise ValueError(
                    "Trajectory sample grids differ across runs for "
                    f"week={week_id}, method={method}. No interpolation is "
                    "performed; inspect the corresponding cache files."
                )

            objective_rows.append(objectives)

        assert reference_samples is not None
        run_matrix = np.vstack(objective_rows)
        count = int(run_matrix.shape[0])
        means = run_matrix.mean(axis=0)

        if count == 1:
            standard_deviations = np.zeros_like(means)
        else:
            standard_deviations = run_matrix.std(axis=0, ddof=1)
        standard_errors = standard_deviations / np.sqrt(count)
        lowers = means - standard_errors
        uppers = means + standard_errors

        for sample, mean, sd, se, lower, upper in zip(
            reference_samples,
            means,
            standard_deviations,
            standard_errors,
            lowers,
            uppers,
            strict=True,
        ):
            rows.append(
                {
                    "week_id": week_id,
                    "week": week,
                    "method": method,
                    "samples": int(sample),
                    
                    "objective": float(mean),
                    "objective_mean": float(mean),
                    "objective_sd": float(sd),
                    "objective_se": float(se),
                    "objective_lower": float(lower),
                    "objective_upper": float(upper),
                    "n_runs": count,
                }
            )

    return pd.DataFrame(rows, columns=output_columns)






def _finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    return finite[np.isfinite(finite)]


def _uncertainty_limit_frame(frame: pd.DataFrame) -> pd.DataFrame:

    lower = frame[["method", "samples", "objective_lower"]].rename(
        columns={"objective_lower": "objective"}
    )
    upper = frame[["method", "samples", "objective_upper"]].rename(
        columns={"objective_upper": "objective"}
    )
    return pd.concat([lower, upper], ignore_index=True)


def _post_drop_values(
    frame: pd.DataFrame,
    methods: list[str],
    *,
    remaining_fraction: float,
    tail_fraction: float,
) -> np.ndarray:

    retained: list[np.ndarray] = []

    for method in methods:
        method_frame = frame[frame["method"] == method].sort_values("samples")
        values = _finite_values(method_frame["objective"])
        n_values = values.size
        if n_values == 0:
            continue
        if n_values < 5:
            retained.append(values)
            continue

        head_count = max(1, min(3, n_values // 5))
        tail_count = max(3, int(np.ceil(tail_fraction * n_values)))
        initial_level = float(np.median(values[:head_count]))
        tail_level = float(np.median(values[-tail_count:]))
        net_drop = initial_level - tail_level
        full_span = float(values.max() - values.min())

        if net_drop <= max(1e-12, 0.10 * full_span):
            retained.append(values)
            continue

        threshold = tail_level + remaining_fraction * net_drop
        smoothed = (
            pd.Series(values)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )

        cutoff = 0
        for index in range(n_values):
            confirmation = smoothed[index : min(index + 3, n_values)]
            if confirmation.size and float(np.median(confirmation)) <= threshold:
                cutoff = index
                break

        retained.append(values[cutoff:])

    if not retained:
        return np.asarray([], dtype=float)
    return np.concatenate(retained)


def _final_window_values(
    frame: pd.DataFrame,
    methods: list[str],
    *,
    fraction: float,
) -> np.ndarray:

    retained: list[np.ndarray] = []
    fraction = float(fraction)

    for method in methods:
        method_frame = frame[frame["method"] == method].sort_values("samples")
        if method_frame.empty:
            continue

        samples = np.asarray(method_frame["samples"], dtype=float)
        values = np.asarray(method_frame["objective"], dtype=float)
        finite = np.isfinite(samples) & np.isfinite(values)
        samples = samples[finite]
        values = values[finite]
        if values.size == 0:
            continue

        max_sample = float(samples.max())
        min_sample = float(samples.min())
        cutoff = max_sample - fraction * (max_sample - min_sample)
        selected = values[samples >= cutoff]

        if selected.size < min(3, values.size):
            selected = values[-min(3, values.size) :]
        retained.append(selected)

    if not retained:
        return np.asarray([], dtype=float)
    return np.concatenate(retained)


def _axis_limits(
    values: np.ndarray | pd.Series,
    margin_fraction: float,
) -> tuple[float, float] | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None

    y_min = float(finite.min())
    y_max = float(finite.max())
    span = y_max - y_min

    if span > 0.0:
        margin = margin_fraction * span
    else:
        margin = max(abs(y_min) * margin_fraction, 0.5)

    return y_min - margin, y_max + margin







def _format_axis(axis: object) -> None:
    axis.xaxis.label.set_size(17.0)
    axis.yaxis.label.set_size(17.0)
    axis.xaxis.label.set_weight("bold")
    axis.yaxis.label.set_weight("bold")
    axis.title.set_size(17.0)
    axis.title.set_weight("bold")
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize(13.0)
        tick.set_fontweight("bold")

def _plot(
    data: pd.DataFrame,
    methods: list[str],
    png_path: Path,
    pdf_path: Path,
    *,
    y_margin_fraction: float,
    y_limit_mode: str,
    post_drop_remaining_fraction: float,
    post_drop_tail_fraction: float,
    final_window_fraction: float,
) -> None:
    weeks = list(
        data[["week_id", "week"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if not weeks:
        raise ValueError("No trajectory data are available for plotting")

    ncols = 3
    nrows = (len(weeks) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.6, 9.2),
        squeeze=False,
    )

    legend_handles: dict[str, object] = {}

    for axis, (week_id, week) in zip(axes.flat, weeks, strict=False):
        frame = data[data["week_id"] == week_id]
        plotted_frame = frame[frame["method"].isin(methods)]

        for method in methods:
            values = (
                frame[frame["method"] == method]
                .sort_values("samples")
                .reset_index(drop=True)
            )
            if values.empty:
                continue

            style = STYLES.get(method, {})
            samples = values["samples"].to_numpy(dtype=float)
            means = values["objective_mean"].to_numpy(dtype=float)
            lowers = values["objective_lower"].to_numpy(dtype=float)
            uppers = values["objective_upper"].to_numpy(dtype=float)

            (line,) = axis.plot(
                samples,
                means,
                label=DISPLAY_NAMES.get(method, method),
                linewidth=2.20,
                markersize=5.5,
                markevery=max(1, len(values) // 10),
                **style,
            )

            axis.fill_between(
                samples,
                lowers,
                uppers,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0.0,
                interpolate=True,
            )

            legend_handles.setdefault(method, line)

        
        limit_frame = _uncertainty_limit_frame(plotted_frame)

        if y_limit_mode == "post_drop":
            limit_values = _post_drop_values(
                limit_frame,
                methods,
                remaining_fraction=post_drop_remaining_fraction,
                tail_fraction=post_drop_tail_fraction,
            )
        elif y_limit_mode == "final_window":
            limit_values = _final_window_values(
                limit_frame,
                methods,
                fraction=final_window_fraction,
            )
        else:
            limit_values = limit_frame["objective"]

        limits = _axis_limits(limit_values, y_margin_fraction)
        if limits is not None:
            axis.set_ylim(*limits)

        axis.set_title(str(week))
        axis.set_xlabel("sample number")
        axis.set_ylabel("objective")
        axis.grid(True, alpha=0.45)
        _format_axis(axis)

    for axis in axes.flat[len(weeks) :]:
        axis.set_visible(False)

    ordered_methods = [method for method in methods if method in legend_handles]
    if ordered_methods:
        fig.legend(
            [legend_handles[method] for method in ordered_methods],
            [DISPLAY_NAMES.get(method, method) for method in ordered_methods],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=len(ordered_methods),
            prop={"size": 12.0, "weight": "bold"},
            frameon=False,
            handlelength=1.25,
            columnspacing=0.48,
            handletextpad=0.28,
        )

    fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
