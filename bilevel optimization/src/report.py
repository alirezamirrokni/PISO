from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



plt.rcParams.update(
    {
        "font.size": 14.0,
        "font.weight": "bold",
        "axes.labelsize": 18.0,
        "axes.labelweight": "bold",
        "axes.titlesize": 18.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 14.0,
        "ytick.labelsize": 14.0,
        "legend.fontsize": 12.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from src.methods.common import MethodTrace


DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    
    "GZO_NS": {"color": "blue", "marker": "o", "linestyle": "-", "linewidth": 2.20, "label": "GZO-NS"},
    "GZO_HS": {"color": "cyan", "marker": "s", "linestyle": "-", "linewidth": 2.20, "label": "GZO-HS"},
    "ZO_TG": {"color": "green", "marker": "^", "linestyle": "-", "linewidth": 2.20, "label": "ZO-TG"},
    "ZO_OG": {"color": "red", "marker": "D", "linestyle": "-", "linewidth": 2.20, "label": "ZO-OG"},
    "ZO_OGVR": {"color": "black", "marker": "*", "linestyle": "-", "linewidth": 2.20, "label": "ZO-OGVR"},
    "GaussianPISO": {"color": "purple", "marker": "X", "linestyle": "-", "linewidth": 2.20, "label": "GaussianPISO"},
    "CyclePISO": {"color": "magenta", "marker": "v", "linestyle": "-", "linewidth": 2.20, "label": "CyclePISO"},
    "GaussianPISO2": {"color": "tab:brown", "marker": "h", "linestyle": "-", "linewidth": 2.20, "label": "GaussianPISO²"},
    "CyclePISO2": {"color": "tab:pink", "marker": "<", "linestyle": "-", "linewidth": 2.20, "label": "CyclePISO²"},
    
    
    "PZOS": {"color": "darkorange", "marker": "P", "linestyle": "-", "linewidth": 2.35, "label": "PZOS"},
    "ZOS": {"color": "tab:olive", "marker": ">", "linestyle": "--", "linewidth": 2.35, "label": "ZOS"},
}



def _style(method: str) -> dict[str, Any]:
    return DEFAULT_STYLES.get(method, {"linewidth": 2.00, "label": method})


def _slug(method: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", method).lower()
    return text.replace("p_i_s_o2", "piso2").replace("p_i_s_o", "piso")


def _legend_below(
    fig: Any,
    axes: Any,
    *,
    fontsize: float = 12.5,
) -> float:

    axes_array = np.atleast_1d(axes).ravel()
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes_array:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return 0.06
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=False,
        prop={"size": fontsize, "weight": "bold"},
        columnspacing=0.48,
        handlelength=1.25,
        handletextpad=0.28,
        borderaxespad=0.0,
    )
    
    
    
    
    return max(0.050, min(0.120, 0.58 / float(fig.get_figheight())))


def _format_axis(
    axis: Any,
    *,
    label_size: float = 18.0,
    tick_size: float = 14.0,
    title_size: float = 18.0,
) -> None:

    axis.xaxis.label.set_size(label_size)
    axis.yaxis.label.set_size(label_size)
    axis.xaxis.label.set_weight("bold")
    axis.yaxis.label.set_weight("bold")
    axis.title.set_size(title_size)
    axis.title.set_weight("bold")
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize(tick_size)
        tick.set_fontweight("bold")



def _normalize_per_instance(raw: pd.DataFrame, reference_methods: list[str]) -> pd.DataFrame:
    normalized = raw.copy()
    reference = raw[raw["method"].isin(reference_methods)]
    if reference.empty:
        raise ValueError(
            "none of plotting.normalization_methods are available in the loaded traces"
        )
    extrema = (
        reference.groupby("instance_id", as_index=False)["objective"]
        .agg(objective_low="min", objective_high="max")
    )
    normalized = normalized.merge(extrema, on="instance_id", how="left")
    span = normalized["objective_high"] - normalized["objective_low"]
    normalized["normalized_objective"] = np.where(
        span > 1.0e-14,
        (normalized["objective"] - normalized["objective_low"]) / span,
        0.00,
    )
    return normalized


def _trajectory_summary(normalized: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_frame = normalized[normalized["method"] == method]
        for iteration, group in method_frame.groupby("iteration"):
            values = group["normalized_objective"].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "iteration": int(iteration),
                    "median": float(np.median(values)),
                    "p10": float(np.quantile(values, 0.10)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p90": float(np.quantile(values, 0.90)),
                    "count": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _plot_trajectory(
    summary: pd.DataFrame,
    methods: list[str],
    destination: Path,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10.40, 6.20))
    for method in methods:
        frame = summary[summary["method"] == method].sort_values("iteration")
        if frame.empty:
            continue
        style = _style(method)
        x = frame["iteration"].to_numpy(dtype=float)
        axis.fill_between(
            x,
            frame["p10"].to_numpy(dtype=float),
            frame["p90"].to_numpy(dtype=float),
            color=style.get("color"),
            alpha=0.10,
            linewidth=0,
        )
        axis.fill_between(
            x,
            frame["p25"].to_numpy(dtype=float),
            frame["p75"].to_numpy(dtype=float),
            color=style.get("color"),
            alpha=0.22,
            linewidth=0,
        )
        axis.plot(
            x,
            frame["median"].to_numpy(dtype=float),
            color=style.get("color"),
            linewidth=style.get("linewidth", 2.20),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker"),
            markersize=5.6,
            markevery=max(1, len(frame) // 10),
            label=style.get("label", method),
        )
    axis.set_xlabel("iteration")
    axis.set_ylabel(ylabel.lower())
    axis.grid(alpha=0.28)
    _format_axis(axis, label_size=18.0, tick_size=14.0, title_size=18.0)
    bottom = _legend_below(fig, axis, fontsize=12.5)
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _routing_pairwise_data(
    normalized: pd.DataFrame,
    snapshots: list[int],
    piso_method: str,
) -> pd.DataFrame:
    references = ("PZOS", "ZOS", "GZO_NS", "GZO_HS")
    required = {*references, piso_method}
    available = set(normalized["method"].unique())
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(
            f"cannot build pairwise plot for {piso_method}; missing methods: {missing}"
        )

    pivot = normalized.pivot_table(
        index=["instance_id", "dimension"],
        columns=["method", "iteration"],
        values="normalized_objective",
        aggfunc="last",
    )
    rows: list[dict[str, Any]] = []
    for (instance_id, dimension), row in pivot.iterrows():
        for snapshot in snapshots:
            item: dict[str, Any] = {
                "instance_id": int(instance_id),
                "dimension": int(dimension),
                "iteration": int(snapshot),
                piso_method: float(row[(piso_method, snapshot)]),
            }
            for reference in references:
                item[reference] = float(row[(reference, snapshot)])
            rows.append(item)
    return pd.DataFrame(rows)



def _plot_routing_pairwise(
    normalized: pd.DataFrame,
    snapshots: list[int],
    piso_method: str,
    destination: Path,
) -> pd.DataFrame:
    data = _routing_pairwise_data(normalized, snapshots, piso_method)
    references = ("PZOS", "ZOS", "GZO_NS", "GZO_HS")
    n_rows = len(references)
    n_columns = len(snapshots)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.60 * n_columns + 1.05, 3.55 * n_rows + 1.55),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(n_rows, n_columns)

    value_arrays = [data[piso_method].to_numpy(dtype=float)]
    value_arrays.extend(data[reference].to_numpy(dtype=float) for reference in references)
    values = np.concatenate(value_arrays)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        lower, upper = 0.00, 1.00
    else:
        span = max(float(finite.max() - finite.min()), 0.10)
        margin = 0.06 * span
        lower = min(0.00, float(finite.min()) - margin)
        upper = max(1.00, float(finite.max()) + margin)

    image = None
    method_label = _style(piso_method).get("label", piso_method)
    for row_index, reference in enumerate(references):
        for column_index, snapshot in enumerate(snapshots):
            axis = axes[row_index, column_index]
            frame = data[data["iteration"] == snapshot]
            image = axis.scatter(
                frame[reference],
                frame[piso_method],
                c=frame["dimension"],
                cmap="turbo",
                s=32,
                alpha=0.88,
                edgecolors="none",
            )
            axis.plot(
                [lower, upper],
                [lower, upper],
                "k--",
                linewidth=1.85,
                label="equal performance",
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.grid(alpha=0.25)
            if row_index == 0:
                axis.set_title(f"t = {snapshot}")
            reference_label = str(_style(reference).get("label", reference))
            axis.set_xlabel(f"normalized {reference_label} objective")
            _format_axis(
                axis,
                label_size=15.0,
                tick_size=12.5,
                title_size=17.0,
            )

    fig.supylabel(
        f"normalized {method_label} objective",
        x=0.022,
        fontsize=17.0,
        fontweight="bold",
    )
    bottom = max(_legend_below(fig, axes, fontsize=12.5), 0.075)
    fig.subplots_adjust(
        left=0.105,
        right=0.865,
        bottom=bottom,
        top=0.965,
        hspace=0.48,
        wspace=0.24,
    )
    if image is not None:
        
        colorbar_axis = fig.add_axes([0.900, bottom + 0.060, 0.020, 0.80 - bottom])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            "problem dimension $d_x$",
            fontsize=15.0,
            fontweight="bold",
            labelpad=11.0,
        )
        for tick in colorbar.ax.get_yticklabels():
            tick.set_fontsize(12.5)
            tick.set_fontweight("bold")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return data



def _plot_security_dimension(
    normalized: pd.DataFrame,
    snapshots: list[int],
    dimension_min: int,
    methods: list[str],
    destination: Path,
) -> pd.DataFrame:
    filtered = normalized[normalized["dimension"] >= dimension_min]
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        at_snapshot = filtered[filtered["iteration"] == snapshot]
        for (method, dimension), group in at_snapshot.groupby(["method", "dimension"]):
            values = group["normalized_objective"].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "dimension": int(dimension),
                    "iteration": int(snapshot),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "count": int(values.size),
                }
            )
    data = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        len(snapshots),
        figsize=(max(14.20, 4.55 * len(snapshots)), 5.85),
        sharex=True,
        sharey=True,
    )
    if len(snapshots) == 1:
        axes = [axes]
    for axis, snapshot in zip(axes, snapshots, strict=True):
        for method in methods:
            frame = data[
                (data["method"] == method) & (data["iteration"] == snapshot)
            ].sort_values("dimension")
            if frame.empty:
                continue
            style = _style(method)
            x = frame["dimension"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                frame["minimum"].to_numpy(dtype=float),
                frame["maximum"].to_numpy(dtype=float),
                color=style.get("color"),
                alpha=0.08,
                linewidth=0,
            )
            axis.plot(
                x,
                frame["median"].to_numpy(dtype=float),
                color=style.get("color"),
                linewidth=style.get("linewidth", 2.20),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker"),
                markersize=5.4,
                markevery=max(1, len(frame) // 10),
                label=style.get("label", method),
            )
        axis.set_title(f"t = {snapshot}")
        axis.set_xlabel("problem dimension $d_x$")
        axis.set_xlim(dimension_min, int(filtered["dimension"].max()))
        axis.grid(alpha=0.25)
        _format_axis(axis, label_size=16.5, tick_size=13.0, title_size=17.0)
    axes[0].set_ylabel("normalized objective")
    _format_axis(axes[0], label_size=16.5, tick_size=13.0, title_size=17.0)
    bottom = _legend_below(fig, axes, fontsize=12.5)
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    fig.savefig(destination.with_suffix(".png"), dpi=240)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    return data


def write_outputs(
    family: str,
    traces: list[MethodTrace],
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for iteration, (objective, calls) in enumerate(
            zip(trace.objectives, trace.oracle_calls, strict=True)
        ):
            rows.append(
                {
                    "family": family,
                    "instance_id": trace.instance_id,
                    "dimension": trace.dimension,
                    "method": trace.method,
                    "iteration": iteration,
                    "objective": float(objective),
                    "algorithm_oracle_calls": int(calls),
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "raw_runs.csv", index=False)

    plotting = config.get("plotting", {})
    available = set(raw["method"])
    methods = [
        method
        for method in plotting.get("methods", list(config["methods"]))
        if method in available
    ]
    normalization_methods = [
        method
        for method in plotting.get("normalization_methods", ["ZOS", "PZOS"])
        if method in available
    ]
    if not normalization_methods:
        normalization_methods = list(methods)
    normalized = _normalize_per_instance(raw, normalization_methods)
    normalized.to_csv(output_dir / "normalized_runs.csv", index=False)

    summary = _trajectory_summary(normalized, methods)
    summary.to_csv(output_dir / "trajectory_summary.csv", index=False)

    final = (
        normalized.sort_values("iteration")
        .groupby(["instance_id", "dimension", "method"], as_index=False)
        .tail(1)
    )
    final.to_csv(output_dir / "final_scores.csv", index=False)

    if family == "routing":
        _plot_trajectory(
            summary,
            methods,
            output_dir / "routing_normalized_objective_trajectories",
            "normalized objective",
        )
        snapshots = [
            int(value)
            for value in plotting.get("scatter_iterations", [10, 25, 50])
        ]
        pairwise_references = {"PZOS", "ZOS", "GZO_NS", "GZO_HS"}
        pairwise_methods = []
        if pairwise_references.issubset(available):
            pairwise_methods = [
                method
                for method in plotting.get(
                    "pairwise_methods",
                    [
                        "GaussianPISO",
                        "CyclePISO",
                        "GaussianPISO2",
                        "CyclePISO2",
                    ],
                )
                if method in available
            ]
        for method in pairwise_methods:
            stem = output_dir / f"routing_{_slug(method)}_pairwise_comparison"
            data = _plot_routing_pairwise(
                normalized,
                snapshots,
                method,
                stem,
            )
            data.to_csv(stem.with_suffix(".csv"), index=False)
    elif family == "security":
        _plot_trajectory(
            summary,
            methods,
            output_dir / "security_normalized_objective_trajectories",
            "normalized objective",
        )
        dimension_data = _plot_security_dimension(
            normalized,
            [
                int(value)
                for value in plotting.get(
                    "dimension_iterations", [50, 150, 250]
                )
            ],
            int(plotting.get("dimension_plot_min", 100)),
            methods,
            output_dir / "security_objective_by_dimension",
        )
        dimension_data.to_csv(
            output_dir / "security_objective_by_dimension.csv",
            index=False,
        )
    else:
        raise ValueError(f"unknown family: {family}")
