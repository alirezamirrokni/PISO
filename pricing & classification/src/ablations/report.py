from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .suites import VariantSpec


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
        "legend.fontsize": 13.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COLORS = [
    "purple",
    "magenta",
    "blue",
    "green",
    "red",
    "black",
    "darkorange",
    "tab:brown",
]
MARKERS = ["X", "v", "o", "^", "D", "*", "s", "h"]
LINESTYLES = ["-", "-", "--", "-.", ":", "--", "-.", ":"]

COMPONENT_STYLES = {
    0: {"color": "black", "marker": "o", "linestyle": "--"},
    1: {"color": "blue", "marker": "s", "linestyle": "-."},
    2: {"color": "purple", "marker": "X", "linestyle": "-"},
    3: {"color": "tab:brown", "marker": "h", "linestyle": "-"},
}


def _stepwise_align(
    samples: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    values = np.asarray(values, dtype=float)
    if samples.ndim != 1 or values.ndim != 1 or samples.size != values.size:
        raise ValueError("samples and values must be one-dimensional and equal length")
    if samples.size == 0 or samples[0] != 0 or np.any(np.diff(samples) <= 0):
        raise ValueError("trace samples must start at zero and be strictly increasing")
    indices = np.searchsorted(samples, grid, side="right") - 1
    indices = np.clip(indices, 0, values.size - 1)
    return values[indices]


def _style_axis(axis: Any) -> None:
    axis.xaxis.label.set_size(17.0)
    axis.yaxis.label.set_size(17.0)
    axis.xaxis.label.set_weight("bold")
    axis.yaxis.label.set_weight("bold")
    axis.title.set_size(17.0)
    axis.title.set_weight("bold")
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize(13.0)
        tick.set_fontweight("bold")


def _axis_limits(
    frame: pd.DataFrame,
    variant_ids: list[str],
    *,
    mode: str,
    final_window_fraction: float,
    margin_fraction: float,
) -> tuple[float, float] | None:
    selected = frame[frame["variant_id"].isin(variant_ids)]
    if selected.empty:
        return None

    if mode == "final_window":
        cutoff = float(selected["samples"].max()) * (1.0 - final_window_fraction)
        selected = selected[selected["samples"] >= cutoff]

    values = np.concatenate(
        [
            selected["objective_lower"].to_numpy(dtype=float),
            selected["objective_upper"].to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    margin = margin_fraction * span if span > 0.0 else max(abs(lower) * 0.05, 0.5)
    return lower - margin, upper + margin


def _variant_style(variant: VariantSpec) -> dict[str, Any]:
    index = int(variant.metadata.get("variant_index", 0))
    if variant.metadata.get("kind") == "grouped_explicit":
        return dict(COMPONENT_STYLES[index % len(COMPONENT_STYLES)])
    return {
        "color": COLORS[index % len(COLORS)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
    }


def _plot_suite(
    *,
    figure_data: pd.DataFrame,
    variants: list[VariantSpec],
    suite_config: dict[str, Any],
    output_dir: Path,
) -> None:
    panels = list(suite_config["panels"])
    n_panels = len(panels)
    if n_panels not in {2, 4}:
        raise ValueError("the focused ablation plots support exactly 2 or 4 panels")

    plotting = suite_config.get("plotting", {})
    
    y_limit_mode = str(plotting.get("y_limit_mode", "final_window"))
    final_window_fraction = float(plotting.get("final_window_fraction", 0.30))
    y_margin_fraction = float(plotting.get("y_margin_fraction", 0.10))
    legend_fontsize = float(plotting.get("legend_fontsize", 13.0))
    line_width = float(plotting.get("line_width", 2.20))
    marker_size = float(plotting.get("marker_size", 5.50))

    width = 12.8 if n_panels == 2 else 20.0
    figure, axes = plt.subplots(1, n_panels, figsize=(width, 5.8), squeeze=False)
    axes = axes.ravel()

    variant_by_panel: dict[str, list[VariantSpec]] = {}
    for variant in variants:
        panel_id = str(variant.metadata["panel_id"])
        variant_by_panel.setdefault(panel_id, []).append(variant)
    for panel_variants in variant_by_panel.values():
        panel_variants.sort(key=lambda item: int(item.metadata["variant_index"]))

    legend_handles: dict[str, Any] = {}
    legend_order: list[str] = []

    for axis, panel in zip(axes, panels, strict=True):
        panel_id = str(panel["id"])
        panel_variants = variant_by_panel.get(panel_id, [])
        if not panel_variants:
            raise ValueError(f"panel {panel_id!r} has no generated variants")

        for variant in panel_variants:
            values = figure_data[
                figure_data["variant_id"] == variant.variant_id
            ].sort_values("samples")
            if values.empty:
                continue
            style = _variant_style(variant)
            markevery = max(1, len(values) // 10)
            (line,) = axis.plot(
                values["samples"].to_numpy(dtype=float),
                values["objective_mean"].to_numpy(dtype=float),
                label=variant.label,
                linewidth=line_width,
                markersize=marker_size,
                markevery=markevery,
                **style,
            )
            axis.fill_between(
                values["samples"].to_numpy(dtype=float),
                values["objective_lower"].to_numpy(dtype=float),
                values["objective_upper"].to_numpy(dtype=float),
                color=line.get_color(),
                alpha=0.18,
                linewidth=0.0,
                interpolate=True,
            )
            if variant.label not in legend_handles:
                legend_handles[variant.label] = line
                legend_order.append(variant.label)

        limits = _axis_limits(
            figure_data,
            [variant.variant_id for variant in panel_variants],
            mode=y_limit_mode,
            final_window_fraction=final_window_fraction,
            margin_fraction=y_margin_fraction,
        )
        if limits is not None:
            axis.set_ylim(*limits)

        axis.set_title(str(panel["title"]))
        axis.set_xlabel("sample number")
        axis.set_ylabel("objective")
        axis.grid(True, alpha=0.45)
        _style_axis(axis)

    if legend_order:
        figure.legend(
            [legend_handles[label] for label in legend_order],
            legend_order,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=len(legend_order),
            frameon=False,
            prop={"size": legend_fontsize, "weight": "bold"},
            handlelength=1.30,
            columnspacing=0.65,
            handletextpad=0.32,
            borderaxespad=0.0,
        )

    figure.tight_layout(rect=(0.0, 0.135, 1.0, 1.0), w_pad=1.8)
    filename = str(suite_config.get("filename", "ablation"))
    figure.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(figure)


def build_suite_report(
    *,
    suite_name: str,
    suite_config: dict[str, Any],
    records: list[dict[str, Any]],
    variants: list[VariantSpec],
    reference_id: str,
    output_dir: Path,
    max_samples: int,
    grid_points: int,
    bootstrap_resamples: int = 0,
    permutation_resamples: int = 0,
    base_seed: int = 0,
) -> None:
    del bootstrap_resamples, permutation_resamples, base_seed
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, float(max_samples), int(grid_points))
    variant_by_id = {variant.variant_id: variant for variant in variants}

    raw_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for record in records:
        variant_id = str(record["variant"]["variant_id"])
        variant = variant_by_id[variant_id]
        trace = record["trace"]
        samples = np.asarray(trace["samples"], dtype=float)
        objectives = np.asarray(trace["objectives"], dtype=float)
        failed = bool(trace["failed"]) or not np.all(np.isfinite(objectives))

        for sample, objective in zip(samples, objectives, strict=True):
            raw_rows.append(
                {
                    "suite": suite_name,
                    "panel_id": variant.metadata["panel_id"],
                    "panel_title": variant.metadata["panel_title"],
                    "variant_id": variant_id,
                    "label": variant.label,
                    "week_id": record["week_id"],
                    "run_index": int(record["run_index"]),
                    "samples": int(sample),
                    "objective": float(objective),
                }
            )

        aligned = (
            np.full_like(grid, np.nan, dtype=float)
            if failed
            else _stepwise_align(samples, objectives, grid)
        )
        for sample, objective in zip(grid, aligned, strict=True):
            aligned_rows.append(
                {
                    "suite": suite_name,
                    "panel_id": variant.metadata["panel_id"],
                    "panel_title": variant.metadata["panel_title"],
                    "variant_id": variant_id,
                    "label": variant.label,
                    "week_id": record["week_id"],
                    "run_index": int(record["run_index"]),
                    "samples": float(sample),
                    "objective": float(objective),
                }
            )

        unit_rows.append(
            {
                "suite": suite_name,
                "panel_id": variant.metadata["panel_id"],
                "panel_title": variant.metadata["panel_title"],
                "variant_id": variant_id,
                "label": variant.label,
                "week_id": record["week_id"],
                "run_index": int(record["run_index"]),
                "failed": failed,
                "failure_reason": str(trace["failure_reason"]),
                "final_objective": float(aligned[-1]) if not failed else np.nan,
                "runtime_seconds": float(trace["runtime_seconds"]),
                "iterations": int(trace["iterations"]),
            }
        )

    raw = pd.DataFrame(raw_rows)
    aligned = pd.DataFrame(aligned_rows)
    unit_metrics = pd.DataFrame(unit_rows)
    raw.to_csv(output_dir / "raw_trajectories.csv", index=False)
    aligned.to_csv(output_dir / "aligned_trajectories.csv", index=False)
    unit_metrics.to_csv(output_dir / "unit_metrics.csv", index=False)

    figure_rows: list[dict[str, Any]] = []
    for (variant_id, label, panel_id, panel_title, sample), frame in aligned.groupby(
        ["variant_id", "label", "panel_id", "panel_title", "samples"],
        sort=False,
    ):
        values = frame["objective"].dropna().to_numpy(dtype=float)
        count = int(values.size)
        mean = float(values.mean()) if count else np.nan
        sd = float(values.std(ddof=1)) if count > 1 else 0.0 if count == 1 else np.nan
        se = sd / np.sqrt(count) if count else np.nan
        figure_rows.append(
            {
                "variant_id": variant_id,
                "label": label,
                "panel_id": panel_id,
                "panel_title": panel_title,
                "samples": float(sample),
                "objective_mean": mean,
                "objective_sd": sd,
                "objective_se": se,
                
                "objective_lower": mean - se,
                "objective_upper": mean + se,
                "n_units": count,
            }
        )
    figure_data = pd.DataFrame(figure_rows)
    figure_data.to_csv(output_dir / "figure_data.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        frame = unit_metrics[unit_metrics["variant_id"] == variant.variant_id]
        values = frame["final_objective"].dropna().to_numpy(dtype=float)
        count = int(values.size)
        mean = float(values.mean()) if count else np.nan
        sd = float(values.std(ddof=1)) if count > 1 else 0.0 if count == 1 else np.nan
        se = sd / np.sqrt(count) if count else np.nan
        reference_frame = unit_metrics[
            unit_metrics["variant_id"] == reference_id
        ][["week_id", "run_index", "final_objective"]].rename(
            columns={"final_objective": "reference_objective"}
        )
        paired = frame.merge(reference_frame, on=["week_id", "run_index"], how="inner")
        differences = (
            paired["final_objective"] - paired["reference_objective"]
        ).dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "suite": suite_name,
                "panel_id": variant.metadata["panel_id"],
                "panel_title": variant.metadata["panel_title"],
                "variant_id": variant.variant_id,
                "label": variant.label,
                "final_mean": mean,
                "final_sd": sd,
                "final_se": se,
                "final_lower_95ci": mean - 1.96 * se if count else np.nan,
                "final_upper_95ci": mean + 1.96 * se if count else np.nan,
                "paired_difference_vs_reference": (
                    float(differences.mean()) if differences.size else np.nan
                ),
                "failure_rate": float(frame["failed"].mean()) if not frame.empty else np.nan,
                "n_successful_units": count,
                "n_total_units": int(len(frame)),
                **{f"param_{key}": value for key, value in variant.params.items()},
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    manifest = {
        "suite": suite_name,
        "reference_id": reference_id,
        "figure": str(suite_config.get("filename", "ablation")),
        "aggregation": (
            "mean and standard error over all paired week-by-run units at each "
            "sample-budget checkpoint"
        ),
        "panels": suite_config["panels"],
        "variants": [
            {
                "variant_id": variant.variant_id,
                "label": variant.label,
                "params": variant.params,
                "metadata": variant.metadata,
            }
            for variant in variants
        ],
    }
    (output_dir / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    _plot_suite(
        figure_data=figure_data,
        variants=variants,
        suite_config=suite_config,
        output_dir=output_dir,
    )
