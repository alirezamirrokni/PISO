from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


plt.rcParams.update(
    {
        "font.size": 14.0,
        "font.weight": "bold",
        "axes.labelsize": 17.5,
        "axes.labelweight": "bold",
        "axes.titlesize": 17.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DISPLAY_NAMES = {
    "VanillaPG": "Vanilla PG",
    "ZO_TG": "ZO-TG",
    "ZO_OG": "ZO-OG",
    "ZO_OGVR": "ZO-OGVR",
    "GZO_NS": "GZO-NS",
    "GZO_HS": "GZO-HS",
    "GaussianPISO": "GaussianPISO",
    "CyclePISO": "CyclePISO",
    "GaussianPISO2": "GaussianPISO²",
    "CyclePISO2": "CyclePISO²",
    "PePG": "PePG",
    "OraclePePG": "Exact-oracle PePG",
}


# Shared methods use exactly the same colors, markers, and line styles as the
# pricing/classification and bilevel figures. PePG and Vanilla PG use the two
# fixed experiment-exclusive styles used across the paper.
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
    "PePG": {"color": "darkorange", "marker": "P", "linestyle": "-"},
    "VanillaPG": {"color": "tab:olive", "marker": ">", "linestyle": "--"},
    "OraclePePG": {"color": "dimgray", "marker": "D", "linestyle": "--"},
}

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _statistics(values: list[float]) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    mean = float(array.mean())
    if n <= 1:
        return mean, 0.0, 0.0, n
    sd = float(array.std(ddof=1))
    return mean, sd, sd / math.sqrt(n), n


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean that never uses future evaluation points.

    The previous centered smoother leaked post-update values into the point at
    budget zero. That could make fast-changing methods look as though they
    started from a different initial policy.
    """
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size <= 1:
        return values.copy()
    width = max(int(window), 1)
    result = np.empty_like(values)
    for index in range(values.size):
        left = max(0, index - width + 1)
        result[index] = float(np.mean(values[left : index + 1]))
    return result


def _bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    resamples: int = 4000,
) -> tuple[float, float]:
    """Deterministic percentile-bootstrap CI for a sample mean.

    Occupancy shifts are nonnegative and strongly skewed.  A Gaussian
    mean-plus/minus-SE interval can cross zero; clipping that interval to zero
    and then drawing it on a logarithmic axis creates an artificial band down
    to 1e-8.  The bootstrap interval respects the empirical support and avoids
    that plotting artifact without shrinking genuine between-seed variation.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1 or np.allclose(array, array[0], atol=0.0, rtol=0.0):
        return mean, mean
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    indices = rng.integers(0, array.size, size=(int(resamples), array.size))
    boot_means = array[indices].mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.025, 0.975])
    return float(max(lower, 0.0)), float(max(upper, 0.0))


def build_reports(traces: list, config: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    final_rows: list[dict] = []
    for trace in traces:
        initial_value = float(trace.values[0])
        for samples, value, change in zip(
            trace.samples, trace.values, trace.occupancy_changes
        ):
            raw_rows.append(
                {
                    "method": trace.method,
                    "seed": trace.seed,
                    "samples": int(samples),
                    "performative_value": float(value),
                    "value_improvement": float(value) - initial_value,
                    "occupancy_change": float(change),
                }
            )
        final_rows.append(
            {
                "method": trace.method,
                "seed": trace.seed,
                "initial_performative_value": initial_value,
                "final_samples": int(trace.samples[-1]),
                "final_performative_value": float(trace.values[-1]),
                "final_improvement": float(trace.values[-1]) - initial_value,
                "final_occupancy_change": float(trace.occupancy_changes[-1]),
                "runtime_seconds": float(getattr(trace, "runtime_seconds", 0.0)),
                "oracle_model_calls": int(getattr(trace, "oracle_model_calls", 0)),
                "environment_transitions": int(
                    getattr(trace, "environment_transitions", 0)
                ),
                "minimum_follower_action_gap": float(
                    getattr(trace, "minimum_follower_action_gap", float("nan"))
                ),
                "last_gradient_norm": float(
                    getattr(trace, "last_gradient_norm", float("nan"))
                ),
                "last_policy_gradient_norm": float(
                    getattr(trace, "last_policy_gradient_norm", float("nan"))
                ),
                "last_transition_gradient_norm": float(
                    getattr(trace, "last_transition_gradient_norm", float("nan"))
                ),
                "last_reward_gradient_norm": float(
                    getattr(trace, "last_reward_gradient_norm", float("nan"))
                ),
                "last_value_loss": float(
                    getattr(trace, "last_value_loss", float("nan"))
                ),
            }
        )

    _write_csv(
        output / "raw_runs.csv",
        [
            "method",
            "seed",
            "samples",
            "performative_value",
            "value_improvement",
            "occupancy_change",
        ],
        raw_rows,
    )
    final_fields = list(final_rows[0].keys()) if final_rows else []
    _write_csv(output / "final_scores.csv", final_fields, final_rows)

    samples_by_method_seed: dict[str, dict[int, set[int]]] = defaultdict(dict)
    for trace in traces:
        samples_by_method_seed[str(trace.method)][int(trace.seed)] = {
            int(value) for value in trace.samples
        }
    common_samples: dict[str, set[int]] = {}
    for method, seed_samples in samples_by_method_seed.items():
        sets = list(seed_samples.values())
        common_samples[method] = set.intersection(*sets) if sets else set()

    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"value": [], "improvement": [], "change": []}
    )
    for row in raw_rows:
        method = str(row["method"])
        samples = int(row["samples"])
        if samples not in common_samples.get(method, set()):
            continue
        key = (method, samples)
        grouped[key]["value"].append(float(row["performative_value"]))
        grouped[key]["improvement"].append(float(row["value_improvement"]))
        grouped[key]["change"].append(float(row["occupancy_change"]))

    aggregate_rows: list[dict] = []
    z_value = 1.96
    for (method, samples), values in sorted(grouped.items()):
        value_mean, value_sd, value_se, value_n = _statistics(values["value"])
        improvement_mean, improvement_sd, improvement_se, improvement_n = _statistics(
            values["improvement"]
        )
        change_mean, change_sd, change_se, change_n = _statistics(values["change"])
        # Use a support-respecting bootstrap interval for the nonnegative,
        # highly skewed occupancy-shift statistic.
        bootstrap_seed = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(f"{method}:{samples}")
        )
        change_lower, change_upper = _bootstrap_mean_interval(
            values["change"], seed=bootstrap_seed
        )
        aggregate_rows.append(
            {
                "method": method,
                "samples": samples,
                "value_mean": value_mean,
                "value_sd": value_sd,
                "value_se": value_se,
                "value_lower": value_mean - z_value * value_se,
                "value_upper": value_mean + z_value * value_se,
                "improvement_mean": improvement_mean,
                "improvement_sd": improvement_sd,
                "improvement_se": improvement_se,
                "improvement_lower": improvement_mean - z_value * improvement_se,
                "improvement_upper": improvement_mean + z_value * improvement_se,
                "occupancy_mean": change_mean,
                "occupancy_sd": change_sd,
                "occupancy_se": change_se,
                "occupancy_lower": change_lower,
                "occupancy_upper": min(change_upper, 1.0),
                "n_runs": min(value_n, improvement_n, change_n),
            }
        )

    aggregate_fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
    _write_csv(output / "figure_data.csv", aggregate_fields, aggregate_rows)

    summary_rows = []
    for method in sorted({str(row["method"]) for row in final_rows}):
        method_rows = [row for row in final_rows if row["method"] == method]
        final_mean, final_sd, final_se, n = _statistics(
            [float(row["final_performative_value"]) for row in method_rows]
        )
        improvement_mean, improvement_sd, improvement_se, _ = _statistics(
            [float(row["final_improvement"]) for row in method_rows]
        )
        improvement_lower = improvement_mean - z_value * improvement_se
        improvement_upper = improvement_mean + z_value * improvement_se
        if n < 2:
            conclusion = "insufficient_runs"
        elif improvement_lower > 0.0:
            conclusion = "positive"
        elif improvement_upper < 0.0:
            conclusion = "negative"
        else:
            conclusion = "inconclusive"
        summary_rows.append(
            {
                "method": method,
                "final_value_mean": final_mean,
                "final_value_sd": final_sd,
                "final_value_se": final_se,
                "final_value_lower": final_mean - z_value * final_se,
                "final_value_upper": final_mean + z_value * final_se,
                "final_improvement_mean": improvement_mean,
                "final_improvement_sd": improvement_sd,
                "final_improvement_se": improvement_se,
                "final_improvement_lower": improvement_lower,
                "final_improvement_upper": improvement_upper,
                "improvement_conclusion_95ci": conclusion,
                "n_runs": n,
            }
        )
    summary_rows.sort(
        key=lambda row: float(row["final_improvement_mean"]), reverse=True
    )
    _write_csv(
        output / "summary.csv",
        [
            "method",
            "final_value_mean",
            "final_value_sd",
            "final_value_se",
            "final_value_lower",
            "final_value_upper",
            "final_improvement_mean",
            "final_improvement_sd",
            "final_improvement_se",
            "final_improvement_lower",
            "final_improvement_upper",
            "improvement_conclusion_95ci",
            "n_runs",
        ],
        summary_rows,
    )

    # Independent-run differences in final improvement.  The fixed benchmark
    # has no between-seed problem variation, so these intervals quantify only
    # stochastic optimizer variation.  PePG and the exact oracle are reported
    # as separate references because they have stronger information access.
    def write_reference_comparison(reference_method: str, filename: str) -> None:
        reference = next(
            (row for row in summary_rows if row["method"] == reference_method),
            None,
        )
        comparison_rows = []
        if reference is not None:
            for row in summary_rows:
                if row["method"] == reference_method:
                    continue
                difference = float(row["final_improvement_mean"]) - float(
                    reference["final_improvement_mean"]
                )
                difference_se = math.sqrt(
                    float(row["final_improvement_se"]) ** 2
                    + float(reference["final_improvement_se"]) ** 2
                )
                lower = difference - z_value * difference_se
                upper = difference + z_value * difference_se
                comparison_rows.append(
                    {
                        "method": row["method"],
                        "reference": reference_method,
                        "final_improvement_difference": difference,
                        "difference_se": difference_se,
                        "difference_lower_95ci": lower,
                        "difference_upper_95ci": upper,
                        "conclusion_95ci": (
                            "insufficient_runs"
                            if min(int(row["n_runs"]), int(reference["n_runs"])) < 2
                            else "better"
                            if lower > 0.0
                            else "worse"
                            if upper < 0.0
                            else "inconclusive"
                        ),
                    }
                )
        _write_csv(
            output / filename,
            [
                "method",
                "reference",
                "final_improvement_difference",
                "difference_se",
                "difference_lower_95ci",
                "difference_upper_95ci",
                "conclusion_95ci",
            ],
            comparison_rows,
        )

    write_reference_comparison("VanillaPG", "comparison_to_vanilla.csv")
    write_reference_comparison("PePG", "comparison_to_pepg.csv")
    write_reference_comparison("OraclePePG", "comparison_to_oracle.csv")

    runtime_rows = []
    for method in sorted({str(row["method"]) for row in final_rows}):
        method_rows = [row for row in final_rows if row["method"] == method]
        runtime_mean, runtime_sd, runtime_se, runtime_n = _statistics(
            [float(row["runtime_seconds"]) for row in method_rows]
        )
        oracle_calls = [int(row["oracle_model_calls"]) for row in method_rows]
        runtime_rows.append(
            {
                "method": method,
                "runtime_seconds_mean": runtime_mean,
                "runtime_seconds_sd": runtime_sd,
                "runtime_seconds_se": runtime_se,
                "oracle_model_calls_mean": float(np.mean(oracle_calls)),
                "n_runs": runtime_n,
            }
        )
    _write_csv(
        output / "runtime_summary.csv",
        [
            "method",
            "runtime_seconds_mean",
            "runtime_seconds_sd",
            "runtime_seconds_se",
            "oracle_model_calls_mean",
            "n_runs",
        ],
        runtime_rows,
    )

    metadata_rows = []
    access_notes = {
        "VanillaPG": (
            "on-policy trajectory returns and score-function gradients at the "
            "currently deployed policy"
        ),
        "ZO_TG": "paired diagonal deployments at theta+ and theta-",
        "ZO_OG": "one-point diagonal deployment returns with a moving baseline",
        "ZO_OGVR": (
            "one-point diagonal deployment returns with a finite-window, "
            "distance-weighted variance-reduction baseline"
        ),
        "GZO_NS": (
            "on-policy score-gradient guide plus paired diagonal deployments"
        ),
        "GZO_HS": (
            "historical score-gradient guide plus paired diagonal deployments"
        ),
        "GaussianPISO": (
            "on-policy score-gradient component plus paired diagonal deployments"
        ),
        "CyclePISO": (
            "on-policy score-gradient component plus paired diagonal deployments"
        ),
        "GaussianPISO2": (
            "on-policy score-gradient component plus paired diagonal deployments"
        ),
        "CyclePISO2": (
            "on-policy score-gradient component plus paired diagonal deployments"
        ),
        "PePG": (
            "real trajectories plus exact reward/transition response derivatives "
            "and exact tabular advantages"
        ),
        "OraclePePG": (
            "exact full performative gradient and exact objective values for "
            "monotone line search"
        ),
    }
    for method in list(config.get("methods", {}).get("enabled", [])):
        is_oracle = method == "OraclePePG"
        is_pepg = method == "PePG"
        metadata_rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "access_class": (
                    "privileged exact-model oracle"
                    if is_oracle
                    else "privileged derivative-aided trajectory method"
                    if is_pepg
                    else "model-free diagonal-deployment access"
                ),
                "access_details": access_notes.get(method, "trajectory access"),
                "uses_exact_reward_transition_derivatives": bool(is_oracle or is_pepg),
                "uses_exact_full_performative_gradient": bool(is_oracle),
                "uses_exact_objective_line_search": bool(is_oracle),
                "uses_cross_deployment_queries": False,
                "primary_budget": (
                    "trajectory-equivalent update axis; no sampled trajectories"
                    if is_oracle
                    else "real environment trajectories"
                ),
                "interpretation": (
                    "privileged upper-information reference, not a sample-matched method"
                    if is_oracle
                    else "sample-budget PePG with privileged exact response derivatives"
                    if is_pepg
                    else "included in the model-free sample-budget comparison"
                ),
            }
        )
    _write_csv(
        output / "method_metadata.csv",
        [
            "method",
            "display_name",
            "access_class",
            "access_details",
            "uses_exact_reward_transition_derivatives",
            "uses_exact_full_performative_gradient",
            "uses_exact_objective_line_search",
            "uses_cross_deployment_queries",
            "primary_budget",
            "interpretation",
        ],
        metadata_rows,
    )

    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    report_config = config.get("report", {})
    smoothing_window = int(report_config.get("smoothing_window", 5))
    plotting_methods = report_config.get("plotting_methods")
    show_raw_curves = bool(report_config.get("show_raw_curves", False))
    total_iterations = int(report_config.get("total_iterations", 1000))
    _plot(
        aggregate_rows,
        output,
        smoothing_window=smoothing_window,
        plotting_methods=plotting_methods,
        show_raw_curves=show_raw_curves,
        total_iterations=total_iterations,
    )


def _format_axis(axis: object) -> None:
    axis.xaxis.label.set_size(17.5)
    axis.yaxis.label.set_size(17.5)
    axis.xaxis.label.set_weight("bold")
    axis.yaxis.label.set_weight("bold")
    axis.title.set_size(17.5)
    axis.title.set_weight("bold")
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize(13.5)
        tick.set_fontweight("bold")


def _legend(
    figure: object,
    handles: list[object],
    labels: list[str],
    *,
    y: float,
) -> None:
    """Draw the paper legend as one compact row close to the axes."""
    if not handles:
        return
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        prop={"size": 13.0, "weight": "bold"},
        handlelength=1.15,
        columnspacing=0.40,
        handletextpad=0.24,
        borderaxespad=0.0,
    )


def _plot(
    rows: list[dict],
    output: Path,
    *,
    smoothing_window: int = 5,
    plotting_methods: list[str] | None = None,
    show_raw_curves: bool = False,
    total_iterations: int = 1000,
) -> None:
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")

    available_methods: list[str] = []
    for row in rows:
        method = str(row["method"])
        if method not in available_methods:
            available_methods.append(method)

    if plotting_methods is None:
        methods = available_methods
    else:
        # The configuration controls both selection and legend order. During
        # grouped/partial runs, methods that do not yet have rows are skipped.
        methods = [
            method for method in plotting_methods if method in available_methods
        ]

    if not methods:
        raise ValueError(
            "none of report.plotting_methods are present in figure_data.csv"
        )

    # Two panels need enough horizontal room for a single-row legend, but the
    # canvas stays compact enough for a two-column paper figure.
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.6))
    return_axis, stability_axis = axes

    for method in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["samples"]),
        )
        iterations = np.linspace(
            0.0, float(total_iterations), num=len(selected), dtype=float
        )
        value_mean = np.asarray(
            [row["value_mean"] for row in selected], dtype=float
        )
        value_lower = np.asarray(
            [row["value_lower"] for row in selected], dtype=float
        )
        value_upper = np.asarray(
            [row["value_upper"] for row in selected], dtype=float
        )
        occupancy_mean = np.asarray(
            [row["occupancy_mean"] for row in selected], dtype=float
        )
        occupancy_lower = np.asarray(
            [row["occupancy_lower"] for row in selected], dtype=float
        )
        occupancy_upper = np.asarray(
            [row["occupancy_upper"] for row in selected], dtype=float
        )

        smooth_mean = _moving_average(value_mean, smoothing_window)
        smooth_lower = _moving_average(value_lower, smoothing_window)
        smooth_upper = _moving_average(value_upper, smoothing_window)

        style = STYLES.get(method, {"linestyle": "-"})
        marker = style.get("marker")
        markevery = max(1, int(math.ceil(len(iterations) / 10))) if marker else None

        # Raw means are available only as an explicit debugging layer. Keeping
        # them off by default avoids the thin hollow duplicate-line artifact.
        if show_raw_curves:
            return_axis.plot(
                iterations,
                value_mean,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=0.8,
                alpha=0.18,
                label="_nolegend_",
                zorder=2,
            )

        return_axis.fill_between(
            iterations,
            smooth_lower,
            smooth_upper,
            color=style.get("color"),
            alpha=0.16,
            linewidth=0,
            edgecolor="none",
            zorder=1,
        )
        return_axis.plot(
            iterations,
            smooth_mean,
            label=DISPLAY_NAMES.get(method, method),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=marker,
            markersize=5.5,
            markevery=markevery,
            linewidth=2.3,
            zorder=4,
        )

        positive_mean = np.maximum(occupancy_mean, 1e-8)
        positive_lower = np.maximum(occupancy_lower, 1e-8)
        positive_upper = np.maximum(occupancy_upper, 1e-8)
        stability_axis.fill_between(
            iterations,
            _moving_average(positive_lower, smoothing_window),
            _moving_average(positive_upper, smoothing_window),
            color=style.get("color"),
            alpha=0.16,
            linewidth=0,
            edgecolor="none",
            zorder=1,
        )
        stability_axis.plot(
            iterations,
            _moving_average(positive_mean, smoothing_window),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=marker,
            markersize=5.5,
            markevery=markevery,
            linewidth=2.3,
            zorder=4,
        )

    return_axis.set_xlim(0.0, float(total_iterations))
    return_axis.set_xlabel(r"iteration $t$")
    return_axis.set_ylabel(r"$\hat{V}^{\pi}_{\pi}$")
    return_axis.grid(alpha=0.30)
    _format_axis(return_axis)

    stability_axis.set_xlim(0.0, float(total_iterations))
    stability_axis.set_xlabel(r"iteration $t$")
    stability_axis.set_ylabel(r"$\Vert d_{t+1} - d_t \Vert_2$")
    stability_axis.set_yscale("log")
    stability_axis.set_ylim(bottom=1e-8, top=1.05)
    stability_axis.grid(alpha=0.30)
    _format_axis(stability_axis)

    handles, labels = return_axis.get_legend_handles_labels()
    _legend(fig, handles, labels, y=0.018)
    fig.tight_layout(rect=(0.0, 0.125, 1.0, 1.0), w_pad=2.0)
    fig.savefig(output / "figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "figure.pdf", bbox_inches="tight")
    plt.close(fig)

    # Keep an unsmoothed absolute-return figure for auditability, with the same
    # paper typography, styles, and one-row legend.
    absolute_fig, absolute_axis = plt.subplots(figsize=(16.8, 6.4))
    for method in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["samples"]),
        )
        iterations = np.linspace(
            0.0, float(total_iterations), num=len(selected), dtype=float
        )
        mean = np.asarray([row["value_mean"] for row in selected], dtype=float)
        lower = np.asarray([row["value_lower"] for row in selected], dtype=float)
        upper = np.asarray([row["value_upper"] for row in selected], dtype=float)
        style = STYLES.get(method, {"linestyle": "-"})
        marker = style.get("marker")
        markevery = max(1, int(math.ceil(len(iterations) / 10))) if marker else None

        absolute_axis.fill_between(
            iterations,
            lower,
            upper,
            color=style.get("color"),
            alpha=0.14,
            linewidth=0,
            edgecolor="none",
            zorder=1,
        )
        absolute_axis.plot(
            iterations,
            mean,
            label=DISPLAY_NAMES.get(method, method),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=marker,
            markersize=5.5,
            markevery=markevery,
            linewidth=2.3,
            zorder=3,
        )

    absolute_axis.set_xlim(0.0, float(total_iterations))
    absolute_axis.set_xlabel(r"iteration $t$")
    absolute_axis.set_ylabel(r"$\hat{V}^{\pi}_{\pi}$")
    absolute_axis.grid(alpha=0.30)
    _format_axis(absolute_axis)

    handles, labels = absolute_axis.get_legend_handles_labels()
    _legend(absolute_fig, handles, labels, y=0.018)
    absolute_fig.tight_layout(rect=(0.0, 0.125, 1.0, 1.0))
    absolute_fig.savefig(
        output / "figure_absolute.png", dpi=300, bbox_inches="tight"
    )
    absolute_fig.savefig(output / "figure_absolute.pdf", bbox_inches="tight")
    plt.close(absolute_fig)
