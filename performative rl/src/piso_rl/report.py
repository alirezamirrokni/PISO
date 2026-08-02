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


DISPLAY_NAMES = {
    "VanillaPG": "Vanilla PG",
    "ZO_TG": "ZO-TG",
    "ZO_OG": "ZO-OG",
    "GZO_NS": "GZO-NS",
    "GZO_HS": "GZO-HS",
    "GaussianPISO": "GaussianPISO",
    "GuidedPISO": "GuidedPISO",
    "CyclePISO": "CyclePISO",
    "GaussianPISO2": "GaussianPISO²",
    "GuidedPISO2": "GuidedPISO²",
    "CyclePISO2": "CyclePISO²",
    "PePG": "PePG",
}

LINESTYLES = {
    "VanillaPG": "--",
    "ZO_TG": ":",
    "ZO_OG": ":",
    "GZO_NS": "-.",
    "GZO_HS": "-.",
    "GaussianPISO": "-",
    "GuidedPISO": "-",
    "CyclePISO": "-",
    "GaussianPISO2": "--",
    "GuidedPISO2": "--",
    "CyclePISO2": "--",
    "PePG": "-",
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


def build_reports(traces: list, config: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    final_rows: list[dict] = []
    for trace in traces:
        for samples, value, change in zip(
            trace.samples, trace.values, trace.occupancy_changes
        ):
            raw_rows.append(
                {
                    "method": trace.method,
                    "seed": trace.seed,
                    "samples": int(samples),
                    "performative_value": float(value),
                    "occupancy_change": float(change),
                }
            )
        final_rows.append(
            {
                "method": trace.method,
                "seed": trace.seed,
                "final_samples": int(trace.samples[-1]),
                "final_performative_value": float(trace.values[-1]),
                "final_occupancy_change": float(trace.occupancy_changes[-1]),
            }
        )

    _write_csv(
        output / "raw_runs.csv",
        ["method", "seed", "samples", "performative_value", "occupancy_change"],
        raw_rows,
    )
    _write_csv(
        output / "final_scores.csv",
        [
            "method",
            "seed",
            "final_samples",
            "final_performative_value",
            "final_occupancy_change",
        ],
        final_rows,
    )

    # Aggregate only sample positions shared by every seed of a method.  This
    # keeps mean/SE curves valid when a run is resumed from an older dense
    # cache while newer runs use sparse exact evaluation.
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
        lambda: {"value": [], "change": []}
    )
    for row in raw_rows:
        method = str(row["method"])
        samples = int(row["samples"])
        if samples not in common_samples.get(method, set()):
            continue
        key = (method, samples)
        grouped[key]["value"].append(float(row["performative_value"]))
        grouped[key]["change"].append(float(row["occupancy_change"]))

    aggregate_rows: list[dict] = []
    for (method, samples), values in sorted(grouped.items()):
        value_mean, value_sd, value_se, value_n = _statistics(values["value"])
        change_mean, change_sd, change_se, change_n = _statistics(values["change"])
        aggregate_rows.append(
            {
                "method": method,
                "samples": samples,
                "value_mean": value_mean,
                "value_sd": value_sd,
                "value_se": value_se,
                "value_lower": value_mean - value_se,
                "value_upper": value_mean + value_se,
                "occupancy_mean": change_mean,
                "occupancy_sd": change_sd,
                "occupancy_se": change_se,
                "occupancy_lower": max(change_mean - change_se, 0.0),
                "occupancy_upper": change_mean + change_se,
                "n_runs": min(value_n, change_n),
            }
        )

    aggregate_fields = [
        "method",
        "samples",
        "value_mean",
        "value_sd",
        "value_se",
        "value_lower",
        "value_upper",
        "occupancy_mean",
        "occupancy_sd",
        "occupancy_se",
        "occupancy_lower",
        "occupancy_upper",
        "n_runs",
    ]
    _write_csv(output / "figure_data.csv", aggregate_fields, aggregate_rows)

    final_by_method: dict[str, list[float]] = defaultdict(list)
    for row in final_rows:
        final_by_method[str(row["method"])].append(
            float(row["final_performative_value"])
        )
    summary_rows = []
    for method in sorted(final_by_method):
        mean, sd, se, n = _statistics(final_by_method[method])
        summary_rows.append(
            {
                "method": method,
                "final_value_mean": mean,
                "final_value_sd": sd,
                "final_value_se": se,
                "n_runs": n,
            }
        )
    _write_csv(
        output / "summary.csv",
        ["method", "final_value_mean", "final_value_sd", "final_value_se", "n_runs"],
        summary_rows,
    )

    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    _plot(aggregate_rows, output)


def _plot(rows: list[dict], output: Path) -> None:
    methods = []
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    value_axis, stability_axis = axes

    for method in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["samples"]),
        )
        samples = np.asarray([row["samples"] for row in selected], dtype=float)
        value_mean = np.asarray([row["value_mean"] for row in selected], dtype=float)
        value_lower = np.asarray([row["value_lower"] for row in selected], dtype=float)
        value_upper = np.asarray([row["value_upper"] for row in selected], dtype=float)
        occupancy_mean = np.asarray(
            [row["occupancy_mean"] for row in selected], dtype=float
        )
        occupancy_lower = np.asarray(
            [row["occupancy_lower"] for row in selected], dtype=float
        )
        occupancy_upper = np.asarray(
            [row["occupancy_upper"] for row in selected], dtype=float
        )

        line = value_axis.plot(
            samples,
            value_mean,
            label=DISPLAY_NAMES.get(method, method),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=1.8,
        )[0]
        value_axis.fill_between(
            samples,
            value_lower,
            value_upper,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
        )

        stability_line = stability_axis.plot(
            samples,
            occupancy_mean,
            label=DISPLAY_NAMES.get(method, method),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=1.8,
        )[0]
        stability_axis.fill_between(
            samples,
            occupancy_lower,
            occupancy_upper,
            color=stability_line.get_color(),
            alpha=0.18,
            linewidth=0,
        )

    value_axis.set_xlabel("Cumulative trajectories")
    value_axis.set_ylabel("Performative value")
    value_axis.set_title("Performative return (mean ± SE)")
    value_axis.grid(alpha=0.25)

    stability_axis.set_xlabel("Cumulative trajectories")
    stability_axis.set_ylabel("Relative occupancy change")
    stability_axis.set_title("Policy stability (mean ± SE)")
    stability_axis.set_yscale("symlog", linthresh=1e-6)
    stability_axis.grid(alpha=0.25)

    handles, labels = value_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output / "figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "figure.pdf", bbox_inches="tight")
    plt.close(fig)
