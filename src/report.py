from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLES = {
    "GZO_NS": {"color": "blue", "marker": "o"},
    "GZO_HS": {"color": "cyan", "marker": "s"},
    "ZO_TG": {"color": "green", "marker": "^"},
    "ZO_OG": {"color": "red", "marker": "D"},
    "ZO_OGVR": {"color": "black", "marker": "*"},
    "PISO": {"color": "purple", "marker": "X"},
    "PISO_M": {"color": "orange", "marker": "P"},
}
_SWEEP_METHODS = {"PISO", "PISO_M"}


def _selected_alphas(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sweep = final[final["method"].isin(_SWEEP_METHODS)]
    for (week_id, week, method), frame in sweep.groupby(
        ["week_id", "week", "method"],
        sort=False,
    ):
        stats = (
            frame.groupby("residual_alpha", as_index=False)["objective"]
            .agg(mean_objective="mean", sd_objective="std")
            .sort_values(["mean_objective", "residual_alpha"], kind="stable")
        )
        best = stats.iloc[0]
        rows.append(
            {
                "week_id": week_id,
                "week": week,
                "method": method,
                "residual_alpha": float(best["residual_alpha"]),
                "mean_objective": float(best["mean_objective"]),
                "sd_objective": float(best["sd_objective"]),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    results: list[dict],
    methods: list[str],
    figure_run: int,
    output_dir: Path,
) -> None:
    final_rows = []
    for item in results:
        trace = item["trace"]
        final_rows.append(
            {
                "week_id": item["week_id"],
                "week": item["week_label"],
                "run": item["run"] + 1,
                "method": item["method"],
                "residual_alpha": item["residual_alpha"],
                "objective": trace.objectives[-1],
            }
        )
    final = pd.DataFrame(final_rows)
    selected = _selected_alphas(final)

    selected_lookup = {
        (row.week_id, row.method): float(row.residual_alpha)
        for row in selected.itertuples(index=False)
    }

    summary_rows = []
    for (week_id, week), frame in final.groupby(["week_id", "week"], sort=False):
        row: dict[str, object] = {"week_id": week_id, "week": week}
        for method in methods:
            values = frame[frame["method"] == method]
            if method in _SWEEP_METHODS:
                alpha = selected_lookup[(week_id, method)]
                values = values[values["residual_alpha"] == alpha]
                row[f"{method}_alpha"] = alpha
            objectives = values["objective"]
            row[f"{method}_obj"] = objectives.mean()
            row[f"{method}_sd"] = objectives.std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    figure_rows = []
    for item in results:
        if item["run"] != figure_run:
            continue
        method = item["method"]
        alpha = item["residual_alpha"]
        if method in _SWEEP_METHODS:
            selected_alpha = selected_lookup[(item["week_id"], method)]
            if float(alpha) != selected_alpha:
                continue
        trace = item["trace"]
        for samples, objective in zip(trace.samples, trace.objectives, strict=True):
            figure_rows.append(
                {
                    "week_id": item["week_id"],
                    "week": item["week_label"],
                    "method": method,
                    "residual_alpha": alpha,
                    "samples": samples,
                    "objective": objective,
                }
            )
    figure_data = pd.DataFrame(figure_rows)

    final.to_csv(output_dir / "final_scores.csv", index=False)
    selected.to_csv(output_dir / "selected_alphas.csv", index=False)
    figure_data.to_csv(output_dir / "figure_data.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    _plot(figure_data, methods, output_dir / "figure.png", output_dir / "figure.pdf")


def _plot(data: pd.DataFrame, methods: list[str], png_path: Path, pdf_path: Path) -> None:
    weeks = list(data[["week_id", "week"]].drop_duplicates().itertuples(index=False, name=None))
    ncols = 3
    nrows = (len(weeks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6.8), squeeze=False)
    for axis, (week_id, week) in zip(axes.flat, weeks):
        frame = data[data["week_id"] == week_id]
        for method in methods:
            values = frame[frame["method"] == method]
            if values.empty:
                continue
            label = method
            if method in _SWEEP_METHODS:
                alpha = float(values["residual_alpha"].iloc[0])
                label = f"{method} (alpha={alpha:g})"
            axis.plot(
                values["samples"],
                values["objective"],
                label=label,
                linewidth=1.2,
                markersize=3.5,
                markevery=max(1, len(values) // 10),
                **STYLES.get(method, {}),
            )
        axis.set_title(week)
        axis.set_xlabel("sample number")
        axis.set_ylabel("obj")
        axis.grid(True, alpha=0.45)
        axis.legend(fontsize=7.5)
    for axis in axes.flat[len(weeks):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
