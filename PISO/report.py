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
}


def write_outputs(results: list[dict], methods: list[str], figure_run: int, output_dir: Path) -> None:
    final_rows = []
    figure_rows = []
    for item in results:
        trace = item["trace"]
        final_rows.append(
            {
                "week": item["week_label"],
                "run": item["run"],
                "method": item["method"],
                "objective": trace.objectives[-1],
            }
        )
        if item["run"] == figure_run:
            for samples, objective in zip(trace.samples, trace.objectives, strict=True):
                figure_rows.append(
                    {
                        "week": item["week_label"],
                        "method": item["method"],
                        "samples": samples,
                        "objective": objective,
                    }
                )

    final = pd.DataFrame(final_rows)
    figure_data = pd.DataFrame(figure_rows)
    summary_rows = []
    for week, frame in final.groupby("week", sort=False):
        row = {"week": week}
        for method in methods:
            values = frame.loc[frame["method"] == method, "objective"]
            row[f"{method}_obj"] = values.mean()
            row[f"{method}_sd"] = values.std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    final.to_csv(output_dir / "final_scores.csv", index=False)
    figure_data.to_csv(output_dir / "figure_data.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    _plot(figure_data, methods, output_dir / "figure.png", output_dir / "figure.pdf")


def _plot(data: pd.DataFrame, methods: list[str], png_path: Path, pdf_path: Path) -> None:
    weeks = list(data["week"].drop_duplicates())
    ncols = 3
    nrows = (len(weeks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6.8), squeeze=False)
    for axis, week in zip(axes.flat, weeks):
        frame = data[data["week"] == week]
        for method in methods:
            values = frame[frame["method"] == method]
            axis.plot(
                values["samples"],
                values["objective"],
                label=method,
                linewidth=1.2,
                markersize=3.5,
                markevery=max(1, len(values) // 10),
                **STYLES.get(method, {}),
            )
        axis.set_title(week)
        axis.set_xlabel("sample number")
        axis.set_ylabel("obj")
        axis.grid(True, alpha=0.45)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(weeks):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
