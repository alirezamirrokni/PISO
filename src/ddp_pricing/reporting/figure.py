from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "GZO_NS": dict(color="blue", marker="o"),
    "GZO_HS": dict(color="cyan", marker="s"),
    "ZO_TG": dict(color="green", marker="^"),
    "ZO_OG": dict(color="red", marker="D"),
    "ZO_OGVR": dict(color="black", marker="*"),
}


def write_figure(
    trajectories: pd.DataFrame,
    method_order: list[str],
    figure_run_index: int,
    output_png: Path,
    output_pdf: Path,
) -> None:
    selected = trajectories[trajectories["run_index"] == figure_run_index]
    weeks = selected[["week_id", "week_label"]].drop_duplicates().reset_index(drop=True)
    n_weeks = len(weeks)
    ncols = 3
    nrows = (n_weeks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 6.8), squeeze=False)
    for axis, week in zip(axes.flat, weeks.itertuples(index=False)):
        frame = selected[selected["week_id"] == week.week_id]
        for method in method_order:
            method_frame = frame[frame["method"] == method]
            style = STYLE.get(method, {})
            mark_every = max(1, len(method_frame) // 10)
            axis.plot(
                method_frame["sample_count"],
                method_frame["objective"],
                label=method,
                linewidth=1.2,
                markersize=3.5,
                markevery=mark_every,
                **style,
            )
        axis.set_xlabel("sample number")
        axis.set_ylabel("obj")
        axis.set_title(week.week_label)
        axis.grid(True, alpha=0.45)
        axis.legend(fontsize=8)
    for axis in axes.flat[n_weeks:]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
