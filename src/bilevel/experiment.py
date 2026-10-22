from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common import STYLES, RunStore, label, legend_below, save_figure, short_hash, style_axis
from .methods import run_method
from .problems import GENERATORS


def _task(args):
    name, params, problem, seed, instance_id, iterations = args
    return run_method(name, params, problem, seed, instance_id, iterations)


def run(config, output_dir, jobs=1):
    family, experiment = config["family"], config["experiment"]
    seed, iterations = int(experiment["seed"]), int(experiment["iterations"])
    instances = GENERATORS[family](config, seed)
    store = RunStore(output_dir / "runs")
    tasks = [(name, params, problem, seed, i, iterations)
             for i, problem in enumerate(instances) for name, params in config["methods"].items()]
    keys = [short_hash([family, experiment, config["problem"], name, params, i]) for name, params, _, _, i, _ in tasks]
    pending = [(task, key) for task, key in zip(tasks, keys) if store.load(key) is None]
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool, tqdm(total=len(pending), desc=family) as bar:
        for (_, key), result in zip(pending, pool.map(_task, [task for task, _ in pending], chunksize=4)):
            store.save(key, result)
            bar.update()
    rows = []
    for (name, _, problem, _, i, _), key in zip(tasks, keys):
        objectives, calls = store.load(key)
        rows += [{"instance": i, "dimension": problem.dimension, "method": name, "iteration": t,
                  "objective": o, "oracle_calls": c} for t, (o, c) in enumerate(zip(objectives, calls))]
    report(pd.DataFrame(rows), config, output_dir)


def report(raw, config, output_dir):
    family, plotting = config["family"], config["plotting"]
    reference = raw[raw["method"].isin(plotting["normalization_methods"])]
    extrema = reference.groupby("instance")["objective"].agg(low="min", high="max").reset_index()
    raw = raw.merge(extrema, on="instance", how="left")
    span = raw["high"] - raw["low"]
    raw["normalized"] = np.where(span > 1.0e-14, (raw["objective"] - raw["low"]) / span, 0.0)
    raw.drop(columns=["low", "high"]).to_csv(output_dir / "runs.csv", index=False)
    methods = plotting["methods"]
    summary = (raw[raw["method"].isin(methods)].groupby(["method", "iteration"])["normalized"]
               .quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack().reset_index())
    summary.columns = ["method", "iteration", "p10", "p25", "median", "p75", "p90"]
    summary.to_csv(output_dir / "trajectory_summary.csv", index=False)
    plot_trajectories(summary, methods, output_dir / f"{family}_trajectories")
    if family == "routing":
        for method in plotting["pairwise_methods"]:
            plot_pairwise(raw, plotting["scatter_iterations"], method, output_dir / f"routing_pairwise_{method}")
    else:
        plot_by_dimension(raw, plotting["dimension_iterations"], plotting["dimension_plot_min"], methods,
                          output_dir / "security_by_dimension")


def plot_trajectories(summary, methods, stem):
    figure, axis = plt.subplots(figsize=(10.4, 6.2))
    for method in methods:
        frame = summary[summary["method"] == method].sort_values("iteration")
        x, style = frame["iteration"].to_numpy(float), STYLES[method]
        axis.fill_between(x, frame["p10"], frame["p90"], color=style["color"], alpha=0.10, linewidth=0)
        axis.fill_between(x, frame["p25"], frame["p75"], color=style["color"], alpha=0.22, linewidth=0)
        axis.plot(x, frame["median"], linewidth=2.35 if method in ("ZOS", "PZOS") else 2.2, markersize=5.6,
                  markevery=max(1, len(frame) // 10), label=label(method), **style)
    axis.set_xlabel("iteration")
    axis.set_ylabel("normalized objective")
    axis.grid(alpha=0.28)
    style_axis(axis, 18.0, 14.0, 18.0)
    legend_below(figure, *axis.get_legend_handles_labels(), size=12.5, y=0.012)
    figure.tight_layout(rect=(0.0, max(0.05, min(0.12, 0.58 / figure.get_figheight())), 1.0, 1.0))
    save_figure(figure, stem)


def plot_pairwise(raw, snapshots, method, stem):
    references = ["PZOS", "ZOS", "GZO_NS", "GZO_HS"]
    pivot = raw.pivot_table(index=["instance", "dimension"], columns=["method", "iteration"], values="normalized", aggfunc="last")
    values = np.concatenate([pivot[(m, s)].to_numpy() for m in [method, *references] for s in snapshots])
    margin = 0.06 * max(float(values.max() - values.min()), 0.10)
    low, high = min(0.0, float(values.min()) - margin), max(1.0, float(values.max()) + margin)
    figure, axes = plt.subplots(len(references), len(snapshots), figsize=(4.6 * len(snapshots) + 1.05, 3.55 * len(references) + 1.55),
                                sharex=True, sharey=True, squeeze=False)
    for row, reference in enumerate(references):
        for column, snapshot in enumerate(snapshots):
            axis = axes[row, column]
            image = axis.scatter(pivot[(reference, snapshot)], pivot[(method, snapshot)], c=pivot.index.get_level_values("dimension"),
                                 cmap="turbo", s=32, alpha=0.88, edgecolors="none")
            axis.plot([low, high], [low, high], "k--", linewidth=1.85, label="equal performance")
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(f"t = {snapshot}")
            axis.set_xlabel(f"normalized {label(reference)} objective")
            style_axis(axis, 15.0, 12.5, 17.0)
    figure.supylabel(f"normalized {label(method)} objective", x=0.022, fontsize=17.0, fontweight="bold")
    legend_below(figure, *axes[0, 0].get_legend_handles_labels(), size=12.5, y=0.012)
    figure.subplots_adjust(left=0.105, right=0.865, bottom=0.075, top=0.965, hspace=0.48, wspace=0.24)
    colorbar = figure.colorbar(image, cax=figure.add_axes([0.9, 0.135, 0.02, 0.725]))
    colorbar.set_label("problem dimension $d_x$", fontsize=15.0, fontweight="bold", labelpad=11.0)
    save_figure(figure, stem, dpi=240)


def plot_by_dimension(raw, snapshots, dimension_min, methods, stem):
    data = (raw[(raw["dimension"] >= dimension_min) & raw["iteration"].isin(snapshots)]
            .groupby(["method", "dimension", "iteration"])["normalized"].agg(["median", "min", "max"]).reset_index())
    data.to_csv(f"{stem}.csv", index=False)
    figure, axes = plt.subplots(1, len(snapshots), figsize=(max(14.2, 4.55 * len(snapshots)), 5.85), sharex=True, sharey=True, squeeze=False)
    for axis, snapshot in zip(axes[0], snapshots):
        for method in methods:
            frame = data[(data["method"] == method) & (data["iteration"] == snapshot)].sort_values("dimension")
            x, style = frame["dimension"].to_numpy(float), STYLES[method]
            axis.fill_between(x, frame["min"], frame["max"], color=style["color"], alpha=0.08, linewidth=0)
            axis.plot(x, frame["median"], linewidth=2.35 if method in ("ZOS", "PZOS") else 2.2, markersize=5.4,
                      markevery=max(1, len(frame) // 10), label=label(method), **style)
        axis.set_title(f"t = {snapshot}")
        axis.set_xlabel("problem dimension $d_x$")
        axis.set_xlim(dimension_min, int(data["dimension"].max()))
        axis.grid(alpha=0.25)
        style_axis(axis, 16.5, 13.0, 17.0)
    axes[0, 0].set_ylabel("normalized objective")
    style_axis(axes[0, 0], 16.5, 13.0, 17.0)
    legend_below(figure, *axes[0, 0].get_legend_handles_labels(), size=12.5, y=0.012)
    figure.tight_layout(rect=(0.0, max(0.05, min(0.12, 0.58 / figure.get_figheight())), 1.0, 1.0))
    save_figure(figure, stem, dpi=240, tight=False)
