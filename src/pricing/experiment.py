import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common import STYLES, RunStore, hash_seed, run_cached, label, legend_below, save_figure, short_hash, style_axis
from .methods import METHODS, PRIVATE_SEED_ALIAS, STREAM_PADDING, zo_og
from .problem import WEEKS, PricingProblem


def run(config, output_dir, jobs=1):
    experiment, spec, methods = config["experiment"], config["problem"], config["methods"]
    seed, simulations = int(experiment["seed"]), int(experiment["simulations"])
    weeks = [str(week).zfill(2) for week in experiment["weeks"]]
    store, tag = RunStore(output_dir / "runs"), short_hash(config)
    rng = np.random.RandomState(seed)
    budget = int(experiment["max_samples"]), int(experiment["metric_samples"])
    rows = []
    with tqdm(total=len(weeks) * simulations * len(methods), desc="pricing") as progress:
        for week in weeks:
            for run_index in range(simulations):
                rho = spec["rho_low"] + (spec["rho_high"] - spec["rho_low"]) * rng.rand(spec["products"])
                problem = PricingProblem(week, rho, spec)
                for name, params in methods.items():
                    alias = PRIVATE_SEED_ALIAS.get(name)
                    job_rng = rng if alias is None else np.random.RandomState(hash_seed(f"{seed}:{week}:{run_index}:{alias}"))
                    key = f"{tag}_{week}_{run_index}_{name}"
                    trace = run_cached(store, key, job_rng, lambda: METHODS[name](problem, params, job_rng, *budget))
                    if name == "ZO_TG":
                        run_cached(store, f"{key}_padding", rng, lambda: zo_og(problem, STREAM_PADDING, rng, *budget))
                    rows += [{"week": week, "run": run_index + 1, "method": name, "samples": s, "objective": o}
                             for s, o in zip(trace.samples, trace.objectives)]
                    progress.update()
    report(pd.DataFrame(rows), config, output_dir)


def report(trajectories, config, output_dir):
    trajectories.to_csv(output_dir / "trajectories.csv", index=False)
    stats = (trajectories.groupby(["week", "method", "samples"], sort=False)["objective"]
             .agg(mean="mean", sd="std", n="count").reset_index())
    stats["sd"] = stats["sd"].fillna(0.0)
    stats["se"] = stats["sd"] / np.sqrt(stats["n"])
    stats.to_csv(output_dir / "figure_data.csv", index=False)
    final = trajectories.groupby(["week", "method", "run"], sort=False).tail(1)
    summary = final.groupby(["week", "method"], sort=False)["objective"].agg(mean="mean", sd="std", n="count").reset_index()
    summary["sd"] = summary["sd"].fillna(0.0)
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot(stats, config["plotting"], output_dir / "pricing")


def plot(stats, plotting, stem):
    methods = plotting["methods"]
    rows = (stats["week"].nunique() + 2) // 3
    figure, axes = plt.subplots(rows, 3, figsize=(15.6, 4.6 * rows), squeeze=False)
    handles = {}
    for axis, week in zip(axes.flat, stats["week"].unique()):
        frame = stats[(stats["week"] == week) & stats["method"].isin(methods)]
        window = []
        for method in methods:
            values = frame[frame["method"] == method].sort_values("samples")
            if values.empty:
                continue
            x, mean, se = values["samples"].to_numpy(float), values["mean"].to_numpy(float), values["se"].to_numpy(float)
            (line,) = axis.plot(x, mean, label=label(method), linewidth=2.2, markersize=5.5,
                                markevery=max(1, len(values) // 10), **STYLES[method])
            axis.fill_between(x, mean - se, mean + se, color=line.get_color(), alpha=0.18, linewidth=0.0)
            handles.setdefault(method, line)
            cutoff = x.max() - plotting["final_window_fraction"] * (x.max() - x.min())
            window += [(mean - se)[x >= cutoff], (mean + se)[x >= cutoff]]
        window = np.concatenate(window)
        low, high = window.min(), window.max()
        margin = plotting["y_margin_fraction"] * (high - low) if high > low else max(abs(low) * 0.1, 0.5)
        axis.set_ylim(low - margin, high + margin)
        axis.set_title(WEEKS.get(week, week))
        axis.set_xlabel("sample number")
        axis.set_ylabel("objective")
        axis.grid(True, alpha=0.45)
        style_axis(axis)
    for axis in axes.flat[len(stats["week"].unique()):]:
        axis.set_visible(False)
    legend_below(figure, list(handles.values()), [label(m) for m in handles])
    figure.tight_layout(rect=(0.0, 0.23 / rows, 1.0, 1.0))
    save_figure(figure, stem)
