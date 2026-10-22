import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common import STYLES, RunStore, hash_seed, label, legend_below, save_figure, short_hash, style_axis
from .methods import METHODS, Recorder
from .problem import PerformativeRL


def _task(args):
    config, name, seed = args
    experiment = config["experiment"]
    problem = PerformativeRL(config["environment"], config["policy"], experiment["problem_seed"])
    method, alias = METHODS[name]
    rng = np.random.RandomState(hash_seed(f"{int(experiment['master_seed'])}|{int(seed)}|{alias}") & 0x7FFFFFFF)
    return method(problem, config["methods"][name], rng, int(experiment["max_trajectories"]),
                  Recorder(problem, experiment["evaluation_interval"]))


def run(config, output_dir, jobs=1):
    experiment = config["experiment"]
    tasks = [(config, name, seed) for seed in experiment["seeds"] for name in config["methods"]]
    keys = [short_hash([experiment, config["environment"], config["policy"], name, config["methods"][name], seed])
            for _, name, seed in tasks]
    store = RunStore(output_dir / "runs")
    pending = [(task, key) for task, key in zip(tasks, keys) if store.load(key) is None]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=context) as pool, tqdm(total=len(pending), desc="rl") as bar:
        for (_, key), result in zip(pending, pool.map(_task, [task for task, _ in pending])):
            store.save(key, result)
            bar.update()
    rows = []
    for (_, name, seed), key in zip(tasks, keys):
        result = store.load(key)
        rows += [{"method": name, "seed": seed, "samples": s, "value": v, "occupancy_change": c}
                 for s, v, c in zip(result["samples"], result["values"], result["changes"])]
    report(pd.DataFrame(rows), config, output_dir)


def bootstrap_interval(values, seed, resamples=4000):
    values = np.asarray(values, dtype=float)
    if values.size == 1 or np.all(values == values[0]):
        return float(values.mean()), float(values.mean())
    indices = np.random.default_rng(seed & 0xFFFFFFFF).integers(0, values.size, size=(resamples, values.size))
    lower, upper = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return float(max(lower, 0.0)), float(max(upper, 0.0))


def report(raw, config, output_dir):
    raw.to_csv(output_dir / "runs.csv", index=False)
    common = raw.groupby(["method", "samples"])["seed"].transform("nunique") == raw.groupby("method")["seed"].transform("nunique")
    rows = []
    for (method, samples), frame in raw[common].groupby(["method", "samples"]):
        values, changes = frame["value"].to_numpy(), frame["occupancy_change"].to_numpy()
        se = values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else 0.0
        seed = sum((i + 1) * ord(character) for i, character in enumerate(f"{method}:{samples}"))
        low, high = bootstrap_interval(changes, seed)
        rows.append({"method": method, "samples": samples, "value_mean": values.mean(), "value_lower": values.mean() - 1.96 * se,
                     "value_upper": values.mean() + 1.96 * se, "occupancy_mean": changes.mean(),
                     "occupancy_lower": low, "occupancy_upper": min(high, 1.0)})
    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "figure_data.csv", index=False)
    final = raw.groupby(["method", "seed"]).tail(1)
    summary = final.groupby("method")["value"].agg(mean="mean", sem="sem", n="count").sort_values("mean", ascending=False)
    summary.to_csv(output_dir / "summary.csv")
    plot(stats, config["report"], output_dir / "rl")


def smooth(values, window):
    return np.asarray([np.mean(values[max(0, i - window + 1): i + 1]) for i in range(len(values))])


def plot(stats, report_config, stem):
    window = int(report_config["smoothing_window"])
    figure, (value_axis, occupancy_axis) = plt.subplots(1, 2, figsize=(16.8, 6.6))
    for method in report_config["plotting_methods"]:
        frame = stats[stats["method"] == method].sort_values("samples")
        x, style = frame["samples"].to_numpy(float), STYLES[method]
        markevery = max(1, math.ceil(len(x) / 10))
        value_axis.fill_between(x, smooth(frame["value_lower"].to_numpy(), window), smooth(frame["value_upper"].to_numpy(), window),
                                color=style["color"], alpha=0.16, linewidth=0, zorder=1)
        value_axis.plot(x, smooth(frame["value_mean"].to_numpy(), window), label=label(method), markersize=5.5,
                        markevery=markevery, linewidth=2.3, zorder=4, **style)
        occupancy = {key: np.maximum(frame[f"occupancy_{key}"].to_numpy(), 1e-8) for key in ("mean", "lower", "upper")}
        occupancy_axis.fill_between(x, smooth(occupancy["lower"], window), smooth(occupancy["upper"], window),
                                    color=style["color"], alpha=0.16, linewidth=0, zorder=1)
        occupancy_axis.plot(x, smooth(occupancy["mean"], window), markersize=5.5, markevery=markevery, linewidth=2.3, zorder=4, **style)
    for axis, ylabel in ((value_axis, r"$\hat{V}^{\pi}_{\pi}$"), (occupancy_axis, "Occupancy change (normalized L1)")):
        axis.set_xlim(0.0, float(stats["samples"].max()))
        axis.set_xlabel("Trajectories")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.30)
        style_axis(axis, 17.5, 13.5, 17.5)
    occupancy_axis.set_yscale("log")
    occupancy_axis.set_ylim(bottom=1e-8, top=1.05)
    legend_below(figure, *value_axis.get_legend_handles_labels(), size=13.0)
    figure.tight_layout(rect=(0.0, 0.125, 1.0, 1.0), w_pad=2.0)
    save_figure(figure, stem)
