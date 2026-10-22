import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common import STYLES, RunStore, hash_seed, run_cached, label, legend_below, save_figure, short_hash, style_axis
from .methods import METHODS, PRIVATE_SEED_ALIAS, STREAM_PADDING, zo_og
from .problem import StrategicClassification, load_dataset


def run(config, output_dir, jobs=1):
    experiment, methods = config["experiment"], config["methods"]
    seed, simulations = int(experiment["seed"]), int(experiment["simulations"])
    taus = [float(tau) for tau in experiment["taus"]]
    dataset = load_dataset(config["problem"])
    store, tag = RunStore(output_dir / "runs"), short_hash(config)
    rng = np.random.RandomState(seed)
    budget = int(experiment["max_samples"]), 0
    rows, finals = [], []
    with tqdm(total=len(taus) * simulations * len(methods), desc="classification") as progress:
        for tau in taus:
            for run_index in range(simulations):
                problem = StrategicClassification(dataset, tau, config["problem"])
                for name, params in methods.items():
                    alias = PRIVATE_SEED_ALIAS.get(name)
                    job_rng = rng if alias is None else np.random.RandomState(
                        hash_seed(f"classification:{seed}:{tau:.8g}:{run_index}:{alias}"))
                    key = f"{tag}_{tau:g}_{run_index}_{name}"
                    trace = run_cached(store, key, job_rng, lambda: METHODS[name](problem, params, job_rng, *budget))
                    if name == "ZO_TG":
                        run_cached(store, f"{key}_padding", rng, lambda: zo_og(problem, STREAM_PADDING, rng, *budget))
                    rows += [{"tau": tau, "run": run_index + 1, "method": name, "samples": s, "train_loss": o}
                             for s, o in zip(trace.samples, trace.objectives)]
                    finals.append({"tau": tau, "run": run_index + 1, "method": name, **problem.metrics(trace.final_x)})
                    progress.update()
    report(pd.DataFrame(rows), pd.DataFrame(finals), config, output_dir)


def report(trajectories, finals, config, output_dir):
    trajectories.to_csv(output_dir / "trajectories.csv", index=False)
    finals.to_csv(output_dir / "final_scores.csv", index=False)
    summary = finals.drop(columns="run").groupby(["tau", "method"], sort=False).agg(["mean", "sem"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "summary.csv", index=False)
    stats = (trajectories.groupby(["tau", "method", "samples"], sort=False)["train_loss"]
             .agg(mean="mean", sd="std", n="count").reset_index())
    stats["se"] = stats["sd"].fillna(0.0) / np.sqrt(stats["n"])
    stats.to_csv(output_dir / "figure_data.csv", index=False)
    plot(stats, config["plotting"]["methods"], output_dir / "classification")


def plot(stats, methods, stem):
    taus = sorted(stats["tau"].unique())
    figure = plt.figure(figsize=(17.2, 9.4))
    grid = figure.add_gridspec(2, 6)
    slots = [grid[0, 0:2], grid[0, 2:4], grid[0, 4:6], grid[1, 1:3], grid[1, 3:5], grid[1, 0:2]]
    handles = {}
    for slot, tau in zip(slots, taus):
        axis = figure.add_subplot(slot)
        frame = stats[stats["tau"] == tau]
        for method in methods:
            values = frame[frame["method"] == method].sort_values("samples")
            x, mean, se = values["samples"].to_numpy(float), values["mean"].to_numpy(float), values["se"].to_numpy(float)
            axis.fill_between(x, mean - se, mean + se, color=STYLES[method]["color"], alpha=0.18, linewidth=0.0, zorder=1)
            (line,) = axis.plot(x, mean, label=label(method), linewidth=2.2, markersize=5.5,
                                markevery=max(1, len(values) // 10), zorder=3, **STYLES[method])
            handles.setdefault(method, line)
        axis.set_title(rf"$\tau={tau:g}$")
        axis.set_xlabel("sample number")
        axis.set_ylabel("train loss")
        axis.grid(True, alpha=0.35)
        style_axis(axis)
    legend_below(figure, list(handles.values()), [label(m) for m in handles], y=0.01)
    figure.tight_layout(rect=(0.0, 0.125, 1.0, 1.0))
    save_figure(figure, stem, dpi=240)
