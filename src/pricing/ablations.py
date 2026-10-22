from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common import RunStore, hash_seed, save_figure, short_hash, style_axis
from .methods import haar_frame
from .problem import PricingProblem

SWEEP_LABELS = {"gamma": r"$\gamma={}$", "rho": r"$\rho={}$", "rho_k": r"$\rho_K={}$", "lambda": r"$\lambda={}$", "cycle_length": "cycle length={}"}
COLORS = ["purple", "magenta", "blue", "green", "red", "black", "darkorange", "tab:brown"]
MARKERS = ["X", "v", "o", "^", "D", "*", "s", "h"]
LINESTYLES = ["-", "-", "--", "-.", ":", "--", "-.", ":"]
COMPONENT_STYLES = [{"color": "black", "marker": "o", "linestyle": "--"}, {"color": "blue", "marker": "s", "linestyle": "-."},
                    {"color": "purple", "marker": "X", "linestyle": "-"}, {"color": "tab:brown", "marker": "h", "linestyle": "-"}]


def stable_seed(*parts):
    return hash_seed("|".join(str(part) for part in parts))


def allocate_budget(total, known_fraction, use_known, use_residual):
    if total <= 0:
        return 0, 0, 0
    if not use_residual:
        return (total, 0, total) if use_known else (0, 0, 0)
    if total < 2 + int(use_known):
        return 0, 0, 0
    known = max(1, min(int(round(total * known_fraction)), total - 2)) if use_known else 0
    arms = (total - known) // 2
    if arms <= 0:
        return 0, 0, 0
    used = known + 2 * arms
    if use_known and total > used:
        known, used = known + total - used, total
    return known, arms, used


def run_variant(problem, p, base_seed, week, run_index, suite, max_samples, metric_samples):
    rng = np.random.RandomState(stable_seed(base_seed, "train", week, run_index, suite))
    n = problem.n

    def evaluate(x, samples):
        return problem.evaluate(x, metric_samples, np.random.RandomState(stable_seed(base_seed, "metric", week, run_index, samples)))

    cycle = p["direction"] == "orthogonal_cycle"
    cycle_length = min(int(p["cycle_length"]), n)
    frame, position = (haar_frame(rng, n, cycle_length), 0) if cycle else (None, 0)
    x, momentum = problem.initial_x.copy(), np.zeros(n, dtype=float)
    gamma, rho, rho_k, mixing = float(p["gamma"]), float(p["rho"]), float(p["rho_k"]), float(p["lambda"])
    known_state = None
    use_known, use_residual = bool(p["use_known"]), bool(p["use_residual"])
    samples, objectives, failed = [0], [evaluate(x, 0)], False
    count, k = 0, 0
    while count < max_samples:
        batch = int(p["batch_initial"]) + int(p["batch_increment"]) * k
        known_count, arms, used = allocate_budget(min(3 * max(batch, 1), max_samples - count),
                                                  float(p["known_fraction"]), use_known, use_residual)
        if used <= 0:
            break
        known = np.zeros(n, dtype=float)
        if known_count > 0 and use_known:
            known = problem.partial_gradients(problem.sample_demands(x, known_count, rng)).mean(axis=0)
        residual = new_momentum = np.zeros(n, dtype=float)
        if use_residual:
            if cycle:
                if position >= cycle_length:
                    frame, position = haar_frame(rng, n, cycle_length), 0
                direction = np.sqrt(n) * np.asarray(frame, dtype=float)[:, position]
                position += 1
            else:
                direction = rng.normal(size=n)
            mu = float(p["radius"]) * float(p["radius_decay"]) ** k / np.sqrt(float(n))
            plus, _ = problem.sample_losses(x + mu * direction, arms, rng)
            minus, _ = problem.sample_losses(x - mu * direction, arms, rng)
            derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))
            residual = float(derivative - np.dot(gamma * known, direction)) * direction
            new_momentum = rho * momentum + (1.0 - rho) * residual
        readout = mixing * residual + (1.0 - mixing) * new_momentum if p["algorithm"] == "piso2" and use_residual else new_momentum
        known_state = known if known_state is None else rho_k * known_state + (1.0 - rho_k) * known
        gradient = gamma * known_state + readout
        new_x = x - float(p["step_size"]) * float(p["step_decay"]) ** k * gradient
        if not all(np.all(np.isfinite(a)) for a in (known, residual, new_momentum, gradient, new_x)) or np.linalg.norm(new_x) > 1e12:
            failed = True
            break
        x, momentum, count, k = new_x, new_momentum, count + used, k + 1
        objective = evaluate(new_x, count)
        if not np.isfinite(objective):
            failed = True
            break
        samples.append(count)
        objectives.append(objective)
    return {"samples": samples, "objectives": objectives, "failed": failed}


def build_variants(config, suite_name):
    suite, defaults = config["suites"][suite_name], config["defaults"]
    variants = []
    for panel, entries in suite["panels"].items():
        if "parameter" in suite:
            entries = {SWEEP_LABELS[suite["parameter"]].format(f"{value:g}" if isinstance(value, float) else value):
                       {**entries, suite["parameter"]: value} for value in suite["values"]}
        for index, (variant_label, overrides) in enumerate(entries.items()):
            params = {**defaults[overrides.get("template", "piso")], **overrides}
            variants.append({"panel": panel, "label": variant_label, "index": index, "params": params})
    return variants


def _task(args):
    config, suite_name, variant, week, run_index = args
    experiment = config["experiment"]
    spec = config["problem"]
    instance_rng = np.random.RandomState(stable_seed(int(experiment["seed"]), "instance", week, run_index))
    rho = spec["rho_low"] + (spec["rho_high"] - spec["rho_low"]) * instance_rng.rand(spec["products"])
    problem = PricingProblem(week, rho, spec)
    return run_variant(problem, variant["params"], int(experiment["seed"]), week, run_index, suite_name,
                       int(experiment["max_samples"]), int(experiment["metric_samples"]))


def run(config, output_dir, jobs=1):
    experiment = config["experiment"]
    weeks = [str(week).zfill(2) for week in experiment["weeks"]]
    grid = np.linspace(0.0, float(experiment["max_samples"]), int(experiment["report_grid_points"]))
    store = RunStore(output_dir / "runs")
    for suite_name in config["suites"]:
        variants = build_variants(config, suite_name)
        tasks = [(config, suite_name, v, w, r) for v in variants for w in weeks for r in range(int(experiment["simulations"]))]
        keys = [short_hash([config["experiment"], config["problem"], suite_name, v["params"], w, r]) for _, _, v, w, r in tasks]
        pending = [(t, key) for t, key in zip(tasks, keys) if store.load(key) is None]
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool, tqdm(total=len(pending), desc=f"ablation {suite_name}") as bar:
            for (task, key), result in zip(pending, pool.map(_task, [t for t, _ in pending])):
                store.save(key, result)
                bar.update()
        rows = []
        for (_, _, variant, week, run_index), key in zip(tasks, keys):
            result = store.load(key)
            samples = np.asarray(result["samples"], dtype=float)
            aligned = (np.full_like(grid, np.nan) if result["failed"]
                       else np.asarray(result["objectives"])[np.clip(np.searchsorted(samples, grid, side="right") - 1, 0, None)])
            rows += [{"panel": variant["panel"], "label": variant["label"], "index": variant["index"], "week": week,
                      "run": run_index, "samples": s, "objective": o} for s, o in zip(grid, aligned)]
        report(pd.DataFrame(rows), config["suites"][suite_name], output_dir / f"{suite_name}_ablation")


def report(aligned, suite, stem):
    stats = (aligned.groupby(["panel", "label", "index", "samples"], sort=False)["objective"]
             .agg(mean="mean", sd="std", n="count").reset_index())
    stats["se"] = stats["sd"].fillna(0.0) / np.sqrt(stats["n"])
    stats.to_csv(f"{stem}.csv", index=False)
    panels = list(suite["panels"])
    figure, axes = plt.subplots(1, len(panels), figsize=(12.8 if len(panels) == 2 else 20.0, 5.8), squeeze=False)
    handles = {}
    for axis, panel in zip(axes.ravel(), panels):
        frame = stats[stats["panel"] == panel]
        for (variant_label, index), values in frame.groupby(["label", "index"], sort=False):
            style = COMPONENT_STYLES[index] if "parameter" not in suite else {
                "color": COLORS[index], "marker": MARKERS[index], "linestyle": LINESTYLES[index]}
            x, mean, se = values["samples"].to_numpy(), values["mean"].to_numpy(), values["se"].to_numpy()
            (line,) = axis.plot(x, mean, label=variant_label, linewidth=2.2, markersize=5.5, markevery=max(1, len(x) // 10), **style)
            axis.fill_between(x, mean - se, mean + se, color=line.get_color(), alpha=0.18, linewidth=0.0)
            handles.setdefault(variant_label, line)
        window = frame if suite.get("full_y_range") else frame[frame["samples"] >= frame["samples"].max() * 0.7]
        values = np.concatenate([window["mean"] - window["se"], window["mean"] + window["se"]])
        values = values[np.isfinite(values)]
        if values.size:
            low, high = values.min(), values.max()
            margin = 0.1 * (high - low) if high > low else max(abs(low) * 0.05, 0.5)
            axis.set_ylim(low - margin, high + margin)
        axis.set_title(panel)
        axis.set_xlabel("sample number")
        axis.set_ylabel("objective")
        axis.grid(True, alpha=0.45)
        style_axis(axis)
    figure.legend(list(handles.values()), list(handles), loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=len(handles),
                  frameon=False, prop={"size": 13.0, "weight": "bold"}, handlelength=1.3, columnspacing=0.65, handletextpad=0.32)
    figure.tight_layout(rect=(0.0, 0.135, 1.0, 1.0), w_pad=1.8)
    save_figure(figure, stem)
