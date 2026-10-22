from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    samples: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    final_x: np.ndarray = None

    def record(self, problem, x, samples, metric_samples, rng):
        self.samples.append(int(samples))
        self.objectives.append(problem.evaluate(x, metric_samples, rng))
        self.final_x = np.asarray(x, dtype=float).copy()


def batch_size(p, k):
    return int(p["batch_initial"] + k * p["batch_increment"])


def haar_frame(rng, dimension, columns):
    q_matrix, r_matrix = np.linalg.qr(rng.normal(size=(dimension, columns)), mode="reduced")
    return q_matrix * np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)


def piso_batches(p, batch):
    total = 3 * int(batch)
    known = int(round(total * float(p["known_fraction"])))
    plus = (total - known) // 2
    return known, plus, total - known - plus


def gzo_ns(problem, p, rng, max_samples, metric_samples):
    x, n = problem.initial_x, problem.n
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, beta, alpha = float(p["mu0"]), float(p["beta0"]), float(p["alpha0"])
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta *= float(p["beta_decay"])
        guide = problem.partial_gradients(problem.sample_demands(x, m, rng)).mean(axis=0)
        norm = np.linalg.norm(guide)
        if norm > 0:
            guide /= norm
        scalar = rng.normal()
        isotropic = rng.normal(size=n)
        direction = np.sqrt(alpha / n) * isotropic + np.sqrt(1.0 - alpha) * scalar * guide
        plus, _ = problem.sample_losses(x + mu * direction, m, rng)
        minus, _ = problem.sample_losses(x - mu * direction, m, rng)
        gradient = (plus.mean() - minus.mean()) / (2.0 * mu) * direction
        samples += 3 * m
        x = x - beta * gradient
        mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
        alpha = 1.0 - float(p["alpha_damping"]) * (1.0 - alpha)
        k += 1
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


def gzo_hs(problem, p, rng, max_samples, metric_samples):
    x, n = problem.initial_x, problem.n
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, beta, alpha = float(p["mu0"]), float(p["beta0"]), float(p["alpha0"])
    window = int(p["window"])
    history, guide = [], 0.0
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta *= float(p["beta_decay"])
        scalar = rng.normal()
        isotropic = rng.normal(size=n)
        direction = np.sqrt(alpha / n) * isotropic + np.sqrt(1.0 - alpha) * scalar * guide
        plus, plus_demands = problem.sample_losses(x + mu * direction, m, rng)
        minus, minus_demands = problem.sample_losses(x - mu * direction, m, rng)
        gradient = (plus.mean() - minus.mean()) / (2.0 * mu) * direction
        history.append(np.concatenate([plus_demands, minus_demands], axis=0))
        samples += 2 * m
        x = x - beta * gradient
        guide = 0.0
        for offset in range(min(window, k)):
            guide += problem.partial_gradients(history[k - 1 - offset]).mean(axis=0)
        guide = guide / window
        if np.linalg.norm(guide) > 1e-5:
            guide = guide / np.linalg.norm(guide)
        mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
        alpha = 1.0 - float(p["alpha_damping"]) * (1.0 - alpha)
        k += 1
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


def zo_tg(problem, p, rng, max_samples, metric_samples):
    x = problem.initial_x
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, beta = float(p["mu0"]), float(p["beta0"])
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta *= float(p["beta_decay"])
        direction = rng.normal(size=problem.n)
        plus, _ = problem.sample_losses(x + mu * direction, m, rng)
        minus, _ = problem.sample_losses(x - mu * direction, m, rng)
        gradient = (plus.mean() - minus.mean()) / (2.0 * mu) * direction
        samples += 2 * m
        x = x - beta * gradient
        mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
        k += 1
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


def zo_og(problem, p, rng, max_samples, metric_samples):
    x = problem.initial_x
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, beta = float(p["mu"]), float(p["beta0"])
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta *= float(p["beta_decay"])
        direction = rng.normal(size=problem.n)
        losses, _ = problem.sample_losses(x + mu * direction, m, rng)
        gradient = losses.mean() / mu * direction
        samples += m
        x = x - beta * gradient
        k += 1
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


def zo_ogvr(problem, p, rng, max_samples, metric_samples):
    x = problem.initial_x
    initial = int(p["initial_samples"])
    losses, demands = problem.sample_losses(x, initial, rng)
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, beta, control = float(p["mu0"]), float(p["beta0"]), float(losses.mean())
    points, batches, demand_batches = [x.copy()], [initial], [demands]
    samples, k = initial, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta *= float(p["beta_decay"])
        direction = rng.normal(size=problem.n)
        point = x + mu * direction
        points.append(point.copy())
        losses, demands = problem.sample_losses(point, m, rng)
        demand_batches.append(demands)
        gradient = (losses.mean() - control) / mu * direction
        samples += m
        x = x - beta * gradient
        mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
        k += 1
        s = min(int(p["s_max"]), k)
        b = np.zeros(s)
        for i in range(s):
            source = k - 1 - i
            b[s - 1 - i] = float(p["M"]) * np.linalg.norm(x - points[source]) ** 2 + 1.0 / batches[source]
        weights = (1.0 / b) / (1.0 / b).sum()
        control = 0.0
        for i in range(s):
            source = k - 1 - i
            control += weights[s - 1 - i] * float(problem.loss_from_demands(x, demand_batches[source]).mean())
        batches.append(m)
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


def piso(problem, p, rng, max_samples, metric_samples, cycle=False, mixing=None):
    x, n = problem.initial_x, problem.n
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    gamma, rho, rho_k = float(p["gamma"]), float(p["rho"]), float(p["rho_k"])
    momentum, known_state = np.zeros(n, dtype=float), None
    if cycle:
        cycle_length = int(p["cycle_length"])
        frame, position = haar_frame(rng, n, cycle_length), 0
    samples, k = 0, 0
    while samples <= max_samples:
        known_count, plus_count, minus_count = piso_batches(p, batch_size(p, k))
        mu = float(p["radius"]) * float(p["radius_decay"]) ** k / np.sqrt(n)
        beta = float(p["step_size"]) * float(p["step_decay"]) ** k
        direction = np.sqrt(n) * np.asarray(frame[:, position], dtype=float) if cycle else rng.normal(size=n)
        plus, _ = problem.sample_losses(x + mu * direction, plus_count, rng)
        minus, _ = problem.sample_losses(x - mu * direction, minus_count, rng)
        known = problem.partial_gradients(problem.sample_demands(x, known_count, rng)).mean(axis=0)
        derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))
        innovation = derivative - float(np.dot(gamma * known + momentum, direction))
        residual = momentum + innovation * direction
        new_momentum = rho * momentum + (1.0 - rho) * residual
        readout = new_momentum if mixing is None else mixing * residual + (1.0 - mixing) * new_momentum
        known_state = known if known_state is None else rho_k * known_state + (1.0 - rho_k) * known
        samples += known_count + plus_count + minus_count
        x = x - beta * (gamma * known_state + readout)
        momentum = new_momentum
        if cycle:
            position += 1
            if position == cycle_length:
                frame, position = haar_frame(rng, n, cycle_length), 0
        k += 1
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


METHODS = {
    "GZO_NS": gzo_ns,
    "GZO_HS": gzo_hs,
    "ZO_TG": zo_tg,
    "ZO_OGVR": zo_ogvr,
    "GaussianPISO": lambda *a: piso(*a),
    "CyclePISO": lambda *a: piso(*a, cycle=True),
    "GaussianPISO2": lambda *a: piso(*a, mixing=float(a[1]["lambda"])),
    "CyclePISO2": lambda *a: piso(*a, cycle=True, mixing=float(a[1]["lambda"])),
}
STREAM_PADDING = {"mu": 0.001, "beta0": 1.0e-05, "beta_decay": 0.95, "batch_initial": 30, "batch_increment": 2}
PRIVATE_SEED_ALIAS = {name: "GeneralPISO" for name in ("GaussianPISO", "CyclePISO", "GaussianPISO2", "CyclePISO2")}
