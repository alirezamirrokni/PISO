import numpy as np

from ..pricing.methods import Trace, batch_size, haar_frame, piso_batches


def normalized(vector, tolerance=0.0):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= tolerance else vector / norm


def exponential(initial, decay, k):
    return float(initial) * float(decay) ** int(k)


def gzo_ns(problem, p, rng, max_samples, metric_samples):
    x, n = problem.initial_x, problem.n
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, alpha = float(p["mu0"]), float(p["alpha0"])
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        guide_batch = int(p["guide_batch_initial"] + k * p["guide_batch_increment"])
        beta = exponential(p["beta0"], p["beta_decay"], k)
        guide = normalized(problem.known_gradient(x, problem.sample_observations(x, guide_batch, rng)))
        direction = np.sqrt(alpha / n) * rng.normal(size=n) + np.sqrt(1.0 - alpha) * float(rng.normal()) * guide
        plus, _ = problem.sample_losses(x + mu * direction, m, rng)
        minus, _ = problem.sample_losses(x - mu * direction, m, rng)
        gradient = (plus.mean() - minus.mean()) / (2.0 * mu) * direction
        samples += guide_batch + 2 * m
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
    mu, alpha = float(p["mu0"]), float(p["alpha0"])
    history, guide = [], np.zeros(n, dtype=float)
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta = exponential(p["beta0"], p["beta_decay"], k)
        direction = np.sqrt(alpha / n) * rng.normal(size=n) + np.sqrt(1.0 - alpha) * float(rng.normal()) * guide
        plus, plus_obs = problem.sample_losses(x + mu * direction, m, rng)
        minus, minus_obs = problem.sample_losses(x - mu * direction, m, rng)
        gradient = (plus.mean() - minus.mean()) / (2.0 * mu) * direction
        history.append({key: np.concatenate((plus_obs[key], minus_obs[key]), axis=0) for key in ("features", "labels")})
        samples += 2 * m
        x = x - beta * gradient
        recent = history[-min(int(p["window"]), len(history)):][::-1]
        guide = normalized(np.mean([problem.known_gradient(x, obs) for obs in recent], axis=0), tolerance=1.0e-5)
        mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
        alpha = 1.0 - float(p["alpha_damping"]) * (1.0 - alpha)
        k += 1
        trace.record(problem, x, samples, metric_samples, rng)
    return trace


def zo_tg(problem, p, rng, max_samples, metric_samples):
    x = problem.initial_x
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu = float(p["mu0"])
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta = exponential(p["beta0"], p["beta_decay"], k)
        direction = rng.normal(size=problem.n) / np.sqrt(problem.n)
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
    mu = float(p["mu"])
    samples, k = 0, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta = exponential(p["beta0"], p.get("beta_decay", 1.0), k)
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
    losses, observations = problem.sample_losses(x, initial, rng)
    trace = Trace()
    trace.record(problem, x, 0, metric_samples, rng)
    mu, control = float(p["mu0"]), float(losses.mean())
    beta0 = 1.0 / np.sqrt(problem.n) if p["beta0"] == "inverse_sqrt_dimension" else p["beta0"]
    points, batches, observation_batches = [x.copy()], [initial], [observations]
    samples, k = initial, 0
    while samples <= max_samples:
        m = batch_size(p, k)
        beta = exponential(beta0, p["beta_decay"], k)
        direction = rng.normal(size=problem.n)
        point = x + mu * direction
        losses, observations = problem.sample_losses(point, m, rng)
        gradient = (losses.mean() - control) / mu * direction
        points.append(point.copy())
        observation_batches.append(observations)
        samples += m
        x = x - beta * gradient
        mu = max(mu * float(p["mu_decay"]), float(p["mu_min"]))
        k += 1
        window = min(int(p["s_max"]), k)
        b = np.empty(window, dtype=float)
        for offset in range(window):
            source = k - 1 - offset
            b[window - 1 - offset] = float(p["M"]) * np.linalg.norm(x - points[source]) ** 2 + 1.0 / batches[source]
        weights = (1.0 / b) / np.sum(1.0 / b)
        control = 0.0
        for offset in range(window):
            source = k - 1 - offset
            control += weights[window - 1 - offset] * float(problem.loss_from_observations(x, observation_batches[source]).mean())
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
        cycle_length = min(int(p["cycle_length"]), n)
        frame, position = haar_frame(rng, n, cycle_length), 0
    samples, k = 0, 0
    while samples <= max_samples:
        known_count, plus_count, minus_count = piso_batches(p, batch_size(p, k))
        beta = exponential(p["step_size"], p["step_decay"], k)
        known = problem.known_gradient(x, problem.sample_observations(x, known_count, rng))
        mu = exponential(p["radius"], p["radius_decay"], k) / np.sqrt(n)
        if cycle:
            direction = np.sqrt(n) * np.asarray(frame[:, position], dtype=float)
            position += 1
            if position == cycle_length:
                frame, position = haar_frame(rng, n, cycle_length), 0
        else:
            direction = rng.normal(size=n)
        plus, _ = problem.sample_losses(x + mu * direction, plus_count, rng)
        minus, _ = problem.sample_losses(x - mu * direction, minus_count, rng)
        derivative = float((plus.mean() - minus.mean()) / (2.0 * mu))
        innovation = derivative - float(np.dot(gamma * known + momentum, direction))
        residual = momentum + innovation * direction
        new_momentum = rho * momentum + (1.0 - rho) * residual
        readout = new_momentum if mixing is None else mixing * residual + (1.0 - mixing) * new_momentum
        known_state = known if known_state is None else rho_k * known_state + (1.0 - rho_k) * known
        samples += known_count + plus_count + minus_count
        x = x - beta * (gamma * known_state + readout)
        momentum = new_momentum
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
STREAM_PADDING = {"mu": 0.1, "beta0": 0.05, "beta_decay": 1.0, "batch_initial": 30, "batch_increment": 2}
PRIVATE_SEED_ALIAS = {name: "ClassificationPISO" for name in ("GaussianPISO", "CyclePISO", "GaussianPISO2", "CyclePISO2")}
