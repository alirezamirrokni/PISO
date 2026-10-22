import numpy as np

from ..pricing.methods import haar_frame
from .oracle import oracle_model


class Recorder:
    def __init__(self, problem, interval):
        self.problem, self.interval = problem, int(interval)
        value, occupancy = problem.evaluate(problem.initial_x)
        self.samples, self.values, self.changes = [0], [value], [0.0]
        self.last, self.next = occupancy.copy(), self.interval

    def record(self, theta, count, force=False):
        if not force and count < self.next:
            return
        if self.samples[-1] != count:
            value, occupancy = self.problem.evaluate(theta)
            numerator = float(np.sum(np.abs(occupancy - self.last)))
            denominator = float(np.sum(np.abs(occupancy)) + np.sum(np.abs(self.last)))
            self.samples.append(count)
            self.values.append(value)
            self.changes.append(0.0 if denominator <= 0.0 else float(np.clip(numerator / denominator, 0.0, 1.0)))
            self.last = occupancy.copy()
        self.next = (count // self.interval + 1) * self.interval

    def finish(self, theta, count):
        self.record(theta, count, force=True)
        return {"samples": self.samples, "values": self.values, "changes": self.changes}


def schedule(p, name, samples):
    value = float(p[name])
    if p[f"{name}_schedule"] == "inverse_sqrt":
        value = value / np.sqrt(1.0 + float(samples) / float(p.get(f"{name}_scale", 100000.0)))
    return float(max(value, float(p.get(f"{name}_min", 0.0))))


def clip(gradient, limit):
    norm = float(np.linalg.norm(gradient))
    return gradient if limit is None or norm <= limit or norm == 0.0 else gradient * (float(limit) / norm)


def batch_size(p, k):
    return int(p["batch_initial"]) + int(k) * int(p.get("batch_increment", 0))


def guided_direction(rng, n, alpha, hint):
    norm = float(np.linalg.norm(hint))
    unit = hint / norm if norm > 0.0 else np.zeros(n, dtype=float)
    orthogonal = alpha / float(n)
    parallel = 1.0 - alpha + orthogonal if np.linalg.norm(unit) > 0.0 else orthogonal
    direction = np.sqrt(orthogonal) * rng.normal(size=n) + np.sqrt(1.0 - alpha) * float(rng.normal()) * unit
    if norm > 0.0:
        along = float(np.dot(unit, direction)) * unit
        return direction, along / parallel + (direction - along) / orthogonal
    return direction, direction / orthogonal


def vanilla_pg(problem, p, rng, budget, recorder):
    theta, samples, k = problem.initial_x.copy(), 0, 0
    while samples + batch_size(p, k) <= budget:
        count = batch_size(p, k)
        gradient = clip(problem.known_gradient(theta, problem.sample_trajectories(theta, count, rng)), p.get("gradient_clip_norm"))
        theta = theta - schedule(p, "step_size", samples) * gradient
        samples, k = samples + count, k + 1
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


def zo_tg(problem, p, rng, budget, recorder):
    theta, samples, k = problem.initial_x.copy(), 0, 0
    while samples + 2 * batch_size(p, k) <= budget:
        count = batch_size(p, k)
        mu = schedule(p, "radius", samples) / np.sqrt(problem.n)
        eta = schedule(p, "step_size", samples)
        direction = rng.normal(size=problem.n)
        plus, _, minus, _ = problem.sample_paired_losses(theta + mu * direction, theta - mu * direction, count, rng)
        gradient = clip(float((plus.mean() - minus.mean()) / (2.0 * mu)) * direction, p.get("gradient_clip_norm"))
        theta = theta - eta * gradient
        samples, k = samples + 2 * count, k + 1
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


def zo_ogvr(problem, p, rng, budget, recorder):
    theta, samples, k = problem.initial_x.copy(), 0, 0
    window, weight = int(p["history_window"]), float(p["distance_weight"])
    initial = int(p["initial_baseline_batch"])
    if initial > budget:
        return recorder.finish(theta, samples)
    baseline, samples, history = float(problem.sample_losses(theta, initial, rng)[0].mean()), initial, []
    recorder.record(theta, samples)
    while samples + batch_size(p, k) <= budget:
        count = batch_size(p, k)
        mu = schedule(p, "radius", samples) / np.sqrt(problem.n)
        eta = schedule(p, "step_size", samples)
        direction = rng.normal(size=problem.n)
        query = np.asarray(theta + mu * direction, dtype=float)
        losses, _ = problem.sample_losses(query, count, rng)
        gradient = clip(((float(losses.mean()) - baseline) / mu) * direction, p.get("gradient_clip_norm"))
        theta = np.asarray(theta - eta * gradient, dtype=float)
        history = (history + [(query.copy(), np.asarray(losses, dtype=float).copy())])[-window:]
        inverse = [1.0 / (weight * float(np.sum((theta - q) ** 2)) + 1.0 / float(l.size)) for q, l in history]
        baseline = float(np.dot(np.asarray(inverse) / float(np.sum(inverse)), np.asarray([float(l.mean()) for _, l in history])))
        samples, k = samples + count, k + 1
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


def gzo_ns(problem, p, rng, budget, recorder):
    theta, samples, k = problem.initial_x.copy(), 0, 0
    total, known_count = int(p["total_batch"]), int(p["known_batch"])
    query_count = (total - known_count) // 2
    while samples + total <= budget:
        guide = problem.known_gradient(theta, problem.sample_trajectories(theta, known_count, rng))
        direction, score = guided_direction(rng, problem.n, float(p["alpha"]), np.asarray(guide, dtype=float))
        mu, eta = schedule(p, "radius", samples), schedule(p, "step_size", samples)
        plus, _, minus, _ = problem.sample_paired_losses(theta + mu * direction, theta - mu * direction, query_count, rng)
        gradient = clip(float((plus.mean() - minus.mean()) / (2.0 * mu)) * score, p.get("gradient_clip_norm"))
        theta = theta - eta * gradient
        samples, k = samples + total, k + 1
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


def gzo_hs(problem, p, rng, budget, recorder):
    theta, samples, k = problem.initial_x.copy(), 0, 0
    guide, history = np.zeros(problem.n, dtype=float), []
    while samples + 2 * batch_size(p, k) <= budget:
        count = batch_size(p, k)
        direction, score = guided_direction(rng, problem.n, float(p["alpha"]), guide)
        mu, eta = schedule(p, "radius", samples), schedule(p, "step_size", samples)
        plus_theta, minus_theta = theta + mu * direction, theta - mu * direction
        plus, plus_paths, minus, minus_paths = problem.sample_paired_losses(plus_theta, minus_theta, count, rng)
        gradient = clip(float((plus.mean() - minus.mean()) / (2.0 * mu)) * score, p.get("gradient_clip_norm"))
        history = (history + [(plus_theta.copy(), plus_paths, minus_theta.copy(), minus_paths)])[-int(p["window"]):]
        theta = theta - eta * gradient
        samples, k = samples + 2 * count, k + 1
        gradients = []
        for p_theta, p_paths, m_theta, m_paths in history:
            gradients += [problem.known_gradient(p_theta, p_paths), problem.known_gradient(m_theta, m_paths)]
        guide = np.mean(gradients, axis=0)
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


def piso(problem, p, rng, budget, recorder, cycle=False, two_level=False):
    n = problem.n
    theta, samples, k = problem.initial_x.copy(), 0, 0
    gamma, rho, rho_k = float(p["gamma"]), float(p["rho"]), float(p["rho_k"])
    mixing = float(p["lambda"]) if two_level else 0.0
    total = int(p["total_batch"])
    known_count = int(round(total * float(p["known_fraction"])))
    if (total - known_count) % 2:
        known_count += 1
    pair_count = (total - known_count) // 2
    if cycle:
        length = min(int(p["cycle_length"]), n)
        frame, position = haar_frame(rng, n, length), 0
    residual, known_state = np.zeros(n, dtype=float), None
    while samples + total + (known_count if known_state is None else 0) <= budget:
        if known_state is None:
            known_state = problem.known_gradient(theta, problem.sample_trajectories(theta, known_count, rng))
            samples += known_count
        known = problem.known_gradient(theta, problem.sample_trajectories(theta, known_count, rng))
        direction = np.sqrt(n) * np.asarray(frame, dtype=float)[:, position] if cycle else rng.normal(size=n)
        mu = schedule(p, "radius", samples) / np.sqrt(n)
        plus, _, minus, _ = problem.sample_paired_losses(theta + mu * direction, theta - mu * direction, pair_count, rng)
        innovation = float((plus.mean() - minus.mean()) / (2.0 * mu)) - float(np.dot(gamma * known + residual, direction))
        observation = residual + innovation * np.asarray(direction)
        residual_next = rho * residual + (1.0 - rho) * observation
        known_state = rho_k * np.asarray(known_state, dtype=float) + (1.0 - rho_k) * known
        gradient = gamma * known_state + (mixing * observation + (1.0 - mixing) * residual_next)
        eta = schedule(p, "step_size", samples)
        residual, samples, theta, k = residual_next, samples + total, theta - eta * gradient, k + 1
        if cycle:
            position += 1
            if position == length:
                frame, position = haar_frame(rng, n, length), 0
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


def pepg(problem, p, rng, budget, recorder):
    theta, samples, k = problem.initial_x.copy(), 0, 0
    floor = float(p["transition_probability_floor"])
    while samples + batch_size(p, k) <= budget:
        count = batch_size(p, k)
        paths = problem.sample_trajectories(theta, count, rng)
        model = oracle_model(problem, theta)
        policy_part, transition_part, reward_part = (np.zeros(problem.n, dtype=np.float64) for _ in range(3))
        for path in paths:
            for t, (state, action, next_state, _) in enumerate(path):
                discount = float(problem.discount) ** t
                policy_part += discount * float(model["advantage"][state, action]) * model["policy_score"][state, action]
                probability = max(float(model["transitions"][state, action, next_state]), floor)
                transition_part += (discount * float(problem.discount) * float(model["value"][next_state])
                                    * (model["transition_jacobian"][state, action, next_state] / probability))
                reward_part += discount * model["reward_jacobian"][state, action]
        scale = 1.0 / float(len(paths))
        gradient = clip(policy_part * scale + transition_part * scale + reward_part * scale, p.get("gradient_clip_norm"))
        theta = theta + schedule(p, "step_size", samples) * gradient
        samples, k = samples + count, k + 1
        recorder.record(theta, samples)
    return recorder.finish(theta, samples)


METHODS = {
    "VanillaPG": (vanilla_pg, "VanillaPG"),
    "ZO_TG": (zo_tg, "ZO_TG"),
    "ZO_OGVR": (zo_ogvr, "ZO_OGVR"),
    "GZO_NS": (gzo_ns, "GZO_NS"),
    "GZO_HS": (gzo_hs, "GZO_HS"),
    "GaussianPISO": (piso, "GaussianPISOAlgorithm1Family"),
    "CyclePISO": (lambda *a: piso(*a, cycle=True), "CyclePISOAlgorithm1Family"),
    "GaussianPISO2": (lambda *a: piso(*a, two_level=True), "GaussianPISOAlgorithm1Family"),
    "CyclePISO2": (lambda *a: piso(*a, cycle=True, two_level=True), "CyclePISOAlgorithm1Family"),
    "PePG": (pepg, "PePG"),
}
