import numpy as np

from ..common import hash_seed


def stable_seed(*parts):
    return hash_seed(":".join(str(part) for part in parts), nbytes=8) % (2 ** 32)


def normalized(vector):
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= 1.0e-15 else vector / norm


def clip_norm(vector, maximum):
    norm = float(np.linalg.norm(vector))
    return vector if maximum <= 0.0 or norm <= maximum or norm <= 1.0e-15 else vector * (maximum / norm)


def step_size(p, problem, iteration):
    if problem.family == "routing":
        base = float(p["step_size"]) / float(problem.dimension)
    else:
        base = float(p["step_size_low_dimension"] if problem.dimension <= int(p["dimension_threshold"])
                     else p["step_size_high_dimension"])
    return base / np.sqrt(iteration + 1.0) if p["step_schedule"] == "inverse_sqrt" else base


def sphere_direction(seed, dimension):
    direction = np.random.default_rng(seed).normal(size=dimension)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-15 else direction


def haar_frame(seed, dimension, columns):
    q_matrix, r_matrix = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, columns)), mode="reduced")
    return q_matrix * np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)


class Method:
    def __init__(self, params, problem, seed, instance_id):
        self.p, self.problem, self.seed, self.instance_id = params, problem, seed, instance_id

    def tag_seed(self, iteration, tag):
        return stable_seed(self.seed, self.problem.family, self.instance_id, iteration, tag)

    def two_point(self, x, direction, mu, warm):
        problem = self.problem
        plus_y, plus_warm = problem.response(x + mu * direction, warm)
        minus_y, minus_warm = problem.response(x - mu * direction, warm)
        plus = problem.objective(x + mu * direction, plus_y)
        minus = problem.objective(x - mu * direction, minus_y)
        return (plus, plus_y, plus_warm), (minus, minus_y, minus_warm)


class ZOS(Method):
    def gradient(self, x, y, warm, iteration):
        d, mu = self.problem.dimension, float(self.p["mu"])
        direction = sphere_direction(self.tag_seed(iteration, "direction"), d)
        plus, minus = self.two_point(x, direction, mu, warm)
        return d * (plus[0] - minus[0]) * direction / (2.0 * mu), plus, minus, 2


class PZOS(Method):
    def gradient(self, x, y, warm, iteration):
        d, mu = self.problem.dimension, float(self.p["mu"])
        direction = sphere_direction(self.tag_seed(iteration, "direction"), d)
        plus, minus = self.two_point(x, direction, mu, warm)
        chain = float(np.dot(plus[1] - minus[1], self.problem.partial_y(x, y)))
        return self.problem.partial_x(x, y) + d * chain * direction / (2.0 * mu), plus, minus, 3


class GZO(Method):
    historical = False

    def __init__(self, *args):
        super().__init__(*args)
        self.alpha = float(self.p["alpha_initial"])
        self.guide = np.zeros(self.problem.dimension)
        self.responses = []

    def gradient(self, x, y, warm, iteration):
        d, mu = self.problem.dimension, float(self.p["radius"])
        guide = self.guide if self.historical else np.asarray(self.problem.partial_x(x, y), dtype=float)
        rng = np.random.default_rng(self.tag_seed(iteration, "gzo-direction"))
        unit = normalized(guide)
        if float(np.linalg.norm(unit)) <= 1.0e-15:
            direction = rng.normal(size=d) / np.sqrt(float(d))
        else:
            direction = (np.sqrt(self.alpha / float(d)) * rng.normal(size=d)
                         + np.sqrt(max(0.0, 1.0 - self.alpha)) * float(rng.normal()) * unit)
        plus, minus = self.two_point(x, direction, mu, warm)
        self.alpha = float(np.clip(1.0 - float(self.p["alpha_damping"]) * (1.0 - self.alpha), 0.0, 1.0))
        if self.historical:
            self.responses = (self.responses + [plus[1].copy(), minus[1].copy()])[-2 * int(self.p["history_window"]):]
        return float((plus[0] - minus[0]) / (2.0 * mu)) * direction, plus, minus, 2 if self.historical else 3

    def after_step(self, x):
        if self.historical:
            self.guide = normalized(np.mean([self.problem.partial_x(x, r) for r in self.responses], axis=0))


class GZOHS(GZO):
    historical = True


class PISO(Method):
    cycle = False
    two_level = False

    def __init__(self, *args):
        super().__init__(*args)
        self.momentum, self.known_state = np.zeros(self.problem.dimension, dtype=float), None

    def gradient(self, x, y, warm, iteration):
        d, p = self.problem.dimension, self.p
        known = np.asarray(self.problem.partial_x(x, y), dtype=float)
        if self.cycle:
            length = min(int(p["cycle_length"]), d)
            frame = haar_frame(self.tag_seed(iteration // length, "piso-cycle"), d, length)
            direction = np.sqrt(float(d)) * frame[:, iteration % length]
        else:
            direction = np.random.default_rng(self.tag_seed(iteration, "piso-gaussian")).normal(size=d)
        mu = float(p["radius"]) / np.sqrt(float(d))
        plus, minus = self.two_point(x, direction, mu, warm)
        gamma, rho = float(p["gamma"]), float(p["rho"])
        innovation = float((plus[0] - minus[0]) / (2.0 * mu)) - float(np.dot(gamma * known + self.momentum, direction))
        residual = self.momentum + innovation * direction
        momentum = rho * self.momentum + (1.0 - rho) * residual
        if float(p["unknown_clip_ratio"]) > 0.0:
            limit = float(p["unknown_clip_ratio"]) * max(float(np.linalg.norm(known)), float(p["known_norm_floor"]))
            momentum, residual = clip_norm(momentum, limit), clip_norm(residual, limit)
        mixing = float(p["lambda"]) if self.two_level else 0.0
        readout = mixing * residual + (1.0 - mixing) * momentum if self.two_level else momentum
        self.momentum = momentum
        rho_k = float(p["rho_k"])
        self.known_state = known if self.known_state is None else rho_k * self.known_state + (1.0 - rho_k) * known
        return gamma * self.known_state + readout, plus, minus, 3


class CyclePISO(PISO):
    cycle = True


class GaussianPISO2(PISO):
    two_level = True


class CyclePISO2(PISO):
    cycle, two_level = True, True


METHODS = {"ZOS": ZOS, "PZOS": PZOS, "GZO_NS": GZO, "GZO_HS": GZOHS, "GaussianPISO": PISO,
           "CyclePISO": CyclePISO, "GaussianPISO2": GaussianPISO2, "CyclePISO2": CyclePISO2}


def run_method(name, params, problem, seed, instance_id, iterations):
    method = METHODS[name](params, problem, seed, instance_id)
    x = problem.initial_decision()
    y, warm = problem.response(x, None)
    objectives, calls, total = [problem.objective(x, y)], [0], 0
    sign = 1.0 if problem.sense == "max" else -1.0
    for iteration in range(iterations):
        gradient, plus, minus, used = method.gradient(x, y, warm, iteration)
        total += used
        x = problem.project(x + sign * step_size(params, problem, iteration) * gradient)
        if hasattr(method, "after_step"):
            method.after_step(x)
        closer = np.linalg.norm(plus[1] - y) <= np.linalg.norm(minus[1] - y)
        y, warm = problem.response(x, plus[2] if closer else minus[2])
        objectives.append(problem.objective(x, y))
        calls.append(total)
    return np.asarray(objectives, dtype=float), np.asarray(calls, dtype=int)
