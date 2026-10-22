import heapq

import numpy as np


class Routing:
    family, sense = "routing", "max"

    def __init__(self, tails, heads, latency_a, latency_b, origins, destinations, demands, sensitivities, spec):
        self.tails, self.heads = np.asarray(tails, dtype=np.int64), np.asarray(heads, dtype=np.int64)
        self.latency_a, self.latency_b = np.asarray(latency_a, dtype=float), np.asarray(latency_b, dtype=float)
        self.origins, self.destinations = np.asarray(origins, dtype=np.int64), np.asarray(destinations, dtype=np.int64)
        self.demands, self.sensitivities = np.asarray(demands, dtype=float), np.asarray(sensitivities, dtype=float)
        self.regularization = float(spec["regularization"])
        self.tolerance, self.max_iterations = float(spec["wardrop_tolerance"]), int(spec["wardrop_max_iterations"])
        self.n_nodes = int(max(self.tails.max(), self.heads.max()) + 1)
        self.dimension, self.n_groups = int(self.tails.size), int(self.demands.size)
        self.adjacency = [[] for _ in range(self.n_nodes)]
        indegree = np.zeros(self.n_nodes, dtype=np.int64)
        for edge, (tail, head) in enumerate(zip(self.tails, self.heads)):
            self.adjacency[int(tail)].append((int(head), edge))
            indegree[int(head)] += 1
        for outgoing in self.adjacency:
            outgoing.sort()
        queue = [node for node in range(self.n_nodes) if indegree[node] == 0]
        heapq.heapify(queue)
        self.order = []
        while queue:
            u = heapq.heappop(queue)
            self.order.append(u)
            for v, _ in self.adjacency[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    heapq.heappush(queue, v)

    def initial_decision(self):
        return np.zeros(self.dimension, dtype=float)

    def project(self, x):
        return np.asarray(x, dtype=float)

    def objective(self, x, y):
        return float(np.dot(x, y) - self.regularization * np.dot(x, x))

    def partial_x(self, x, y):
        return np.asarray(y, dtype=float) - 2.0 * self.regularization * np.asarray(x, dtype=float)

    def partial_y(self, x, y):
        return np.asarray(x, dtype=float).copy()

    def shortest_path(self, weights, source, target):
        distance = np.full(self.n_nodes, np.inf)
        previous = np.full(self.n_nodes, -1, dtype=np.int64)
        distance[source] = 0.0
        for u in self.order:
            if np.isfinite(distance[u]):
                for v, edge in self.adjacency[u]:
                    candidate = distance[u] + float(weights[edge])
                    if candidate < distance[v] - 1.0e-14:
                        distance[v], previous[v] = candidate, edge
        path, node = [], target
        while node != source:
            path.append(int(previous[node]))
            node = int(self.tails[previous[node]])
        return path[::-1]

    def all_or_nothing(self, toll, flow):
        latency = self.latency_a * flow + self.latency_b
        assignment = np.zeros((self.n_groups, self.dimension), dtype=float)
        for group in range(self.n_groups):
            path = self.shortest_path(latency + toll / self.sensitivities[group],
                                      int(self.origins[group]), int(self.destinations[group]))
            assignment[group, path] = self.demands[group]
        return assignment

    def response(self, x, warm=None):
        toll = np.asarray(x, dtype=float)
        if isinstance(warm, dict):
            group_flow = np.asarray(warm["group_flow"], dtype=float).copy()
        else:
            group_flow = self.all_or_nothing(toll, np.zeros(self.dimension, dtype=float))
        for _ in range(self.max_iterations):
            aggregate = group_flow.sum(axis=0)
            direction = self.all_or_nothing(toll, aggregate) - group_flow
            aggregate_direction = direction.sum(axis=0)
            edge_gradient = self.latency_a * aggregate + self.latency_b
            directional = float(np.dot(edge_gradient, aggregate_direction))
            directional += float(np.sum((toll[None, :] / self.sensitivities[:, None]) * direction))
            if -directional <= self.tolerance * (1.0 + float(np.dot(edge_gradient, aggregate))):
                break
            curvature = float(np.dot(self.latency_a, aggregate_direction ** 2))
            step = (1.0 if directional < 0.0 else 0.0) if curvature <= 1.0e-18 else float(np.clip(-directional / curvature, 0.0, 1.0))
            group_flow += step * direction
        return group_flow.sum(axis=0), {"group_flow": group_flow}


def generate_routing(config, seed):
    spec, rng = config["problem"], np.random.default_rng(seed)
    instances = []
    for _ in range(int(config["experiment"]["instances"])):
        n_nodes = int(rng.integers(int(spec["vertices_min"]), int(spec["vertices_max"]) + 1))
        max_edges = min(int(spec["edges_max"]), n_nodes * (n_nodes - 1) // 2)
        n_edges = int(rng.integers(min(max(int(spec["edges_min"]), n_nodes - 1), max_edges), max_edges + 1))
        permutation = np.asarray(rng.permutation(n_nodes), dtype=np.int64)
        edges = {(int(permutation[i]), int(permutation[i + 1])) for i in range(n_nodes - 1)}
        forward = [(int(permutation[i]), int(permutation[j])) for i in range(n_nodes - 1)
                   for j in range(i + 1, n_nodes) if j != i + 1]
        rng.shuffle(forward)
        for edge in forward:
            if len(edges) >= n_edges:
                break
            edges.add(edge)
        edges = np.asarray(sorted(edges), dtype=np.int64)
        n_groups = int(rng.integers(int(spec["groups_min"]), int(spec["groups_max"]) + 1))
        origin_positions = rng.integers(0, n_nodes - 1, size=n_groups)
        destination_positions = np.asarray([rng.integers(int(p) + 1, n_nodes) for p in origin_positions], dtype=np.int64)
        demands = rng.choice(np.asarray(spec["demand_values"], dtype=float), size=n_groups, replace=True)
        latency_a = rng.uniform(float(spec["latency_a_min"]), float(spec["latency_a_max"]), size=n_edges)
        latency_b = rng.uniform(float(spec["latency_b_min"]), float(spec["latency_b_max"]), size=n_edges)
        sensitivities = rng.uniform(float(spec["sensitivity_min"]), float(spec["sensitivity_max"]), size=n_groups)
        instances.append(Routing(edges[:, 0], edges[:, 1], latency_a, latency_b, permutation[origin_positions],
                                 permutation[destination_positions], demands, sensitivities, spec))
    return instances


def bisect_positive_root(function, size):
    low, high = np.zeros(size, dtype=float), np.ones(size, dtype=float)
    values = function(high)
    for _ in range(80):
        mask = values > 0.0
        if not np.any(mask):
            break
        high[mask] *= 2.0
        values = function(high)
    for _ in range(80):
        middle = 0.5 * (low + high)
        positive = function(middle) > 0.0
        low[positive], high[~positive] = middle[positive], middle[~positive]
    return 0.5 * (low + high)


class Security:
    family, sense = "security", "min"

    def __init__(self, defender_value, attacker_value, baseline, defender_cost, attacker_cost, budget, initial_x, spec):
        self.defender_value, self.attacker_value, self.baseline = defender_value, attacker_value, baseline
        self.defender_cost, self.attacker_cost, self.budget = defender_cost, attacker_cost, budget
        self.initial_x, self.dimension = initial_x, int(defender_value.size)
        self.tolerance, self.max_iterations = float(spec["response_tolerance"]), int(spec["response_max_iterations"])

    def initial_decision(self):
        return self.initial_x.copy()

    def project(self, x):
        return np.maximum(np.asarray(x, dtype=float), 0.0)

    def objective(self, x, y):
        denominator = np.maximum(x + y + self.baseline, 1e-12)
        return float(np.sum(self.defender_value * y / denominator + 0.5 * self.defender_cost * x ** 2))

    def partial_x(self, x, y):
        denominator = np.maximum(x + y + self.baseline, 1e-12)
        return -self.defender_value * y / denominator ** 2 + self.defender_cost * x

    def partial_y(self, x, y):
        denominator = np.maximum(x + y + self.baseline, 1e-12)
        return self.defender_value * (x + self.baseline) / denominator ** 2

    def effort_at_dual(self, c0, dual, initial):
        h0 = self.attacker_value / np.maximum(c0, 1e-12)
        active = h0 > dual
        if initial is None:
            effort = np.maximum((h0 - dual) / (self.attacker_cost + 2.0 * self.attacker_value / np.maximum(c0, 1e-12) ** 2), 0.0)
        else:
            effort = np.maximum(np.asarray(initial, dtype=float), 0.0).copy()
        effort = np.minimum(effort, self.budget)
        effort[~active] = 0.0
        for _ in range(10):
            denominator = np.maximum(c0 + effort, 1e-12)
            equation = self.attacker_value * c0 / denominator ** 2 - self.attacker_cost * effort - dual
            derivative = -2.0 * self.attacker_value * c0 / denominator ** 3 - self.attacker_cost
            update = np.divide(equation, derivative, out=np.zeros_like(equation), where=np.abs(derivative) > 1e-14)
            effort = np.clip(effort - update, 0.0, self.budget)
            effort[~active] = 0.0
        denominator = np.maximum(c0 + effort, 1e-12)
        return effort, -2.0 * self.attacker_value * c0 / denominator ** 3 - self.attacker_cost

    def response(self, x, warm=None):
        c0 = np.maximum(np.asarray(x, dtype=float) + self.baseline, 1e-10)
        low, high = 0.0, float(np.max(self.attacker_value / c0))
        dual = float(np.clip(float(warm.get("dual", 0.5 * high)) if isinstance(warm, dict) else 0.5 * high, low, high))
        initial = warm.get("effort") if isinstance(warm, dict) else None
        effort = None
        for _ in range(self.max_iterations):
            effort, derivative = self.effort_at_dual(c0, dual, initial)
            residual = float(np.sum(effort)) - self.budget
            if abs(residual) <= self.tolerance * max(1.0, self.budget):
                break
            if residual > 0.0:
                low = dual
            else:
                high = dual
            active = effort > 1e-12
            slope = float(np.sum(1.0 / derivative[active])) if np.any(active) else -np.inf
            proposal = dual - residual / slope if np.isfinite(slope) and slope < -1e-14 else np.nan
            dual = float(proposal) if np.isfinite(proposal) and low < proposal < high else float(0.5 * (low + high))
            initial = effort
        if effort is None:
            effort, _ = self.effort_at_dual(c0, dual, initial)
        return effort, {"dual": dual, "effort": effort}


def generate_security(config, seed):
    spec, count, rng = config["problem"], int(config["experiment"]["instances"]), np.random.default_rng(seed)
    instances = []
    for n in rng.integers(int(spec["dimension_min"]), int(spec["dimension_max"]) + 1, size=count):
        n = int(n)
        defender_value = rng.uniform(float(spec["value_min"]), float(spec["value_max"]), n)
        attacker_value = rng.uniform(float(spec["value_min"]), float(spec["value_max"]), n)
        baseline = rng.uniform(float(spec["baseline_min"]), float(spec["baseline_max"]), n)
        defender_cost = rng.uniform(float(spec["cost_min"]), float(spec["cost_max"]), n)
        attacker_cost = rng.uniform(float(spec["cost_min"]), float(spec["cost_max"]), n)
        symmetric = bisect_positive_root(lambda s: attacker_value * (s + baseline) / (2.0 * s + baseline) ** 2 - attacker_cost * s, n)
        initial_x = bisect_positive_root(lambda x: defender_value * baseline / (x + 2.0 * baseline) ** 2 - defender_cost * x, n)
        instances.append(Security(defender_value, attacker_value, baseline, defender_cost, attacker_cost,
                                  float(spec["budget_scale"] * np.sum(symmetric)), initial_x, spec))
    return instances


GENERATORS = {"routing": generate_routing, "security": generate_security}
