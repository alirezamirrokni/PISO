from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np


@dataclass
class RoutingInstance:
    tails: np.ndarray
    heads: np.ndarray
    latency_a: np.ndarray
    latency_b: np.ndarray
    origins: np.ndarray
    destinations: np.ndarray
    demands: np.ndarray
    sensitivities: np.ndarray
    regularization: float
    wardrop_tolerance: float
    wardrop_max_iterations: int

    family: str = "routing"
    sense: str = "max"

    def __post_init__(self) -> None:
        self.tails = np.asarray(self.tails, dtype=np.int64)
        self.heads = np.asarray(self.heads, dtype=np.int64)
        self.latency_a = np.asarray(self.latency_a, dtype=float)
        self.latency_b = np.asarray(self.latency_b, dtype=float)
        self.origins = np.asarray(self.origins, dtype=np.int64)
        self.destinations = np.asarray(self.destinations, dtype=np.int64)
        self.demands = np.asarray(self.demands, dtype=float)
        self.sensitivities = np.asarray(self.sensitivities, dtype=float)

        if self.tails.ndim != 1 or self.heads.ndim != 1:
            raise ValueError("tails and heads must be one-dimensional")
        if self.tails.size == 0 or self.tails.shape != self.heads.shape:
            raise ValueError("routing graph must contain at least one valid edge")
        if np.any(self.tails == self.heads):
            raise ValueError("self-loops are not supported")
        if np.any(self.sensitivities <= 0.0):
            raise ValueError("all toll sensitivities must be positive")
        if np.any(self.demands <= 0.0):
            raise ValueError("all demands must be positive")

        self.n_nodes = int(max(self.tails.max(), self.heads.max()) + 1)
        self.dimension = int(self.tails.size)
        self.n_groups = int(self.demands.size)

        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.n_nodes)]
        indegree = np.zeros(self.n_nodes, dtype=np.int64)
        for edge, (tail, head) in enumerate(zip(self.tails, self.heads, strict=True)):
            u = int(tail)
            v = int(head)
            adjacency[u].append((v, int(edge)))
            indegree[v] += 1
        for outgoing in adjacency:
            outgoing.sort(key=lambda item: (item[0], item[1]))
        self._adjacency = adjacency

        # The paper permits unconstrained tolls.  Negative perturbed tolls can
        # therefore produce negative edge costs.  A directed acyclic graph is a
        # clean way to keep shortest paths well-defined without clipping tolls
        # or changing the ZOS/PZOS updates.  The paper does not specify its graph
        # generator, so this is an explicit independent-reproduction choice.
        queue = [node for node in range(self.n_nodes) if indegree[node] == 0]
        heapq.heapify(queue)
        topological_order: list[int] = []
        mutable_indegree = indegree.copy()
        while queue:
            u = heapq.heappop(queue)
            topological_order.append(u)
            for v, _ in adjacency[u]:
                mutable_indegree[v] -= 1
                if mutable_indegree[v] == 0:
                    heapq.heappush(queue, v)

        if len(topological_order) != self.n_nodes:
            raise ValueError(
                "routing instance is cyclic; regenerate instances with the updated "
                "DAG generator"
            )
        self._topological_order = np.asarray(topological_order, dtype=np.int64)

    def initial_decision(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=float)

    def project(self, x: np.ndarray) -> np.ndarray:
        # Figures 2 and 3 use the unconstrained toll problem tau in R^|E|.
        return np.asarray(x, dtype=float)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        toll = np.asarray(x, dtype=float)
        flow = np.asarray(y, dtype=float)
        return float(np.dot(toll, flow) - self.regularization * np.dot(toll, toll))

    def partial_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - 2.0 * self.regularization * np.asarray(x, dtype=float)

    def partial_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        return np.asarray(x, dtype=float).copy()

    def _shortest_path_edges(self, weights: np.ndarray, source: int, target: int) -> list[int]:
        """Return a minimum-cost path in the generated DAG.

        Topological dynamic programming is valid for arbitrary real edge
        weights, so negative toll perturbations cannot create predecessor cycles.
        """

        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.dimension,):
            raise ValueError(
                f"expected {self.dimension} edge weights, received {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("shortest-path weights contain NaN or infinity")

        source = int(source)
        target = int(target)
        if source == target:
            return []

        distance = np.full(self.n_nodes, np.inf, dtype=float)
        previous_edge = np.full(self.n_nodes, -1, dtype=np.int64)
        distance[source] = 0.0

        for u_raw in self._topological_order:
            u = int(u_raw)
            if not np.isfinite(distance[u]):
                continue
            for v, edge in self._adjacency[u]:
                candidate = distance[u] + float(weights[edge])
                # Deterministic tie handling keeps paired ZOS/PZOS evaluations
                # reproducible on Windows and Unix.
                if candidate < distance[v] - 1.0e-14:
                    distance[v] = candidate
                    previous_edge[v] = edge

        if not np.isfinite(distance[target]):
            raise RuntimeError(f"no directed path from {source} to {target}")

        path: list[int] = []
        node = target
        for _ in range(self.n_nodes):
            if node == source:
                path.reverse()
                return path
            edge = int(previous_edge[node])
            if edge < 0:
                raise RuntimeError(f"could not reconstruct path from {source} to {target}")
            path.append(edge)
            node = int(self.tails[edge])

        # This is unreachable for a DAG and is kept as a defensive assertion.
        raise RuntimeError("path reconstruction exceeded the number of graph nodes")

    def _all_or_nothing(self, toll: np.ndarray, aggregate_flow: np.ndarray) -> np.ndarray:
        latency = self.latency_a * aggregate_flow + self.latency_b
        assignment = np.zeros((self.n_groups, self.dimension), dtype=float)
        for group in range(self.n_groups):
            perceived = latency + toll / self.sensitivities[group]
            path = self._shortest_path_edges(
                perceived,
                int(self.origins[group]),
                int(self.destinations[group]),
            )
            assignment[group, path] = self.demands[group]
        return assignment

    def _initial_feasible_flow(self, toll: np.ndarray) -> np.ndarray:
        return self._all_or_nothing(toll, np.zeros(self.dimension, dtype=float))

    def response(self, x: np.ndarray, warm: Any | None = None) -> tuple[np.ndarray, Any]:
        toll = np.asarray(x, dtype=float)
        if toll.shape != (self.dimension,):
            raise ValueError(
                f"expected toll vector of shape {(self.dimension,)}, received {toll.shape}"
            )
        if not np.all(np.isfinite(toll)):
            raise FloatingPointError("toll vector contains NaN or infinity")

        if isinstance(warm, dict) and "group_flow" in warm:
            group_flow = np.asarray(warm["group_flow"], dtype=float).copy()
            if group_flow.shape != (self.n_groups, self.dimension):
                group_flow = self._initial_feasible_flow(toll)
        else:
            group_flow = self._initial_feasible_flow(toll)

        for _ in range(int(self.wardrop_max_iterations)):
            aggregate = group_flow.sum(axis=0)
            target_flow = self._all_or_nothing(toll, aggregate)
            direction = target_flow - group_flow
            aggregate_direction = direction.sum(axis=0)

            edge_gradient = self.latency_a * aggregate + self.latency_b
            directional = float(np.dot(edge_gradient, aggregate_direction))
            directional += float(
                np.sum((toll[None, :] / self.sensitivities[:, None]) * direction)
            )
            gap = -directional
            scale = 1.0 + float(np.dot(edge_gradient, aggregate))
            if gap <= self.wardrop_tolerance * scale:
                break

            curvature = float(np.dot(self.latency_a, aggregate_direction**2))
            if curvature <= 1.0e-18:
                step = 1.0 if directional < 0.0 else 0.0
            else:
                step = float(np.clip(-directional / curvature, 0.0, 1.0))
            group_flow += step * direction

        aggregate = group_flow.sum(axis=0)
        if not np.all(np.isfinite(aggregate)):
            raise FloatingPointError("Wardrop solver produced non-finite flow")
        return aggregate, {"group_flow": group_flow}


def generate_routing_instances(config: dict, seed: int) -> list[RoutingInstance]:
    """Generate deterministic heterogeneous routing instances.

    The paper specifies the size and parameter ranges but not the random graph
    construction.  We generate directed acyclic graphs.  A random Hamiltonian
    path guarantees connectivity for every sampled OD pair, and extra edges are
    sampled only in the forward topological direction.  This preserves the
    paper's unconstrained toll domain while keeping shortest paths well-defined
    even when finite-difference queries contain negative toll components.
    """

    exp = config["experiment"]
    problem = config["problem"]
    count = int(exp["instances"])
    rng = np.random.default_rng(seed)
    instances: list[RoutingInstance] = []

    for _ in range(count):
        n_nodes = int(
            rng.integers(
                int(problem["vertices_min"]),
                int(problem["vertices_max"]) + 1,
            )
        )

        # A DAG on n vertices has at most n(n-1)/2 directed edges.
        maximum_edges = min(
            int(problem["edges_max"]),
            n_nodes * (n_nodes - 1) // 2,
        )
        minimum_edges = min(
            max(int(problem["edges_min"]), n_nodes - 1),
            maximum_edges,
        )
        n_edges = int(rng.integers(minimum_edges, maximum_edges + 1))

        permutation = np.asarray(rng.permutation(n_nodes), dtype=np.int64)

        # Backbone: every earlier topological position can reach every later one.
        edge_set: set[tuple[int, int]] = {
            (int(permutation[index]), int(permutation[index + 1]))
            for index in range(n_nodes - 1)
        }

        forward_pairs = [
            (int(permutation[left]), int(permutation[right]))
            for left in range(n_nodes - 1)
            for right in range(left + 1, n_nodes)
            if right != left + 1
        ]
        rng.shuffle(forward_pairs)
        for edge in forward_pairs:
            if len(edge_set) >= n_edges:
                break
            edge_set.add(edge)

        if len(edge_set) != n_edges:
            raise RuntimeError(
                f"could only generate {len(edge_set)} of {n_edges} requested DAG edges"
            )
        edges = np.asarray(sorted(edge_set), dtype=np.int64)

        n_groups = int(
            rng.integers(
                int(problem["groups_min"]),
                int(problem["groups_max"]) + 1,
            )
        )

        # Sample OD pairs in topological order.  The backbone then guarantees at
        # least one directed path for every group.
        origin_positions = rng.integers(0, n_nodes - 1, size=n_groups)
        destination_positions = np.asarray(
            [rng.integers(int(position) + 1, n_nodes) for position in origin_positions],
            dtype=np.int64,
        )
        origins = permutation[origin_positions]
        destinations = permutation[destination_positions]

        demand_values = np.asarray(problem["demand_values"], dtype=float)
        demands = rng.choice(demand_values, size=n_groups, replace=True)

        instances.append(
            RoutingInstance(
                tails=edges[:, 0],
                heads=edges[:, 1],
                latency_a=rng.uniform(
                    float(problem["latency_a_min"]),
                    float(problem["latency_a_max"]),
                    size=n_edges,
                ),
                latency_b=rng.uniform(
                    float(problem["latency_b_min"]),
                    float(problem["latency_b_max"]),
                    size=n_edges,
                ),
                origins=origins,
                destinations=destinations,
                demands=demands,
                sensitivities=rng.uniform(
                    float(problem["sensitivity_min"]),
                    float(problem["sensitivity_max"]),
                    size=n_groups,
                ),
                regularization=float(problem["regularization"]),
                wardrop_tolerance=float(problem["wardrop_tolerance"]),
                wardrop_max_iterations=int(problem["wardrop_max_iterations"]),
            )
        )

    return instances