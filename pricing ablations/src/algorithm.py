from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from .directions import initialize_direction_state, sample_directions


@dataclass
class Trace:
    samples: list[int]
    objectives: list[float]
    known_norms: list[float]
    residual_norms: list[float]
    momentum_norms: list[float]
    gradient_norms: list[float]
    innovation_rms: list[float]
    final_x: np.ndarray
    iterations: int
    runtime_seconds: float
    known_training_samples: int
    zeroth_order_samples: int
    failed: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["final_x"] = np.asarray(self.final_x, dtype=float)
        return result


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _step_size(params: dict[str, Any], iteration: int) -> float:
    return float(params["step_size"]) * float(params["step_decay"]) ** int(iteration)


def _radius(params: dict[str, Any], iteration: int, dimension: int) -> float:
    value = float(params["radius"]) * float(params["radius_decay"]) ** int(iteration)
    # Gaussian, sphere, Rademacher, coordinate and orthogonal directions are
    # all normalized so E[u u^T] = I. The /sqrt(d) convention matches the
    # current Gaussian and cycle pricing implementations.
    return value / np.sqrt(float(dimension))


def _iteration_budget(params: dict[str, Any], iteration: int) -> int:
    batch = int(params["batch_initial"]) + int(params["batch_increment"]) * int(iteration)
    return 3 * max(batch, 1)


def allocate_budget(
    total: int,
    *,
    known_fraction: float,
    use_known: bool,
    use_residual: bool,
    directions_per_step: int,
) -> tuple[int, int, int]:
    """Return known samples, samples per finite-difference arm, and used total.

    Every residual direction uses two sample groups for both central and forward
    differences. The final one-sample remainder, when present, is assigned to
    the known component if it is active; otherwise it is left unused.
    """
    total = int(total)
    directions_per_step = int(directions_per_step)
    if total <= 0:
        return 0, 0, 0

    if not use_residual:
        return (total if use_known else 0), 0, (total if use_known else 0)

    minimum_zo = 2 * directions_per_step
    if total < minimum_zo + (1 if use_known else 0):
        return 0, 0, 0

    known = int(round(total * float(known_fraction))) if use_known else 0
    if use_known:
        known = max(1, min(known, total - minimum_zo))
    else:
        known = 0

    arm_count = (total - known) // (2 * directions_per_step)
    if arm_count <= 0:
        return 0, 0, 0
    used = known + 2 * directions_per_step * arm_count
    remainder = total - used
    if use_known and remainder > 0:
        known += remainder
        used += remainder
    return known, arm_count, used


def _evaluate(
    problem,
    x: np.ndarray,
    metric_samples: int,
    base_seed: int,
    week_id: str,
    run_index: int,
    sample_count: int,
) -> float:
    rng = np.random.RandomState(
        stable_seed(base_seed, "metric", week_id, run_index, sample_count)
    )
    return problem.evaluate(x, int(metric_samples), rng)


def _known_gradient(
    problem,
    x: np.ndarray,
    count: int,
    rng: np.random.RandomState,
    state: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    if count <= 0 or not bool(params["use_known"]):
        return np.zeros(problem.n, dtype=float)

    demands = problem.sample_demands(x, count, rng)
    current = problem.partial_gradients(demands).mean(axis=0)

    mask_fraction = float(params["known_mask_fraction"])
    if mask_fraction < 1.0:
        mask = state.get("known_mask")
        if mask is None:
            coordinates = max(1, int(round(mask_fraction * problem.n)))
            indices = rng.choice(problem.n, size=coordinates, replace=False)
            mask = np.zeros(problem.n, dtype=float)
            mask[indices] = 1.0
            state["known_mask"] = mask
        current = current * np.asarray(mask, dtype=float)

    noise_std = float(params["known_noise_std"])
    if noise_std > 0.0:
        scale = max(float(np.linalg.norm(current)) / np.sqrt(problem.n), 1e-12)
        current = current + noise_std * scale * rng.normal(size=problem.n)

    history = list(state.get("known_history", []))
    history.append(np.asarray(current, dtype=float))
    delay = int(params["known_delay"])
    keep = max(delay + 1, 1)
    if len(history) > keep:
        history = history[-keep:]
    state["known_history"] = history
    if delay <= 0:
        return np.asarray(history[-1], dtype=float)
    if len(history) <= delay:
        return np.asarray(history[0], dtype=float)
    return np.asarray(history[-delay - 1], dtype=float)


def _directional_derivative(
    problem,
    x: np.ndarray,
    direction: np.ndarray,
    mu: float,
    arm_count: int,
    rng: np.random.RandomState,
    difference: str,
    coupling: str,
) -> float:
    if coupling == "common_random_numbers":
        uniforms = rng.rand(int(arm_count), problem.m)
        plus, _ = problem.losses_from_uniforms(x + mu * direction, uniforms)
        if difference == "central":
            minus, _ = problem.losses_from_uniforms(x - mu * direction, uniforms)
            return float((plus.mean() - minus.mean()) / (2.0 * mu))
        baseline, _ = problem.losses_from_uniforms(x, uniforms)
        return float((plus.mean() - baseline.mean()) / mu)

    plus, _ = problem.sample_losses(x + mu * direction, arm_count, rng)
    if difference == "central":
        minus, _ = problem.sample_losses(x - mu * direction, arm_count, rng)
        return float((plus.mean() - minus.mean()) / (2.0 * mu))
    baseline, _ = problem.sample_losses(x, arm_count, rng)
    return float((plus.mean() - baseline.mean()) / mu)


def _initial_state(
    problem,
    params: dict[str, Any],
    rng: np.random.RandomState,
    objective: float,
) -> dict[str, Any]:
    return {
        "x": problem.initial_x.copy(),
        "momentum": np.zeros(problem.n, dtype=float),
        "sample_count": 0,
        "iteration": 0,
        "samples": [0],
        "objectives": [float(objective)],
        "known_norms": [0.0],
        "residual_norms": [0.0],
        "momentum_norms": [0.0],
        "gradient_norms": [0.0],
        "innovation_rms": [0.0],
        "direction_state": initialize_direction_state(
            str(params["direction"]),
            problem.n,
            int(params["cycle_length"]),
            rng,
        ),
        "known_history": [],
        "known_mask": None,
        "failed": False,
        "failure_reason": "",
        "known_training_samples": 0,
        "zeroth_order_samples": 0,
    }


def run_piso_variant(
    problem,
    params: dict[str, Any],
    *,
    base_seed: int,
    week_id: str,
    run_index: int,
    variant_seed_key: str,
    max_samples: int,
    metric_samples: int,
    checkpoint_every: int,
    load_checkpoint: Callable[[], dict[str, Any] | None],
    save_checkpoint: Callable[[dict[str, Any]], None],
    clear_checkpoint: Callable[[], None],
) -> Trace:
    import time

    start_time = time.perf_counter()
    training_rng = np.random.RandomState(
        stable_seed(base_seed, "train", week_id, run_index, variant_seed_key)
    )
    checkpoint = load_checkpoint()
    if checkpoint is None:
        initial_objective = _evaluate(
            problem,
            problem.initial_x,
            metric_samples,
            base_seed,
            week_id,
            run_index,
            0,
        )
        state = _initial_state(problem, params, training_rng, initial_objective)
    else:
        state = checkpoint["state"]
        training_rng.set_state(checkpoint["rng_state"])

    while int(state["sample_count"]) < int(max_samples) and not state["failed"]:
        iteration = int(state["iteration"])
        remaining = int(max_samples) - int(state["sample_count"])
        target_budget = min(_iteration_budget(params, iteration), remaining)
        known_count, arm_count, used_budget = allocate_budget(
            target_budget,
            known_fraction=float(params["known_fraction"]),
            use_known=bool(params["use_known"]),
            use_residual=bool(params["use_residual"]),
            directions_per_step=int(params["directions_per_step"]),
        )
        if used_budget <= 0:
            break

        x = np.asarray(state["x"], dtype=float)
        momentum = np.asarray(state["momentum"], dtype=float)
        known = _known_gradient(
            problem,
            x,
            known_count,
            training_rng,
            state,
            params,
        )
        gamma = float(params["gamma"])

        residual = np.zeros(problem.n, dtype=float)
        innovation_values: list[float] = []
        if bool(params["use_residual"]):
            direction_count = int(params["directions_per_step"])
            directions, scores = sample_directions(
                str(params["direction"]),
                problem.n,
                direction_count,
                int(params["cycle_length"]),
                training_rng,
                state["direction_state"],
            )
            mu = _radius(params, iteration, problem.n)
            derivatives = [
                _directional_derivative(
                    problem,
                    x,
                    direction,
                    mu,
                    arm_count,
                    training_rng,
                    str(params["difference"]),
                    str(params["coupling"]),
                )
                for direction in directions
            ]
            zo_gradient = np.mean(
                [derivative * score for derivative, score in zip(derivatives, scores)],
                axis=0,
            )

            mode = str(params["residual_mode"])
            rho = float(params["rho"])
            if mode == "recursive":
                # Estimate the residual component itself, then apply temporal
                # momentum to that residual below:
                #   R_k = mean_j[(D_{k,j} - <gamma g_K, u_{k,j}>) s_{k,j}],
                #   m_{k+1} = rho m_k + (1-rho) R_k.
                # Keeping m_k inside the directional control variate makes the
                # recursion excessively aggressive for moderate rho (notably
                # rho=0.5) and can cause the observed blow-up.
                control = gamma * known
                innovation_values = [
                    float(derivative - np.dot(control, direction))
                    for derivative, direction in zip(derivatives, directions)
                ]
                residual = np.mean(
                    [
                        innovation * score
                        for innovation, score in zip(innovation_values, scores)
                    ],
                    axis=0,
                )
            elif mode == "fresh_centered":
                # Fresh directional control variate with no recursive state:
                #   R_k = mean_j[(D_{k,j} - <gamma g_K, u_{k,j}>) s_{k,j}].
                # When E[s u^T] = I, E[R_k] approximates g - gamma g_K.
                control = gamma * known
                innovation_values = [
                    float(derivative - np.dot(control, direction))
                    for derivative, direction in zip(derivatives, directions)
                ]
                residual = np.mean(
                    [
                        innovation * score
                        for innovation, score in zip(innovation_values, scores)
                    ],
                    axis=0,
                )
            elif mode == "fresh_uncentered":
                residual = zo_gradient
                innovation_values = [float(value) for value in derivatives]
            else:
                raise ValueError(f"unsupported residual mode: {mode}")

            if iteration == 0 and str(params["momentum_initialization"]) == "first_residual":
                new_momentum = residual.copy()
            else:
                new_momentum = rho * momentum + (1.0 - rho) * residual
        else:
            residual = np.zeros(problem.n, dtype=float)
            new_momentum = np.zeros(problem.n, dtype=float)

        if str(params["algorithm"]) == "piso2" and bool(params["use_residual"]):
            mixing = float(params["lambda"])
            unknown_component = mixing * residual + (1.0 - mixing) * new_momentum
        else:
            unknown_component = new_momentum
        gradient = gamma * known + unknown_component
        beta = _step_size(params, iteration)
        new_x = x - beta * gradient

        arrays = [known, residual, new_momentum, gradient, new_x]
        if not all(np.all(np.isfinite(array)) for array in arrays):
            state["failed"] = True
            state["failure_reason"] = "non-finite iterate or gradient"
            break
        if float(np.linalg.norm(new_x)) > 1e12:
            state["failed"] = True
            state["failure_reason"] = "iterate norm exceeded 1e12"
            break

        state["x"] = new_x
        state["momentum"] = new_momentum
        state["sample_count"] = int(state["sample_count"]) + used_budget
        state["known_training_samples"] = int(state.get("known_training_samples", 0)) + known_count
        state["zeroth_order_samples"] = int(state.get("zeroth_order_samples", 0)) + (used_budget - known_count)
        state["iteration"] = iteration + 1

        objective = _evaluate(
            problem,
            new_x,
            metric_samples,
            base_seed,
            week_id,
            run_index,
            int(state["sample_count"]),
        )
        if not np.isfinite(objective):
            state["failed"] = True
            state["failure_reason"] = "non-finite metric objective"
            break

        state["samples"].append(int(state["sample_count"]))
        state["objectives"].append(float(objective))
        state["known_norms"].append(float(np.linalg.norm(known)))
        state["residual_norms"].append(float(np.linalg.norm(residual)))
        state["momentum_norms"].append(float(np.linalg.norm(new_momentum)))
        state["gradient_norms"].append(float(np.linalg.norm(gradient)))
        state["innovation_rms"].append(
            float(np.sqrt(np.mean(np.square(innovation_values))))
            if innovation_values
            else 0.0
        )

        if int(checkpoint_every) > 0 and state["iteration"] % int(checkpoint_every) == 0:
            save_checkpoint({"state": state, "rng_state": training_rng.get_state()})

    clear_checkpoint()
    runtime = time.perf_counter() - start_time
    return Trace(
        samples=[int(value) for value in state["samples"]],
        objectives=[float(value) for value in state["objectives"]],
        known_norms=[float(value) for value in state["known_norms"]],
        residual_norms=[float(value) for value in state["residual_norms"]],
        momentum_norms=[float(value) for value in state["momentum_norms"]],
        gradient_norms=[float(value) for value in state["gradient_norms"]],
        innovation_rms=[float(value) for value in state["innovation_rms"]],
        final_x=np.asarray(state["x"], dtype=float),
        iterations=int(state["iteration"]),
        runtime_seconds=float(runtime),
        known_training_samples=int(state.get("known_training_samples", 0)),
        zeroth_order_samples=int(state.get("zeroth_order_samples", 0)),
        failed=bool(state["failed"]),
        failure_reason=str(state["failure_reason"]),
    )
