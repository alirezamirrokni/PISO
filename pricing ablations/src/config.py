from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for key in ("experiment", "protocols", "profiles", "problem", "defaults", "suites"):
        if key not in config:
            raise ValueError(f"missing top-level configuration section: {key}")

    experiment = config["experiment"]
    if int(experiment["max_samples"]) <= 0:
        raise ValueError("experiment.max_samples must be positive")
    if int(experiment["metric_samples"]) <= 0:
        raise ValueError("experiment.metric_samples must be positive")
    if int(experiment["report_grid_points"]) < 2:
        raise ValueError("experiment.report_grid_points must be at least 2")

    for name, values in config["defaults"].items():
        validate_algorithm_params(values, f"defaults.{name}")

    known_suites = set(config["suites"])
    for profile_name, profile in config["profiles"].items():
        missing = set(profile.get("suites", [])).difference(known_suites)
        if missing:
            raise ValueError(
                f"profile {profile_name!r} references unknown suites: {sorted(missing)}"
            )


def validate_algorithm_params(params: dict[str, Any], context: str = "variant") -> None:
    algorithm = str(params["algorithm"])
    if algorithm not in {"piso", "piso2"}:
        raise ValueError(f"{context}: algorithm must be 'piso' or 'piso2'")

    residual_mode = str(params["residual_mode"])
    if residual_mode not in {"recursive", "fresh_centered", "fresh_uncentered"}:
        raise ValueError(
            f"{context}: residual_mode must be recursive, fresh_centered, or fresh_uncentered"
        )

    if str(params["direction"]) not in {
        "gaussian",
        "sphere",
        "rademacher",
        "random_coordinate",
        "coordinate_cycle",
        "orthogonal_cycle",
    }:
        raise ValueError(f"{context}: unsupported direction family")

    if str(params["difference"]) not in {"central", "forward"}:
        raise ValueError(f"{context}: difference must be central or forward")
    if str(params["coupling"]) not in {"independent", "common_random_numbers"}:
        raise ValueError(f"{context}: unsupported coupling")
    if str(params["momentum_initialization"]) not in {"zero", "first_residual"}:
        raise ValueError(f"{context}: unsupported momentum_initialization")

    for key in ("radius", "step_size", "batch_initial", "directions_per_step"):
        if float(params[key]) <= 0.0:
            raise ValueError(f"{context}: {key} must be positive")
    for key in ("radius_decay", "step_decay"):
        value = float(params[key])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{context}: {key} must lie in (0, 1]")
    rho = float(params["rho"])
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"{context}: rho must lie in [0, 1)")
    mixing = float(params["lambda"])
    if not 0.0 <= mixing <= 1.0:
        raise ValueError(f"{context}: lambda must lie in [0, 1]")
    known_fraction = float(params["known_fraction"])
    if not 0.0 <= known_fraction < 1.0:
        raise ValueError(f"{context}: known_fraction must lie in [0, 1)")
    if bool(params["use_known"]) and float(params["gamma"]) != 0.0 and known_fraction <= 0.0:
        raise ValueError(
            f"{context}: a nonzero known term requires known_fraction > 0"
        )
    if int(params["batch_increment"]) < 0:
        raise ValueError(f"{context}: batch_increment must be nonnegative")
    if int(params["cycle_length"]) <= 0:
        raise ValueError(f"{context}: cycle_length must be positive")
    if float(params["known_noise_std"]) < 0.0:
        raise ValueError(f"{context}: known_noise_std must be nonnegative")
    mask_fraction = float(params["known_mask_fraction"])
    if not 0.0 < mask_fraction <= 1.0:
        raise ValueError(f"{context}: known_mask_fraction must lie in (0, 1]")
    if int(params["known_delay"]) < 0:
        raise ValueError(f"{context}: known_delay must be nonnegative")


def apply_profile(
    config: dict[str, Any],
    profile_name: str,
    *,
    suites_override: list[str] | None = None,
    simulations_override: int | None = None,
) -> dict[str, Any]:
    if profile_name not in config["profiles"]:
        raise ValueError(
            f"unknown profile {profile_name!r}; available: {sorted(config['profiles'])}"
        )
    resolved = deepcopy(config)
    profile = resolved["profiles"][profile_name]
    experiment = resolved["experiment"]
    for key in (
        "weeks",
        "max_samples",
        "metric_samples",
        "report_grid_points",
        "bootstrap_resamples",
        "permutation_resamples",
    ):
        if key in profile:
            experiment[key] = profile[key]
    resolved["active_suites"] = (
        list(suites_override) if suites_override is not None else list(profile["suites"])
    )
    resolved["profile_name"] = profile_name
    selected_simulations = simulations_override
    if selected_simulations is None and "simulations" in profile:
        selected_simulations = int(profile["simulations"])
    if selected_simulations is not None:
        if int(selected_simulations) <= 0:
            raise ValueError("simulation count must be positive")
        resolved["simulations_override"] = int(selected_simulations)
    return resolved
