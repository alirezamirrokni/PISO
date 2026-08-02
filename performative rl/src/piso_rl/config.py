from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")
    for section in ("experiment", "environment", "policy", "methods"):
        if section not in config or not isinstance(config[section], dict):
            raise ConfigError(f"missing mapping section: {section}")

    experiment = config["experiment"]
    seeds = experiment.get("seeds", [0])
    if isinstance(seeds, int):
        seeds = list(range(seeds))
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("experiment.seeds must be a nonempty list or a positive integer")
    experiment["seeds"] = [int(seed) for seed in seeds]
    if int(experiment.get("max_trajectories", 0)) <= 0:
        raise ConfigError("experiment.max_trajectories must be positive")
    if int(experiment.get("evaluation_interval", 1000)) <= 0:
        raise ConfigError("experiment.evaluation_interval must be positive")
    if int(experiment.get("n_jobs", 1)) <= 0:
        raise ConfigError("experiment.n_jobs must be positive")

    enabled = config["methods"].get("enabled", [])
    if not isinstance(enabled, list) or not enabled:
        raise ConfigError("methods.enabled must be a nonempty list")
    return config
