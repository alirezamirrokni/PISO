from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.methods.common import Trace


CACHE_SCHEMA = 3


def build_fingerprint(project_root: Path, config: dict[str, Any]) -> str:
    """Fingerprint the configuration, implementation, and selected input data."""
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(f"cache-schema:{CACHE_SCHEMA}".encode("utf-8"))

    tracked = [project_root / "run.py"]
    tracked.extend(sorted((project_root / "src").rglob("*.py")))
    tracked.append(project_root / "data" / "weeks.yaml")
    for week in config["experiment"]["weeks"]:
        tracked.append(project_root / "data" / "prices" / f"2022_{str(week).zfill(2)}.csv")

    for path in tracked:
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe(value: Any) -> str:
    text = str(value)
    text = text.replace("-", "m").replace(".", "p").replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9_]+", "-", text).strip("-")


def _run_tag(config: dict[str, Any]) -> str:
    exp = config["experiment"]
    weeks = "-".join(str(value).zfill(2) for value in exp["weeks"])
    return (
        f"seed{_safe(exp['seed'])}"
        f"_budget{_safe(exp['max_samples'])}"
        f"_metric{_safe(exp['metric_samples'])}"
        f"_sims{_safe(exp['simulations'])}"
        f"_weeks{weeks}"
    )


def _method_tag(method: str, params: dict[str, Any]) -> str:
    labels = {
        "mu0": "mu0",
        "mu": "mu",
        "mu_min": "mumin",
        "beta0": "beta0",
        "beta_decay": "bdecay",
        "mu_decay": "mdecay",
        "alpha0": "alpha0",
        "alpha_damping": "adamp",
        "batch_initial": "batch0",
        "batch_increment": "batchinc",
        "window": "window",
        "initial_samples": "init",
        "s_max": "smax",
        "M": "M",
    }
    parts = [method]
    for key, label in labels.items():
        if key in params:
            parts.append(f"{label}{_safe(params[key])}")
    return "_".join(parts)


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported cache value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        marker = value.get("__type__")
        if marker == "ndarray":
            array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"]))
        if marker == "tuple":
            return tuple(_decode(item) for item in value["items"])
        return {key: _decode(item) for key, item in value.items()}
    return value


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.csv")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"])
        writer.writeheader()
        for key, value in payload.items():
            writer.writerow(
                {
                    "key": key,
                    "value": json.dumps(_encode(value), separators=(",", ":"), ensure_ascii=False),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    _set_csv_field_limit()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["key", "value"]:
                raise ValueError("unexpected columns")
            payload = {
                row["key"]: _decode(json.loads(row["value"]))
                for row in reader
                if row.get("key") and row.get("value") is not None
            }
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    return payload


class CacheManager:
    def __init__(
        self,
        output_dir: Path,
        fingerprint: str,
        config: dict[str, Any],
        reset: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.root = output_dir / "cache"
        self.fingerprint = fingerprint
        self.config = config
        self.run_tag = _run_tag(config)
        self.manifest_path = self.root / f"manifest_{self.run_tag}.csv"

        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for temp in self.root.glob("*.tmp.csv"):
            temp.unlink(missing_ok=True)
        self._check_manifest()

    def _check_manifest(self) -> None:
        manifest = _read_payload(self.manifest_path)
        if manifest is not None and manifest.get("fingerprint") == self.fingerprint:
            return

        for path in self.root.glob(f"{self.run_tag}__*.csv"):
            path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)
        _write_payload(
            self.manifest_path,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "complete": False,
                "run_tag": self.run_tag,
            },
        )

    def job(self, week_id: str, run_index: int, method: str) -> "JobCache":
        method_params = self.config["methods"][method]
        filename = (
            f"{self.run_tag}"
            f"__week{week_id}"
            f"_run{run_index + 1:03d}"
            f"_{_method_tag(method, method_params)}.csv"
        )
        return JobCache(self.root / filename, self.fingerprint)

    def is_complete(self, required_outputs: list[Path]) -> bool:
        manifest = _read_payload(self.manifest_path)
        return bool(manifest and manifest.get("complete")) and all(
            path.is_file() and path.stat().st_size > 0 for path in required_outputs
        )

    def mark_complete(self) -> None:
        _write_payload(
            self.manifest_path,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "complete": True,
                "run_tag": self.run_tag,
            },
        )


class JobCache:
    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint

    def _load(self) -> dict[str, Any] | None:
        value = _read_payload(self.path)
        if value is None:
            return None
        if value.get("schema") != CACHE_SCHEMA or value.get("fingerprint") != self.fingerprint:
            self.path.unlink(missing_ok=True)
            return None
        if "rng_state" not in value:
            self.path.unlink(missing_ok=True)
            return None
        return value

    def load_progress(self) -> dict[str, Any] | None:
        value = self._load()
        if value is None or value.get("status") != "progress" or "state" not in value:
            return None
        return {"state": value["state"], "rng_state": value["rng_state"]}

    def save_progress(self, state: dict[str, Any], rng_state: tuple) -> None:
        _write_payload(
            self.path,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "status": "progress",
                "sample_count": int(state["sample_count"]),
                "iteration": int(state["iteration"]),
                "state": state,
                "rng_state": rng_state,
            },
        )

    def load_final(self) -> dict[str, Any] | None:
        value = self._load()
        if value is None or value.get("status") != "final" or "trace" not in value:
            return None
        trace_data = value["trace"]
        trace = Trace(
            samples=[int(item) for item in trace_data["samples"]],
            objectives=[float(item) for item in trace_data["objectives"]],
            final_x=np.asarray(trace_data["final_x"], dtype=float),
        )
        return {"trace": trace, "rng_state": value["rng_state"]}

    def save_final(self, trace: Trace, rng_state: tuple) -> None:
        _write_payload(
            self.path,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "status": "final",
                "sample_count": int(trace.samples[-1]),
                "trace": {
                    "samples": trace.samples,
                    "objectives": trace.objectives,
                    "final_x": trace.final_x,
                },
                "rng_state": rng_state,
            },
        )
