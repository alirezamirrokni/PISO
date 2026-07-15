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


CACHE_SCHEMA = 5
_CACHE_COLUMNS = [
    "run",
    "variant",
    "schema",
    "fingerprint",
    "status",
    "sample_count",
    "iteration",
    "state",
    "trace",
    "rng_state",
]


def build_fingerprint(project_root: Path, config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(f"cache-schema:{CACHE_SCHEMA}".encode("utf-8"))

    tracked = [project_root / "run.py"]
    tracked.extend(sorted((project_root / "src").rglob("*.py")))
    tracked.append(project_root / "data" / "weeks.yaml")
    for week in config["experiment"]["weeks"]:
        tracked.append(project_root / "data" / "prices" / f"2022_{str(week).zfill(2)}.csv")

    for path in tracked:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe(value: Any) -> str:
    if isinstance(value, list):
        value = "-".join(str(item) for item in value)
    text = str(value).replace("-", "m").replace(".", "p").replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9_]+", "-", text).strip("-")


def _run_tag(config: dict[str, Any]) -> str:
    exp = config["experiment"]
    return (
        f"seed{_safe(exp['seed'])}"
        f"_budget{_safe(exp['max_samples'])}"
        f"_metric{_safe(exp['metric_samples'])}"
        f"_sims{_safe(exp['simulations'])}"
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
        "residual_alphas": "ralphas",
        "shape_alpha": "shape",
        "step_normalization": "stepnorm",
        "tau": "tau",
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


def _dumps(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(_encode(value), separators=(",", ":"), ensure_ascii=False)


def _loads(value: str) -> Any:
    if not value:
        return None
    return _decode(json.loads(value))


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _row_key(run_index: int, variant: str) -> tuple[int, str]:
    return int(run_index), str(variant)


def _atomic_write_rows(
    path: Path,
    rows: dict[tuple[int, str], dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.csv")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CACHE_COLUMNS)
        writer.writeheader()
        for run_index, variant in sorted(rows):
            row = rows[(run_index, variant)]
            writer.writerow(
                {
                    "run": run_index + 1,
                    "variant": variant,
                    "schema": row["schema"],
                    "fingerprint": row["fingerprint"],
                    "status": row["status"],
                    "sample_count": row["sample_count"],
                    "iteration": row["iteration"],
                    "state": _dumps(row.get("state")),
                    "trace": _dumps(row.get("trace")),
                    "rng_state": _dumps(row["rng_state"]),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_rows(
    path: Path,
    fingerprint: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    _set_csv_field_limit()
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _CACHE_COLUMNS:
                raise ValueError("unexpected cache columns")
            for raw in reader:
                run_index = int(raw["run"]) - 1
                variant = raw["variant"]
                row = {
                    "schema": int(raw["schema"]),
                    "fingerprint": raw["fingerprint"],
                    "status": raw["status"],
                    "sample_count": int(raw["sample_count"]),
                    "iteration": int(raw["iteration"]),
                    "state": _loads(raw["state"]),
                    "trace": _loads(raw["trace"]),
                    "rng_state": _loads(raw["rng_state"]),
                }
                if row["schema"] != CACHE_SCHEMA or row["fingerprint"] != fingerprint:
                    raise ValueError("incompatible cache row")
                if row["status"] not in {"progress", "final"} or row["rng_state"] is None:
                    raise ValueError("incomplete cache row")
                rows[_row_key(run_index, variant)] = row
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return {}
    return rows


def _write_manifest(path: Path, fingerprint: str, run_tag: str, complete: bool) -> None:
    temp = path.with_name(path.stem + ".tmp.csv")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["schema", "fingerprint", "run_tag", "complete"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": fingerprint,
                "run_tag": run_tag,
                "complete": int(complete),
            }
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError("invalid manifest")
        row = rows[0]
        return {
            "schema": int(row["schema"]),
            "fingerprint": row["fingerprint"],
            "run_tag": row["run_tag"],
            "complete": bool(int(row["complete"])),
        }
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None


class CacheManager:
    """Manage one checkpoint CSV for every dataset-method combination."""

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
        self.manifest_path = self.root / "cache_manifest.csv"
        self._groups: dict[tuple[str, str], DatasetMethodCache] = {}

        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for temp in self.root.rglob("*.tmp.csv"):
            temp.unlink(missing_ok=True)
        self._check_manifest()

    def _check_manifest(self) -> None:
        manifest = _read_manifest(self.manifest_path)
        valid = bool(
            manifest
            and manifest["schema"] == CACHE_SCHEMA
            and manifest["fingerprint"] == self.fingerprint
            and manifest["run_tag"] == self.run_tag
        )
        if valid:
            return

        for path in self.root.rglob("*.csv"):
            path.unlink(missing_ok=True)
        for folder in self.root.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
        _write_manifest(self.manifest_path, self.fingerprint, self.run_tag, complete=False)

    def _group(self, week_id: str, method: str) -> "DatasetMethodCache":
        key = (week_id, method)
        if key not in self._groups:
            method_params = self.config["methods"][method]
            folder = self.root / f"week{week_id}"
            filename = f"{_method_tag(method, method_params)}_{self.run_tag}.csv"
            self._groups[key] = DatasetMethodCache(folder / filename, self.fingerprint)
        return self._groups[key]

    def job(
        self,
        week_id: str,
        run_index: int,
        method: str,
        variant: str = "",
    ) -> "JobCache":
        return JobCache(self._group(week_id, method), run_index, variant)

    def is_complete(self, required_outputs: list[Path]) -> bool:
        manifest = _read_manifest(self.manifest_path)
        return bool(manifest and manifest["complete"]) and all(
            path.is_file() and path.stat().st_size > 0 for path in required_outputs
        )

    def mark_complete(self) -> None:
        _write_manifest(self.manifest_path, self.fingerprint, self.run_tag, complete=True)


class DatasetMethodCache:
    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.rows = _read_rows(path, fingerprint)

    def get(self, run_index: int, variant: str) -> dict[str, Any] | None:
        return self.rows.get(_row_key(run_index, variant))

    def put(self, run_index: int, variant: str, row: dict[str, Any]) -> None:
        self.rows[_row_key(run_index, variant)] = row
        _atomic_write_rows(self.path, self.rows)


class JobCache:
    def __init__(
        self,
        group: DatasetMethodCache,
        run_index: int,
        variant: str,
    ) -> None:
        self.group = group
        self.run_index = run_index
        self.variant = variant

    def load_progress(self) -> dict[str, Any] | None:
        value = self.group.get(self.run_index, self.variant)
        if value is None or value["status"] != "progress" or value["state"] is None:
            return None
        return {"state": value["state"], "rng_state": value["rng_state"]}

    def save_progress(self, state: dict[str, Any], rng_state: tuple) -> None:
        self.group.put(
            self.run_index,
            self.variant,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.group.fingerprint,
                "status": "progress",
                "sample_count": int(state["sample_count"]),
                "iteration": int(state["iteration"]),
                "state": state,
                "trace": None,
                "rng_state": rng_state,
            },
        )

    def load_final(self) -> dict[str, Any] | None:
        value = self.group.get(self.run_index, self.variant)
        if value is None or value["status"] != "final" or value["trace"] is None:
            return None
        trace_data = value["trace"]
        trace = Trace(
            samples=[int(item) for item in trace_data["samples"]],
            objectives=[float(item) for item in trace_data["objectives"]],
            final_x=np.asarray(trace_data["final_x"], dtype=float),
        )
        return {"trace": trace, "rng_state": value["rng_state"]}

    def save_final(self, trace: Trace, rng_state: tuple) -> None:
        self.group.put(
            self.run_index,
            self.variant,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.group.fingerprint,
                "status": "final",
                "sample_count": int(trace.samples[-1]),
                "iteration": len(trace.samples) - 1,
                "state": None,
                "trace": {
                    "samples": trace.samples,
                    "objectives": trace.objectives,
                    "final_x": trace.final_x,
                },
                "rng_state": rng_state,
            },
        )
