from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.pricing_methods.common import Trace
from src.problem import PricingProblem, ProblemSpec


CACHE_ROW_SCHEMA = 5
MANIFEST_SCHEMA = 1
RNG_SCHEME_VERSION = 2

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
_PROBLEM_COLUMNS = [
    "week",
    "run",
    "schema",
    "fingerprint",
    "rho",
    "rng_state",
]
_MANIFEST_COLUMNS = [
    "manifest_schema",
    "run_fingerprint",
    "run_tag",
    "complete",
]


def _hash_json(digest: "hashlib._Hash", value: Any) -> None:
    digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _hash_file(
    digest: "hashlib._Hash",
    project_root: Path,
    path: Path,
    *,
    logical_path: str | None = None,
    replacements: tuple[tuple[bytes, bytes], ...] = (),
) -> None:





    label = logical_path or path.relative_to(project_root).as_posix()
    content = path.read_bytes()
    for current, legacy in replacements:
        content = content.replace(current, legacy)
    digest.update(label.encode("utf-8"))
    digest.update(content)


def build_run_fingerprint(project_root: Path, config: dict[str, Any]) -> str:






    digest = hashlib.sha256()
    _hash_json(digest, config)
    digest.update(f"manifest-schema:{MANIFEST_SCHEMA}".encode("utf-8"))
    _hash_file(digest, project_root, project_root / "src" / "report.py")
    return digest.hexdigest()


def build_problem_fingerprint(
    project_root: Path,
    config: dict[str, Any],
    week_id: str,
) -> str:
    digest = hashlib.sha256()
    experiment = config["experiment"]
    _hash_json(
        digest,
        {
            "seed": experiment["seed"],
            "problem": config["problem"],
            "week": week_id,
            "rng_scheme": RNG_SCHEME_VERSION,
        },
    )
    _hash_file(digest, project_root, project_root / "src" / "problem.py")
    _hash_file(
        digest,
        project_root,
        project_root / "data" / "prices" / f"2022_{week_id}.csv",
    )
    return digest.hexdigest()


def build_method_fingerprint(
    project_root: Path,
    config: dict[str, Any],
    week_id: str,
    method: str,
) -> str:





    experiment = config["experiment"]
    digest = hashlib.sha256()
    _hash_json(
        digest,
        {
            "cache_row_schema": CACHE_ROW_SCHEMA,
            "rng_scheme": RNG_SCHEME_VERSION,
            "method": method,
            "method_params": config["methods"][method],
            "seed": experiment["seed"],
            "max_samples": experiment["max_samples"],
            "metric_samples": experiment["metric_samples"],
            "problem": config["problem"],
            "week": week_id,
        },
    )
    pricing_methods = project_root / "src" / "pricing_methods"
    method_filename = f"{method.lower()}.py"
    tracked_paths: list[tuple[Path, str | None, tuple[tuple[bytes, bytes], ...]]] = [
        (project_root / "src" / "problem.py", None, ()),
        (pricing_methods / "common.py", "src/methods/common.py", ()),
        (
            pricing_methods / method_filename,
            f"src/methods/{method_filename}",
            ((b"src.pricing_methods", b"src.methods"),),
        ),
        (project_root / "data" / "prices" / f"2022_{week_id}.csv", None, ()),
    ]
    if method in {"GaussianPISO", "CyclePISO", "GaussianPISO2", "CyclePISO2"}:
        tracked_paths.append(
            (
                pricing_methods / "general_piso.py",
                "src/methods/general_piso.py",
                (),
            )
        )
        if method.endswith("PISO2"):
            tracked_paths.append(
                (
                    pricing_methods / "general_piso2.py",
                    "src/methods/general_piso2.py",
                    (),
                )
            )
    for path, logical_path, replacements in tracked_paths:
        _hash_file(
            digest,
            project_root,
            path,
            logical_path=logical_path,
            replacements=replacements,
        )
    return digest.hexdigest()


def _safe(value: Any) -> str:
    if isinstance(value, list):
        value = "-".join(str(item) for item in value)
    text = str(value).replace("-", "m").replace(".", "p").replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9_]+", "-", text).strip("-")


def _compact_safe(value: Any) -> str:

    if isinstance(value, float):
        value = format(value, ".6g")
    elif isinstance(value, list):
        value = "-".join(format(item, ".6g") if isinstance(item, float) else str(item) for item in value)
    return _safe(value)


def _run_tag(config: dict[str, Any]) -> str:
    exp = config["experiment"]
    return (
        f"seed{_safe(exp['seed'])}"
        f"_budget{_safe(exp['max_samples'])}"
        f"_metric{_safe(exp['metric_samples'])}"
        f"_sims{_safe(exp['simulations'])}"
    )


def _method_tag(method: str, params: dict[str, Any]) -> str:

    if method in {"GaussianPISO", "CyclePISO", "GaussianPISO2", "CyclePISO2"}:
        keys = [
            ("radius", "r"), ("radius_decay", "rd"),
            ("step_size", "s"), ("step_decay", "sd"),
            ("gamma", "g"), ("rho", "rho"),
            ("known_fraction", "kf"),
        ]
        if method.endswith("PISO2"):
            keys.append(("lambda", "l"))
        if method in {"CyclePISO", "CyclePISO2"}:
            keys.append(("cycle_length", "c"))
    else:
        keys = [
            ("mu0", "mu0"), ("mu", "mu"), ("mu_min", "umin"),
            ("beta0", "b0"), ("beta_decay", "bd"), ("mu_decay", "ud"),
            ("alpha0", "a0"), ("alpha_damping", "ad"), ("window", "w"),
            ("initial_samples", "i"), ("s_max", "s"), ("M", "M"),
        ]

    parts = [method]
    for key, label in keys:
        if key in params:
            parts.append(f"{label}{_compact_safe(params[key])}")
    if method in {"GaussianPISO", "CyclePISO", "GaussianPISO2", "CyclePISO2"}:
        config_digest = hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        parts.append(f"cfg{config_digest}")
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


def _replace_with_retry(temp: Path, destination: Path) -> None:
    delay = 0.05
    attempts = 30
    for attempt in range(attempts):
        try:
            os.replace(temp, destination)
            return
        except PermissionError as error:
            if attempt == attempts - 1:
                raise PermissionError(
                    f"Could not replace checkpoint after {attempts} attempts: {destination}. "
                    "Close Excel, editors, File Explorer preview panes, cloud-sync clients, "
                    "or antivirus tools that may be holding the CSV open."
                ) from error
            time.sleep(delay)
            delay = min(delay * 1.5, 1.0)


def _unlink_with_retry(path: Path) -> None:
    if not path.exists():
        return
    delay = 0.05
    for attempt in range(20):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 0.75)


def _atomic_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.csv")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _row_key(run_index: int, variant: str) -> tuple[int, str]:
    return int(run_index), str(variant)


def _atomic_write_cache_rows(
    path: Path,
    rows: dict[tuple[int, str], dict[str, Any]],
) -> None:
    serialized = []
    for run_index, variant in sorted(rows):
        row = rows[(run_index, variant)]
        serialized.append(
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
    _atomic_csv(path, _CACHE_COLUMNS, serialized)


def _read_cache_rows(
    path: Path,
    fingerprint: str,
    allow_legacy_fingerprint: bool,
) -> tuple[dict[tuple[int, str], dict[str, Any]], bool]:
    if not path.exists():
        return {}, False
    _set_csv_field_limit()
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    migrated = False
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
                if row["schema"] != CACHE_ROW_SCHEMA:
                    raise ValueError("incompatible cache schema")
                if row["fingerprint"] != fingerprint:
                    if not allow_legacy_fingerprint:
                        raise ValueError("incompatible method fingerprint")
                    row["fingerprint"] = fingerprint
                    migrated = True
                if row["status"] not in {"progress", "final"} or row["rng_state"] is None:
                    raise ValueError("incomplete cache row")
                rows[_row_key(run_index, variant)] = row
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
        _unlink_with_retry(path)
        return {}, False
    return rows, migrated


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames or []
        if len(rows) != 1:
            raise ValueError("invalid manifest")
        row = rows[0]
        if fields == _MANIFEST_COLUMNS:
            return {
                "legacy": False,
                "manifest_schema": int(row["manifest_schema"]),
                "run_fingerprint": row["run_fingerprint"],
                "run_tag": row["run_tag"],
                "complete": bool(int(row["complete"])),
            }
        if fields == ["schema", "fingerprint", "run_tag", "complete"]:
            return {
                "legacy": True,
                "manifest_schema": int(row["schema"]),
                "run_fingerprint": row["fingerprint"],
                "run_tag": row["run_tag"],
                "complete": bool(int(row["complete"])),
            }
        raise ValueError("unexpected manifest columns")
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        return None


def _write_manifest(path: Path, run_fingerprint: str, run_tag: str, complete: bool) -> None:
    _atomic_csv(
        path,
        _MANIFEST_COLUMNS,
        [
            {
                "manifest_schema": MANIFEST_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "run_tag": run_tag,
                "complete": int(complete),
            }
        ],
    )


def _method_candidates(
    folder: Path,
    method: str,
    configured_methods: list[str],
) -> list[Path]:






    ordered = sorted(configured_methods, key=len, reverse=True)
    candidates: list[Path] = []
    for path in folder.glob("*.csv"):
        owner = next(
            (name for name in ordered if path.name.startswith(f"{name}_")),
            None,
        )
        if owner == method:
            candidates.append(path)
    return candidates


class ProblemCache:


    def __init__(self, path: Path, project_root: Path, config: dict[str, Any]) -> None:
        self.path = path
        self.project_root = project_root
        self.config = config
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        self._read()

    def _read(self) -> None:
        if not self.path.exists():
            return
        _set_csv_field_limit()
        try:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != _PROBLEM_COLUMNS:
                    raise ValueError("unexpected problem cache columns")
                for raw in reader:
                    key = (raw["week"], int(raw["run"]) - 1)
                    self.rows[key] = {
                        "schema": int(raw["schema"]),
                        "fingerprint": raw["fingerprint"],
                        "rho": _loads(raw["rho"]),
                        "rng_state": _loads(raw["rng_state"]),
                    }
        except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
            _unlink_with_retry(self.path)
            self.rows = {}

    def _write(self) -> None:
        serialized = []
        for week_id, run_index in sorted(self.rows):
            row = self.rows[(week_id, run_index)]
            serialized.append(
                {
                    "week": week_id,
                    "run": run_index + 1,
                    "schema": row["schema"],
                    "fingerprint": row["fingerprint"],
                    "rho": _dumps(row["rho"]),
                    "rng_state": _dumps(row["rng_state"]),
                }
            )
        _atomic_csv(self.path, _PROBLEM_COLUMNS, serialized)

    def get_or_create(
        self,
        data_dir: Path,
        week_id: str,
        run_index: int,
        rng: np.random.RandomState,
        spec: ProblemSpec,
    ) -> PricingProblem:
        key = (week_id, int(run_index))
        expected = build_problem_fingerprint(self.project_root, self.config, week_id)
        row = self.rows.get(key)
        if (
            row is not None
            and row["schema"] == CACHE_ROW_SCHEMA
            and row["fingerprint"] == expected
            and row["rho"] is not None
            and row["rng_state"] is not None
        ):
            rng.set_state(row["rng_state"])
            return PricingProblem.from_week_with_rho(data_dir, week_id, row["rho"], spec)

        problem = PricingProblem.from_week(data_dir, week_id, rng, spec)
        self.rows[key] = {
            "schema": CACHE_ROW_SCHEMA,
            "fingerprint": expected,
            "rho": problem.rho.copy(),
            "rng_state": rng.get_state(),
        }
        self._write()
        return problem


class CacheManager:


    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        config: dict[str, Any],
        reset: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.project_root = project_root
        self.root = output_dir / "cache"
        self.config = config
        self.run_tag = _run_tag(config)
        self.run_fingerprint = build_run_fingerprint(project_root, config)
        self.manifest_path = self.root / "cache_manifest.csv"
        self._groups: dict[tuple[str, str], DatasetMethodCache] = {}

        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for temp in self.root.rglob("*.tmp.csv"):
            temp.unlink(missing_ok=True)

        manifest = _read_manifest(self.manifest_path)
        self.legacy_mode = bool(manifest and manifest.get("legacy"))
        self.problem_cache = ProblemCache(
            self.root / "problem_instances.csv",
            project_root,
            config,
        )

    def problem(
        self,
        data_dir: Path,
        week_id: str,
        run_index: int,
        rng: np.random.RandomState,
        spec: ProblemSpec,
    ) -> PricingProblem:
        return self.problem_cache.get_or_create(
            data_dir,
            week_id,
            run_index,
            rng,
            spec,
        )

    def _group(self, week_id: str, method: str) -> "DatasetMethodCache":
        key = (week_id, method)
        if key not in self._groups:
            method_params = self.config["methods"][method]
            folder = self.root / f"week{week_id}"
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"{_method_tag(method, method_params)}_{self.run_tag}.csv"
            path = folder / filename

            
            
            for candidate in _method_candidates(
                folder, method, list(self.config["methods"])
            ):
                if candidate != path:
                    _unlink_with_retry(candidate)

            fingerprint = build_method_fingerprint(
                self.project_root,
                self.config,
                week_id,
                method,
            )
            self._groups[key] = DatasetMethodCache(
                path,
                fingerprint,
                allow_legacy_fingerprint=self.legacy_mode,
            )
        return self._groups[key]

    def job(
        self,
        week_id: str,
        run_index: int,
        method: str,
        variant: str = "",
    ) -> "JobCache":
        return JobCache(self._group(week_id, method), run_index, variant)

    def _all_method_rows_complete(self) -> bool:





        simulations = int(self.config["experiment"]["simulations"])
        for week_value in self.config["experiment"]["weeks"]:
            week_id = str(week_value).zfill(2)
            for method in self.config["methods"]:
                group = self._group(week_id, method)
                for run_index in range(simulations):
                    row = group.get(run_index, "")
                    if (
                        row is None
                        or row.get("status") != "final"
                        or row.get("trace") is None
                        or row.get("rng_state") is None
                    ):
                        return False
        return True

    def is_complete(self, required_outputs: list[Path]) -> bool:
        manifest = _read_manifest(self.manifest_path)
        manifest_complete = bool(
            manifest
            and not manifest.get("legacy")
            and manifest["manifest_schema"] == MANIFEST_SCHEMA
            and manifest["run_fingerprint"] == self.run_fingerprint
            and manifest["run_tag"] == self.run_tag
            and manifest["complete"]
        )
        outputs_complete = all(
            path.is_file() and path.stat().st_size > 0 for path in required_outputs
        )
        return manifest_complete and outputs_complete and self._all_method_rows_complete()

    def mark_complete(self) -> None:
        _write_manifest(
            self.manifest_path,
            self.run_fingerprint,
            self.run_tag,
            complete=True,
        )


class DatasetMethodCache:
    def __init__(
        self,
        path: Path,
        fingerprint: str,
        allow_legacy_fingerprint: bool,
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.rows, migrated = _read_cache_rows(
            path,
            fingerprint,
            allow_legacy_fingerprint,
        )
        if migrated:
            _atomic_write_cache_rows(self.path, self.rows)

    def get(self, run_index: int, variant: str) -> dict[str, Any] | None:
        return self.rows.get(_row_key(run_index, variant))

    def put(self, run_index: int, variant: str, row: dict[str, Any]) -> None:
        self.rows[_row_key(run_index, variant)] = row
        _atomic_write_cache_rows(self.path, self.rows)


class JobCache:
    def __init__(self, group: DatasetMethodCache, run_index: int, variant: str) -> None:
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
                "schema": CACHE_ROW_SCHEMA,
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
                "schema": CACHE_ROW_SCHEMA,
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
