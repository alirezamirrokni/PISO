from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any


CACHE_SCHEMA = 2


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


class CacheManager:
    def __init__(self, output_dir: Path, fingerprint: str, reset: bool = False) -> None:
        self.output_dir = output_dir
        self.root = output_dir / ".cache"
        self.fingerprint = fingerprint
        self.manifest_path = self.root / "manifest.json"

        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for temp in self.root.glob("*.tmp"):
            temp.unlink(missing_ok=True)
        self._check_manifest()

    def _check_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if manifest.get("fingerprint") != self.fingerprint:
                shutil.rmtree(self.root)
                self.root.mkdir(parents=True, exist_ok=True)
                self.manifest_path = self.root / "manifest.json"
                self._write_json(
                    self.manifest_path,
                    {"schema": CACHE_SCHEMA, "fingerprint": self.fingerprint, "complete": False},
                )
        else:
            self._write_json(
                self.manifest_path,
                {"schema": CACHE_SCHEMA, "fingerprint": self.fingerprint, "complete": False},
            )

    @staticmethod
    def _atomic_pickle(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp, path)

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def job(self, week_id: str, run_index: int, method: str) -> "JobCache":
        return JobCache(self.root / f"{week_id}_{run_index:03d}_{method}")

    def is_complete(self, required_outputs: list[Path]) -> bool:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return bool(manifest.get("complete")) and all(path.is_file() and path.stat().st_size > 0 for path in required_outputs)

    def mark_complete(self) -> None:
        self._write_json(
            self.manifest_path,
            {"schema": CACHE_SCHEMA, "fingerprint": self.fingerprint, "complete": True},
        )


class JobCache:
    def __init__(self, prefix: Path) -> None:
        self.progress_path = prefix.with_suffix(".progress.pkl")
        self.final_path = prefix.with_suffix(".final.pkl")

    @staticmethod
    def _load(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(value, dict) or "rng_state" not in value:
            path.unlink(missing_ok=True)
            return None
        return value

    def load_progress(self) -> dict[str, Any] | None:
        value = self._load(self.progress_path)
        if value is not None and "state" not in value:
            self.progress_path.unlink(missing_ok=True)
            return None
        return value

    def save_progress(self, state: dict[str, Any], rng_state: tuple) -> None:
        CacheManager._atomic_pickle(self.progress_path, {"state": state, "rng_state": rng_state})

    def load_final(self) -> dict[str, Any] | None:
        value = self._load(self.final_path)
        if value is not None and "trace" not in value:
            self.final_path.unlink(missing_ok=True)
            return None
        return value

    def save_final(self, trace: Any, rng_state: tuple) -> None:
        CacheManager._atomic_pickle(self.final_path, {"trace": trace, "rng_state": rng_state})
        self.progress_path.unlink(missing_ok=True)
