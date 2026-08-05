from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


CACHE_VERSION = 3


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class RunCache:
    def __init__(
        self,
        root: str | Path,
        method_name: str,
        seed: int,
        fingerprint_payload: dict,
    ) -> None:
        self.directory = Path(root) / method_name / f"seed_{int(seed):03d}"
        self.manifest_path = self.directory / "manifest.json"
        self.progress_path = self.directory / "progress.pkl"
        self.final_path = self.directory / "final.pkl"
        self.fingerprint_payload = {
            "cache_version": CACHE_VERSION,
            **fingerprint_payload,
        }
        self.fingerprint = stable_hash(self.fingerprint_payload)
        self._runtime_started: float | None = None
        self._runtime_base = 0.0
        self.checkpoint_interval = 1


    def start_timing(self) -> None:
        saved = self.load_progress()
        base = 0.0
        if saved is not None and isinstance(saved.get("state"), dict):
            base = float(saved["state"].get("runtime_seconds", 0.0))
        self._runtime_base = base
        self._runtime_started = time.perf_counter()

    def elapsed_runtime(self) -> float:
        if self._runtime_started is None:
            return float(self._runtime_base)
        return float(self._runtime_base + time.perf_counter() - self._runtime_started)

    def stop_timing(self) -> float:
        value = self.elapsed_runtime()
        self._runtime_base = value
        self._runtime_started = None
        return value

    def reset(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory)

    def _manifest_matches(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("fingerprint") == self.fingerprint

    def _ensure_manifest(self) -> None:
        if self.manifest_path.exists() and not self._manifest_matches():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            _atomic_json(
                self.manifest_path,
                {
                    "fingerprint": self.fingerprint,
                    "payload": self.fingerprint_payload,
                },
            )

    def load_progress(self) -> dict | None:
        if not self._manifest_matches() or not self.progress_path.exists():
            return None
        try:
            with self.progress_path.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError):
            return None

    def save_progress(self, state: dict, rng_state: tuple) -> None:
        iteration = int(state.get("iteration", 0))
        interval = max(int(self.checkpoint_interval), 1)
        if iteration > 0 and iteration % interval != 0:
            return
        self._ensure_manifest()
        state_to_save = dict(state)
        state_to_save["runtime_seconds"] = self.elapsed_runtime()
        _atomic_pickle(
            self.progress_path,
            {"state": state_to_save, "rng_state": rng_state},
        )

    def load_final(self):
        if not self._manifest_matches() or not self.final_path.exists():
            return None
        try:
            with self.final_path.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError):
            return None

    def save_final(self, trace) -> None:
        self._ensure_manifest()
        _atomic_pickle(self.final_path, trace)
        try:
            self.progress_path.unlink()
        except FileNotFoundError:
            pass
