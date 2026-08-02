from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


CACHE_VERSION = 1


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
        self._ensure_manifest()
        _atomic_pickle(
            self.progress_path,
            {"state": state, "rng_state": rng_state},
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
