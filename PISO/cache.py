from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any


class CacheManager:
    def __init__(self, output_dir: Path, config: dict[str, Any], reset: bool = False) -> None:
        self.output_dir = output_dir
        self.root = output_dir / ".cache"
        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for temp in self.root.glob("*.tmp"):
            temp.unlink(missing_ok=True)
        self.fingerprint = self._fingerprint(config)
        self.manifest_path = self.root / "manifest.json"
        self._check_manifest()

    @staticmethod
    def _fingerprint(config: dict[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _check_manifest(self) -> None:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != self.fingerprint:
                raise RuntimeError(
                    "The output directory contains checkpoints for a different configuration. "
                    "Use another output directory or add --reset-cache."
                )
        else:
            self._write_json(self.manifest_path, {"fingerprint": self.fingerprint, "complete": False})

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
        return bool(manifest.get("complete")) and all(path.exists() for path in required_outputs)

    def mark_complete(self) -> None:
        self._write_json(self.manifest_path, {"fingerprint": self.fingerprint, "complete": True})


class JobCache:
    def __init__(self, prefix: Path) -> None:
        self.progress_path = prefix.with_suffix(".progress.pkl")
        self.final_path = prefix.with_suffix(".final.pkl")

    def load_progress(self) -> dict[str, Any] | None:
        if not self.progress_path.exists():
            return None
        with self.progress_path.open("rb") as handle:
            return pickle.load(handle)

    def save_progress(self, state: dict[str, Any], rng_state: tuple) -> None:
        CacheManager._atomic_pickle(self.progress_path, {"state": state, "rng_state": rng_state})

    def load_final(self) -> dict[str, Any] | None:
        if not self.final_path.exists():
            return None
        with self.final_path.open("rb") as handle:
            return pickle.load(handle)

    def save_final(self, trace: Any, rng_state: tuple) -> None:
        CacheManager._atomic_pickle(self.final_path, {"trace": trace, "rng_state": rng_state})
        self.progress_path.unlink(missing_ok=True)
