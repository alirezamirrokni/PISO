from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable


CACHE_SCHEMA = 2


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        # A partial/corrupt cache entry is simply treated as missing. The next
        # successful run will atomically replace it with a valid entry.
        return None


class JobCache:
    def __init__(
        self,
        root: Path,
        fingerprint: str,
        compatible_fingerprints: Iterable[str] = (),
    ) -> None:
        self.root = Path(root)
        self.fingerprint = str(fingerprint)
        self.valid_fingerprints = {
            self.fingerprint,
            *(str(value) for value in compatible_fingerprints),
        }
        self.progress_path = self.root / "progress.pkl"
        self.final_path = self.root / "final.pkl"

    def _valid(self, payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("schema") == CACHE_SCHEMA
            and payload.get("fingerprint") in self.valid_fingerprints
            and "payload" in payload
        )

    def _load(self, path: Path) -> dict[str, Any] | None:
        payload = load_pickle(path)
        if not self._valid(payload):
            return None

        # Transparently migrate a cache produced by the immediately previous
        # fingerprinting scheme. This keeps existing expensive results usable,
        # while future non-numerical source edits no longer invalidate them.
        if payload.get("fingerprint") != self.fingerprint:
            atomic_pickle(
                path,
                {
                    "schema": CACHE_SCHEMA,
                    "fingerprint": self.fingerprint,
                    "payload": payload["payload"],
                },
            )
        return payload["payload"]

    def load_progress(self) -> dict[str, Any] | None:
        return self._load(self.progress_path)

    def save_progress(self, payload: dict[str, Any]) -> None:
        atomic_pickle(
            self.progress_path,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "payload": payload,
            },
        )

    def clear_progress(self) -> None:
        self.progress_path.unlink(missing_ok=True)

    def load_final(self) -> dict[str, Any] | None:
        return self._load(self.final_path)

    def save_final(self, payload: dict[str, Any]) -> None:
        atomic_pickle(
            self.final_path,
            {
                "schema": CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "payload": payload,
            },
        )
        self.clear_progress()
