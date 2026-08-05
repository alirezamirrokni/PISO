from __future__ import annotations

import hashlib
import inspect
import shutil
from pathlib import Path
from typing import Any

from src.util import atomic_pickle_dump, hash_mapping, pickle_load


CACHE_SCHEMA = 1
INSTANCE_SCHEMA = 1


def _source_hash(*objects: object) -> str:
    """Hash complete source modules, not only individual class bodies."""
    digest = hashlib.sha256()
    seen: set[str] = set()
    for object_ in objects:
        source_file = inspect.getsourcefile(object_)
        if source_file is not None:
            path = Path(source_file).resolve()
            key = str(path)
            if key not in seen and path.exists():
                digest.update(path.name.encode("utf-8"))
                digest.update(path.read_bytes())
                seen.add(key)
                continue
        digest.update(inspect.getsource(object_).encode("utf-8"))
    return digest.hexdigest()

# Removing the unused guided variants does not change the implementation of the
# four retained PISO classes.  Keep their prior module fingerprint so existing
# completed/in-flight caches remain reusable after this source cleanup.
_RETAINED_PISO_METHODS = {
    "GaussianPISO",
    "CyclePISO",
    "GaussianPISO2",
    "CyclePISO2",
}
_LEGACY_PISO_SOURCE_HASH = "c47e0d91fb8886844dc5bbe462207ada1efece645bf647e773deb0e6072bf739"


class CacheManager:
    def __init__(self, output_dir: Path, family: str, config: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.family = family
        self.config = config
        self.root = output_dir / "cache"
        self.instances_file = output_dir / "instances" / f"{family}_instances.pkl"
        self.root.mkdir(parents=True, exist_ok=True)

    def clear_all_methods(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def instance_fingerprint(self, generator: object) -> str:
        relevant = {
            "schema": INSTANCE_SCHEMA,
            "family": self.family,
            "seed": self.config["experiment"]["seed"],
            "instances": self.config["experiment"]["instances"],
            "problem": self.config["problem"],
            "source": _source_hash(generator),
        }
        return hash_mapping(relevant)

    def load_instances(self, fingerprint: str) -> list[Any] | None:
        if not self.instances_file.exists():
            return None
        payload = pickle_load(self.instances_file)
        if payload.get("schema") != INSTANCE_SCHEMA or payload.get("fingerprint") != fingerprint:
            return None
        return payload["instances"]

    def save_instances(self, fingerprint: str, instances: list[Any]) -> None:
        atomic_pickle_dump(
            {
                "schema": INSTANCE_SCHEMA,
                "fingerprint": fingerprint,
                "instances": instances,
            },
            self.instances_file,
        )

    def method_path(self, method: str, instance_id: int) -> Path:
        return self.root / method / f"instance_{instance_id:04d}.pkl"

    def method_fingerprint(
        self,
        method: str,
        method_class: object,
        problem_class: object,
        instance_id: int,
    ) -> str:
        method_source = _source_hash(method_class, method_class.__mro__[1])
        if method in _RETAINED_PISO_METHODS:
            method_source = _LEGACY_PISO_SOURCE_HASH
        relevant = {
            "schema": CACHE_SCHEMA,
            "family": self.family,
            "method": method,
            "method_config": self.config["methods"][method],
            "experiment": {
                "seed": self.config["experiment"]["seed"],
                "iterations": self.config["experiment"]["iterations"],
                "checkpoint_interval": self.config["experiment"]["checkpoint_interval"],
            },
            "instance_id": instance_id,
            "problem": self.config["problem"],
            "method_source": method_source,
            "problem_source": _source_hash(problem_class),
        }
        return hash_mapping(relevant)

    def load_method(self, method: str, instance_id: int, fingerprint: str) -> dict[str, Any] | None:
        path = self.method_path(method, instance_id)
        if not path.exists():
            return None
        payload = pickle_load(path)
        if payload.get("schema") != CACHE_SCHEMA or payload.get("fingerprint") != fingerprint:
            return None
        return payload

    def save_method(
        self,
        method: str,
        instance_id: int,
        fingerprint: str,
        state: dict[str, Any],
        *,
        complete: bool,
    ) -> None:
        payload = {
            "schema": CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "complete": bool(complete),
            "state": state,
        }
        atomic_pickle_dump(payload, self.method_path(method, instance_id))
