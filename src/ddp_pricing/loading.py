from __future__ import annotations

import importlib
from typing import Any


def load_factory(path: str) -> type[Any]:
    try:
        module_name, object_name = path.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"factory must have form package.module:Class, got {path!r}") from exc
    module = importlib.import_module(module_name)
    factory = getattr(module, object_name)
    if not isinstance(factory, type):
        raise TypeError(f"{path!r} does not resolve to a class")
    return factory
