from __future__ import annotations

import importlib.util
from typing import Any


def _in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def progress_bar(*args: Any, **kwargs: Any):
    """Return a widget-backed bar in Colab and a terminal bar elsewhere."""
    kwargs.setdefault("mininterval", 0.25)
    kwargs.setdefault("maxinterval", 1.0)
    kwargs.setdefault("dynamic_ncols", True)
    if _in_colab():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm(*args, **kwargs)


def reset_bar(bar, total: int, initial: int, description: str) -> None:
    bar.reset(total=total)
    bar.set_description(description, refresh=False)
    bar.set_postfix_str("", refresh=False)
    if initial > 0:
        bar.update(min(initial, total))
    else:
        bar.refresh()
