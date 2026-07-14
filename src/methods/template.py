from __future__ import annotations

from src.methods.common import finish, initial_state, record, restore_or_initialize, save_step


class MyMethod:
    name = "MY_METHOD"

    def __init__(self, params: dict) -> None:
        self.p = params

    def run(self, problem, rng, context, cache, progress=None):
        state, _ = restore_or_initialize(
            cache,
            rng,
            lambda: initial_state(problem, context.metric_samples, rng),
        )

        while state["sample_count"] <= context.max_samples:
            # Update state["x"] and increase state["sample_count"] here.
            state["iteration"] += 1
            record(state, problem, context.metric_samples, rng)
            save_step(cache, state, rng, progress)

        return finish(state)
