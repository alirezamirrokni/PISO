from __future__ import annotations

import sys
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class ProgressSummary:
    computed: int = 0
    resumed: int = 0
    cached: int = 0


class RunProgress:
    def __init__(self, owner: "ExperimentProgress") -> None:
        self.owner = owner

    def update(self, sample_count: int, iteration: int) -> None:
        self.owner.update_run(sample_count, iteration, "running")


class ExperimentProgress:
    """Two persistent progress bars suitable for Colab and local terminals."""

    def __init__(self, total_pairs: int, simulations: int, max_samples: int) -> None:
        self.total_pairs = int(total_pairs)
        self.simulations = int(simulations)
        self.total_runs = self.total_pairs * self.simulations
        self.max_samples = int(max_samples)
        self.completed_runs = 0
        self.current_label = ""
        self.current_samples = 0
        self.current_iteration = 0
        self.summary = ProgressSummary()

        common = dict(
            mininterval=0.35,
            maxinterval=1.0,
            smoothing=0.1,
            dynamic_ncols=False,
            ncols=132,
            file=sys.stdout,
        )
        self.outer = tqdm(
            total=self.total_pairs,
            desc="Dataset-method pairs",
            unit="pair",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}] {postfix}",
            **common,
        )
        self.inner = tqdm(
            total=self.max_samples,
            desc="Current run",
            unit="sample",
            position=1,
            leave=False,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} samples "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
            **common,
        )
        self.handle = RunProgress(self)

    def _outer_postfix(self) -> str:
        return (
            f"runs {self.completed_runs}/{self.total_runs} | "
            f"computed {self.summary.computed}, resumed {self.summary.resumed}, "
            f"cached {self.summary.cached}"
        )

    def _inner_postfix(self, status: str) -> str:
        return (
            f"iter {self.current_iteration} | "
            f"{self.current_samples:,}/{self.max_samples:,} | {status}"
        )

    def start_run(
        self,
        label: str,
        initial_samples: int,
        iteration: int,
        status: str,
    ) -> RunProgress:
        self.current_label = label
        self.current_samples = int(initial_samples)
        self.current_iteration = int(iteration)
        self.inner.reset(total=self.max_samples)
        self.inner.set_description_str(f"Current: {label}", refresh=False)
        target = min(self.current_samples, self.max_samples)
        if target > 0:
            self.inner.update(target)
        self.inner.set_postfix_str(self._inner_postfix(status), refresh=True)
        self.outer.set_postfix_str(self._outer_postfix(), refresh=True)
        return self.handle

    def update_run(self, sample_count: int, iteration: int, status: str) -> None:
        self.current_samples = int(sample_count)
        self.current_iteration = int(iteration)
        target = min(self.current_samples, self.max_samples)
        self.inner.set_postfix_str(self._inner_postfix(status), refresh=False)
        if target > self.inner.n:
            self.inner.update(target - self.inner.n)

    def finish_run(self, status: str, sample_count: int, iteration: int) -> None:
        self.current_samples = int(sample_count)
        self.current_iteration = int(iteration)
        target = self.max_samples
        self.inner.set_postfix_str(self._inner_postfix(status), refresh=False)
        if target > self.inner.n:
            self.inner.update(target - self.inner.n)

        if status == "cached":
            self.summary.cached += 1
        elif status == "resumed":
            self.summary.resumed += 1
        else:
            self.summary.computed += 1
        self.completed_runs += 1
        self.outer.update(1.0 / self.simulations)
        self.outer.set_postfix_str(self._outer_postfix(), refresh=True)

    def close(self) -> ProgressSummary:
        self.inner.close()
        if self.outer.n > self.total_pairs - 1e-9:
            self.outer.n = self.total_pairs
        self.outer.set_postfix_str(self._outer_postfix(), refresh=False)
        self.outer.close()
        return self.summary
