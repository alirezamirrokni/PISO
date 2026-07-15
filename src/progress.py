from __future__ import annotations

import sys
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class ProgressSummary:
    computed: int = 0
    resumed: int = 0
    cached: int = 0


class VariantProgress:
    def __init__(self, owner: "ExperimentProgress") -> None:
        self.owner = owner

    def update(self, sample_count: int, iteration: int) -> None:
        self.owner.update_variant(sample_count, iteration, "running")


class ExperimentProgress:
    """Two persistent terminal bars that render cleanly in Colab subprocesses.

    The outer bar measures dataset-method pairs across all simulations. The inner bar reports the
    current method's sample progress and, for PISO/PISO-M, spans all alpha
    variants without creating a new tqdm object for every iteration.
    """

    def __init__(self, total_pairs: int, simulations: int, max_samples: int) -> None:
        self.total_pairs = int(total_pairs)
        self.simulations = int(simulations)
        self.max_samples = int(max_samples)
        self.summary = ProgressSummary()
        self.group_number = 0
        self.group_label = ""
        self.variant_index = 0
        self.variant_count = 1
        self.variant_label = ""
        self.variant_base = 0
        self.current_samples = 0
        self.current_iteration = 0

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
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            **common,
        )
        self.inner = tqdm(
            total=self.max_samples,
            desc="Current method",
            unit="sample",
            position=1,
            leave=False,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} samples "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
            **common,
        )
        self.handle = VariantProgress(self)

    def start_group(
        self,
        group_number: int,
        label: str,
        variant_count: int,
    ) -> None:
        self.group_number = int(group_number)
        self.group_label = label
        self.variant_count = int(variant_count)
        total = self.variant_count * self.max_samples
        self.inner.reset(total=total)
        self.inner.set_description_str(f"Current: {label}", refresh=False)
        self.outer.set_postfix_str(
            f"{label} | computed {self.summary.computed}, "
            f"resumed {self.summary.resumed}, cached {self.summary.cached}",
            refresh=True,
        )

    def start_variant(
        self,
        variant_index: int,
        variant_label: str,
        initial_samples: int,
        iteration: int,
        status: str,
    ) -> VariantProgress:
        self.variant_index = int(variant_index)
        self.variant_label = variant_label
        self.variant_base = (self.variant_index - 1) * self.max_samples
        self.current_samples = int(initial_samples)
        self.current_iteration = int(iteration)
        target = self.variant_base + min(self.current_samples, self.max_samples)
        if target > self.inner.n:
            self.inner.update(target - self.inner.n)
        self.inner.set_postfix_str(self._inner_postfix(status), refresh=True)
        return self.handle

    def _inner_postfix(self, status: str) -> str:
        variant = self.variant_label or "default"
        return (
            f"variant {self.variant_index}/{self.variant_count}: {variant} | "
            f"iter {self.current_iteration} | "
            f"{self.current_samples:,}/{self.max_samples:,} | {status}"
        )

    def update_variant(self, sample_count: int, iteration: int, status: str) -> None:
        self.current_samples = int(sample_count)
        self.current_iteration = int(iteration)
        target = self.variant_base + min(self.current_samples, self.max_samples)
        self.inner.set_postfix_str(self._inner_postfix(status), refresh=False)
        if target > self.inner.n:
            self.inner.update(target - self.inner.n)

    def finish_variant(self, status: str, sample_count: int, iteration: int) -> None:
        self.current_samples = int(sample_count)
        self.current_iteration = int(iteration)
        target = self.variant_base + self.max_samples
        self.inner.set_postfix_str(self._inner_postfix(status), refresh=False)
        if target > self.inner.n:
            self.inner.update(target - self.inner.n)
        if status == "cached":
            self.summary.cached += 1
        elif status == "resumed":
            self.summary.resumed += 1
        else:
            self.summary.computed += 1

    def finish_group(self) -> None:
        # Each simulation contributes an equal fraction of its dataset-method
        # pair. This preserves the released execution order while the outer bar
        # still measures exactly dataset-method pairs.
        self.outer.update(1.0 / self.simulations)
        self.outer.set_postfix_str(
            f"finished {self.group_label} | computed {self.summary.computed}, "
            f"resumed {self.summary.resumed}, cached {self.summary.cached}",
            refresh=True,
        )

    def close(self) -> ProgressSummary:
        self.inner.close()
        if self.outer.n > self.total_pairs - 1e-9:
            self.outer.n = self.total_pairs
        self.outer.set_postfix_str(
            f"computed {self.summary.computed} | resumed {self.summary.resumed} | "
            f"cached {self.summary.cached}",
            refresh=False,
        )
        self.outer.close()
        return self.summary
