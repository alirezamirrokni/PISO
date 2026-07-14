from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass


def _tqdm_class():
    """Use notebook widgets only when Python itself runs in a notebook kernel.

    A Colab shell command such as ``!python run.py`` is a subprocess, so it must
    use the standard terminal bar. Selecting tqdm.notebook merely because the
    google.colab package exists causes missing or repeatedly printed bars.
    """
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        shell_name = ""
    if shell_name == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm


@dataclass
class ProgressSummary:
    computed: int = 0
    resumed: int = 0
    cached: int = 0


class JobProgress:
    def __init__(self, owner: "ExperimentProgress") -> None:
        self.owner = owner

    def update(self, sample_count: int, iteration: int) -> None:
        self.owner.update_current(sample_count, iteration, "running")


class ExperimentProgress:
    """One informative progress bar for the whole experiment.

    The bar advances by optimization samples and its postfix reports the current
    dataset, simulation, method, iteration, and method-level sample count.
    A single bar avoids nested-widget and repeated-line output in Colab.
    """

    def __init__(self, total_jobs: int, max_samples: int) -> None:
        self.total_jobs = total_jobs
        self.max_samples = max_samples
        self.current_job = 0
        self.current_base = 0
        self.current_samples = 0
        self.current_iteration = 0
        self.current_label = ""
        self.summary = ProgressSummary()

        tqdm = _tqdm_class()
        colab_shell = (
            importlib.util.find_spec("google.colab") is not None
            and tqdm.__module__.startswith("tqdm.std")
        )
        self.bar = tqdm(
            total=total_jobs * max_samples,
            desc="Experiments",
            unit="sample",
            mininterval=0.25,
            maxinterval=1.0,
            smoothing=0.1,
            dynamic_ncols=not colab_shell,
            ncols=160 if colab_shell else None,
            leave=True,
            position=0,
            file=sys.stdout,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} samples "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
        self.handle = JobProgress(self)

    def _postfix(self, status: str) -> str:
        return (
            f"job {self.current_job}/{self.total_jobs} | {self.current_label} | "
            f"it {self.current_iteration} | "
            f"{self.current_samples:,}/{self.max_samples:,} | {status}"
        )

    def start_job(
        self,
        job_number: int,
        label: str,
        initial_samples: int,
        iteration: int,
        status: str,
    ) -> JobProgress:
        self.current_job = job_number
        self.current_base = (job_number - 1) * self.max_samples
        self.current_label = label
        self.current_samples = int(initial_samples)
        self.current_iteration = int(iteration)
        target = self.current_base + min(self.current_samples, self.max_samples)
        if target > self.bar.n:
            self.bar.update(target - self.bar.n)
        self.bar.set_postfix_str(self._postfix(status), refresh=False)
        return self.handle

    def update_current(self, sample_count: int, iteration: int, status: str) -> None:
        self.current_samples = int(sample_count)
        self.current_iteration = int(iteration)
        target = self.current_base + min(self.current_samples, self.max_samples)
        self.bar.set_postfix_str(self._postfix(status), refresh=False)
        if target > self.bar.n:
            self.bar.update(target - self.bar.n)

    def finish_job(self, status: str, sample_count: int, iteration: int) -> None:
        self.current_samples = int(sample_count)
        self.current_iteration = int(iteration)
        target = self.current_base + self.max_samples
        self.bar.set_postfix_str(self._postfix(status), refresh=False)
        if target > self.bar.n:
            self.bar.update(target - self.bar.n)
        if status == "cached":
            self.summary.cached += 1
        elif status == "resumed":
            self.summary.resumed += 1
        else:
            self.summary.computed += 1

    def close(self) -> ProgressSummary:
        self.bar.set_postfix_str(
            f"computed {self.summary.computed} | resumed {self.summary.resumed} | "
            f"cached {self.summary.cached}",
            refresh=False,
        )
        self.bar.close()
        return self.summary
