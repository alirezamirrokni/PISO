from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from src.cache import (
    CACHE_ROW_SCHEMA,
    MANIFEST_SCHEMA,
    DatasetMethodCache,
    JobCache,
    _hash_file,
    _hash_json,
    _method_candidates,
    _method_tag,
    _read_manifest,
    _run_tag,
    _safe,
    _unlink_with_retry,
    _write_manifest,
)

CLASSIFICATION_RNG_SCHEME_VERSION = 1
_PISO_METHODS = {
    "GaussianPISO",
    "CyclePISO",
    "GaussianPISO2",
    "CyclePISO2",
}


def tau_tag(tau: float) -> str:
    return f"tau{_safe(format(float(tau), '.2f'))}"


def build_classification_run_fingerprint(
    project_root: Path,
    config: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    _hash_json(digest, config)
    digest.update(f"manifest-schema:{MANIFEST_SCHEMA}".encode("utf-8"))
    digest.update(b"problem:classification")
    _hash_file(
        digest,
        project_root,
        project_root / "src" / "classification_report.py",
    )
    return digest.hexdigest()


def _hash_classification_problem_compat(
    digest: "hashlib._Hash",
    project_root: Path,
    path: Path,
) -> None:






    content = path.read_text(encoding="utf-8")
    
    
    
    content = content.replace(
        "train_loss, _, train_predictions = self._split_metrics(",
        "train_loss, _, _ = self._split_metrics(",
        1,
    )
    content = content.replace(
        "test_loss, probabilities, test_predictions = self._split_metrics(",
        "test_loss, probabilities, predictions = self._split_metrics(",
        1,
    )
    content = content.replace(
        '            "train_accuracy": _binary_accuracy(\n'
        '                self.dataset.train_y,\n'
        '                train_predictions,\n'
        '            ),\n',
        "",
        1,
    )
    content = content.replace(
        '            "test_accuracy": _binary_accuracy(\n'
        '                self.dataset.test_y,\n'
        '                test_predictions,\n'
        '            ),',
        '            "test_accuracy": _binary_accuracy(self.dataset.test_y, predictions),',
        1,
    )
    helper_start = content.index("def _standardize_features")
    dataclass_start = content.index("@dataclass", helper_start)
    content = content[:helper_start] + content[dataclass_start:]
    content = content.replace(
        "import pandas as pd\n\n\nOFFICIAL_DATA_URL",
        "import pandas as pd\n"
        "from sklearn.metrics import accuracy_score, roc_auc_score\n"
        "from sklearn.preprocessing import StandardScaler\n\n\n"
        "OFFICIAL_DATA_URL",
        1,
    )
    content = content.replace(
        "features = _standardize_features(features_frame.to_numpy(dtype=float))",
        "features = StandardScaler().fit_transform(features_frame.to_numpy(dtype=float))",
        1,
    )
    content = content.replace(
        '"test_auc": _binary_roc_auc(self.dataset.test_y, probabilities),\n'
        '            "test_accuracy": _binary_accuracy(self.dataset.test_y, predictions),',
        '"test_auc": float(roc_auc_score(self.dataset.test_y, probabilities)),\n'
        '            "test_accuracy": float(\n'
        '                accuracy_score(self.dataset.test_y, predictions)\n'
        '            ),',
        1,
    )
    digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
    digest.update(content.encode("utf-8"))


def build_classification_method_fingerprint(
    project_root: Path,
    config: dict[str, Any],
    tau: float,
    method: str,
) -> str:
    experiment = config["experiment"]
    digest = hashlib.sha256()
    _hash_json(
        digest,
        {
            "cache_row_schema": CACHE_ROW_SCHEMA,
            "rng_scheme": CLASSIFICATION_RNG_SCHEME_VERSION,
            "problem": "classification",
            "method": method,
            "method_params": config["methods"][method],
            "seed": experiment["seed"],
            "max_samples": experiment["max_samples"],
            "metric_samples": experiment.get("metric_samples", 0),
            "tau": float(tau),
            "problem_config": config["problem"],
        },
    )
    tracked = [
        project_root / "src" / "classification_problem.py",
        project_root / "src" / "classification_methods" / "common.py",
        project_root / "src" / "classification_methods" / f"{method.lower()}.py",
        project_root / config["problem"]["data_file"],
    ]
    if method in _PISO_METHODS:
        tracked.append(
            project_root / "src" / "classification_methods" / "general_piso.py"
        )
        if method.endswith("PISO2"):
            tracked.append(
                project_root
                / "src"
                / "classification_methods"
                / "general_piso2.py"
            )
    classification_problem = project_root / "src" / "classification_problem.py"
    for path in tracked:
        if path == classification_problem:
            _hash_classification_problem_compat(digest, project_root, path)
            continue
        replacements = (
            ((b"src.pricing_methods", b"src.methods"),)
            if path == project_root / "src" / "classification_methods" / "common.py"
            else ()
        )
        _hash_file(
            digest,
            project_root,
            path,
            replacements=replacements,
        )
    return digest.hexdigest()


class ClassificationCacheManager:






    def __init__(
        self,
        output_dir: Path,
        project_root: Path,
        config: dict[str, Any],
        reset: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.project_root = project_root
        self.root = output_dir / "cache"
        self.config = config
        self.run_tag = _run_tag(config)
        self.run_fingerprint = build_classification_run_fingerprint(
            project_root, config
        )
        self.manifest_path = self.root / "cache_manifest.csv"
        self._groups: dict[tuple[str, str], DatasetMethodCache] = {}

        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for temp in self.root.rglob("*.tmp.csv"):
            temp.unlink(missing_ok=True)

        manifest = _read_manifest(self.manifest_path)
        self.legacy_mode = bool(manifest and manifest.get("legacy"))

    def _group(self, tau: float, method: str) -> DatasetMethodCache:
        tag = tau_tag(tau)
        key = (tag, method)
        if key not in self._groups:
            params = self.config["methods"][method]
            folder = self.root / tag
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"{_method_tag(method, params)}_{self.run_tag}.csv"
            path = folder / filename
            for candidate in _method_candidates(
                folder, method, list(self.config["methods"])
            ):
                if candidate != path:
                    _unlink_with_retry(candidate)
            fingerprint = build_classification_method_fingerprint(
                self.project_root,
                self.config,
                tau,
                method,
            )
            self._groups[key] = DatasetMethodCache(
                path,
                fingerprint,
                allow_legacy_fingerprint=self.legacy_mode,
            )
        return self._groups[key]

    def job(
        self,
        tau: float,
        run_index: int,
        method: str,
        variant: str = "",
    ) -> JobCache:
        return JobCache(self._group(tau, method), run_index, variant)

    def _all_method_rows_complete(self) -> bool:
        simulations = int(self.config["experiment"]["simulations"])
        for tau in self.config["experiment"]["taus"]:
            for method in self.config["methods"]:
                group = self._group(float(tau), method)
                for run_index in range(simulations):
                    row = group.get(run_index, "")
                    if (
                        row is None
                        or row.get("status") != "final"
                        or row.get("trace") is None
                        or row.get("rng_state") is None
                    ):
                        return False
        return True

    def is_complete(self, required_outputs: list[Path]) -> bool:
        manifest = _read_manifest(self.manifest_path)
        manifest_complete = bool(
            manifest
            and not manifest.get("legacy")
            and manifest["manifest_schema"] == MANIFEST_SCHEMA
            and manifest["run_fingerprint"] == self.run_fingerprint
            and manifest["run_tag"] == self.run_tag
            and manifest["complete"]
        )
        outputs_complete = all(
            path.is_file() and path.stat().st_size > 0
            for path in required_outputs
        )
        return (
            manifest_complete
            and outputs_complete
            and self._all_method_rows_complete()
        )

    def mark_complete(self) -> None:
        _write_manifest(
            self.manifest_path,
            self.run_fingerprint,
            self.run_tag,
            complete=True,
        )
