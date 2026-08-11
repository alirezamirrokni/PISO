from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd


OFFICIAL_DATA_URL = (
    "https://raw.githubusercontent.com/ustunb/actionable-recourse/"
    "master/examples/paper/data/credit_processed.csv"
)
EXPECTED_COLUMNS = (
    "NoDefaultNextMonth",
    "Married",
    "Single",
    "Age_lt_25",
    "Age_in_25_to_40",
    "Age_in_40_to_59",
    "Age_geq_60",
    "EducationLevel",
    "MaxBillAmountOverLast6Months",
    "MaxPaymentAmountOverLast6Months",
    "MonthsWithZeroBalanceOverLast6Months",
    "MonthsWithLowSpendingOverLast6Months",
    "MonthsWithHighSpendingOverLast6Months",
    "MostRecentBillAmount",
    "MostRecentPaymentAmount",
    "TotalOverdueCounts",
    "TotalMonthsOverdue",
    "HistoryOfOverduePayments",
)
DROPPED_DEMOGRAPHICS = (
    "Married",
    "Single",
    "Age_lt_25",
    "Age_in_25_to_40",
    "Age_in_40_to_59",
    "Age_geq_60",
)


def _standardize_features(values: np.ndarray) -> np.ndarray:

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional")
    means = array.mean(axis=0)
    scales = array.std(axis=0, ddof=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    return (array - means) / scales


def _binary_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_values = np.asarray(labels, dtype=float).reshape(-1)
    predicted_values = np.asarray(predictions, dtype=float).reshape(-1)
    if true_values.shape != predicted_values.shape:
        raise ValueError("Labels and predictions must have the same shape")
    if true_values.size == 0:
        raise ValueError("Accuracy is undefined for an empty dataset")
    return float(np.mean(true_values == predicted_values))


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:

    true_values = np.asarray(labels, dtype=float).reshape(-1)
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    if true_values.shape != score_values.shape:
        raise ValueError("Labels and scores must have the same shape")
    if true_values.size == 0:
        raise ValueError("ROC AUC is undefined for an empty dataset")
    if not np.all(np.isin(true_values, (0.0, 1.0))):
        raise ValueError("ROC AUC expects binary labels encoded as 0 and 1")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("ROC AUC scores must be finite")

    positive_count = int(np.count_nonzero(true_values == 1.0))
    negative_count = int(np.count_nonzero(true_values == 0.0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both positive and negative labels")

    order = np.argsort(score_values, kind="mergesort")
    sorted_scores = score_values[order]
    ranks = np.empty(score_values.size, dtype=float)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.50 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = float(ranks[true_values == 1.0].sum())
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


@dataclass(frozen=True)
class ClassificationSpec:
    data_file: str
    data_url: str
    dataset_seed: int
    train_size: int
    test_size: int
    initial_value: float
    reward: float
    auto_download: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ClassificationSpec":
        return cls(**values)


@dataclass(frozen=True)
class ClassificationDataset:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_names: tuple[str, ...]


def ensure_classification_data(project_root: Path, spec: ClassificationSpec) -> Path:
    path = project_root / spec.data_file
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not spec.auto_download:
        raise FileNotFoundError(
            f"Classification data are missing: {path}. Run "
            "python tools/download_classification_data.py"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    url = spec.data_url or OFFICIAL_DATA_URL
    try:
        with urlopen(url, timeout=120) as response:
            payload = response.read()
    except Exception as error:
        raise FileNotFoundError(
            f"Could not download the official processed credit dataset from {url}. "
            f"Place it at {path} or run tools/download_classification_data.py "
            "from a machine with internet access."
        ) from error
    path.write_bytes(payload)
    return path


def load_classification_dataset(
    project_root: Path,
    spec: ClassificationSpec,
) -> ClassificationDataset:
    path = ensure_classification_data(project_root, spec)
    frame = pd.read_csv(path)
    missing = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Unexpected credit dataset schema; missing columns: {missing}")

    
    
    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    labels = frame["NoDefaultNextMonth"].to_numpy(dtype=float)
    labels = np.where(labels > 0.0, 1.0, 0.0)
    features_frame = frame.drop(columns=["NoDefaultNextMonth", *DROPPED_DEMOGRAPHICS])
    features = _standardize_features(features_frame.to_numpy(dtype=float))

    rng = np.random.RandomState(int(spec.dataset_seed))
    first_order = rng.permutation(len(labels))
    labels = labels[first_order]
    features = features[first_order]

    negative = np.flatnonzero(labels == 0.0)
    positive = np.flatnonzero(labels == 1.0)
    per_class = min(len(negative), len(positive))
    required = int(spec.train_size) + int(spec.test_size)
    if 2 * per_class < required:
        raise ValueError(
            f"Balanced dataset has only {2 * per_class} rows, but {required} are required"
        )
    selected = np.concatenate((negative[:per_class], positive[:per_class]))
    selected = selected[rng.permutation(len(selected))][:required]
    features = features[selected]
    labels = labels[selected]

    test_size = int(spec.test_size)
    test_x = np.asarray(features[:test_size], dtype=float)
    test_y = np.asarray(labels[:test_size], dtype=float)
    train_x = np.asarray(features[test_size:], dtype=float)
    train_y = np.asarray(labels[test_size:], dtype=float)
    return ClassificationDataset(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        feature_names=tuple(features_frame.columns),
    )


class StrategicClassificationProblem:
    def __init__(
        self,
        dataset: ClassificationDataset,
        tau: float,
        spec: ClassificationSpec,
    ) -> None:
        self.dataset = dataset
        self.tau = float(tau)
        self.spec = spec
        self.feature_dimension = int(dataset.train_x.shape[1])
        self.n = self.feature_dimension + 1
        if self.feature_dimension != 11:
            raise ValueError(
                f"The Hikima experiment expects 11 features; got {self.feature_dimension}"
            )
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

    @property
    def initial_x(self) -> np.ndarray:
        return np.full(self.n, float(self.spec.initial_value), dtype=float)

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=float)
        positive = values >= 0.0
        output = np.empty_like(values)
        output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        output[~positive] = exp_values / (1.0 + exp_values)
        return output

    def react(self, true_features: np.ndarray, x: np.ndarray) -> np.ndarray:
        values = np.asarray(true_features, dtype=float)
        decision = np.asarray(x, dtype=float).reshape(self.n)
        weights = decision[:-1]
        intercept = float(decision[-1])
        norm_sq = float(np.dot(weights, weights))
        if norm_sq <= 1.0e-16:
            return values.copy()

        margin = values @ weights + intercept
        
        
        movement_cost = np.square(np.minimum(margin, 0.0)) / norm_sq
        should_react = (margin < 0.0) & (
            movement_cost < float(self.spec.reward) / self.tau
        )
        scale = np.where(should_react, -margin / norm_sq, 0.0)
        return values + scale[:, None] * weights[None, :]

    def _sample_true_rows(
        self,
        count: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.randint(0, len(self.dataset.train_y), size=int(count))
        return self.dataset.train_x[indices], self.dataset.train_y[indices]

    def sample_observations(
        self,
        x: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> dict[str, np.ndarray]:
        true_features, labels = self._sample_true_rows(count, rng)
        reacted = self.react(true_features, x)
        return {
            "features": reacted,
            "labels": np.asarray(labels, dtype=float),
        }

    
    def sample_demands(
        self,
        x: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> dict[str, np.ndarray]:
        return self.sample_observations(x, count, rng)

    def loss_from_observations(
        self,
        x: np.ndarray,
        observations: dict[str, np.ndarray],
    ) -> np.ndarray:
        decision = np.asarray(x, dtype=float).reshape(self.n)
        features = np.asarray(observations["features"], dtype=float)
        labels = np.asarray(observations["labels"], dtype=float)
        logits = features @ decision[:-1] + decision[-1]
        
        return np.logaddexp(0.0, logits) - labels * logits

    def sample_losses(
        self,
        x: np.ndarray,
        count: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        observations = self.sample_observations(x, count, rng)
        return self.loss_from_observations(x, observations), observations

    def partial_gradients(
        self,
        x: np.ndarray,
        observations: dict[str, np.ndarray],
    ) -> np.ndarray:
        decision = np.asarray(x, dtype=float).reshape(self.n)
        features = np.asarray(observations["features"], dtype=float)
        labels = np.asarray(observations["labels"], dtype=float)
        probabilities = self._sigmoid(features @ decision[:-1] + decision[-1])
        residual = probabilities - labels
        return np.column_stack((features, np.ones(len(features)))) * residual[:, None]

    def _split_metrics(
        self,
        x: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        decision = np.asarray(x, dtype=float).reshape(self.n)
        reacted = self.react(features, decision)
        logits = reacted @ decision[:-1] + decision[-1]
        loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
        probabilities = self._sigmoid(logits)
        predictions = (logits >= 0.0).astype(float)
        return loss, probabilities, predictions

    def evaluate(
        self,
        x: np.ndarray,
        count: int | None = None,
        rng: np.random.RandomState | None = None,
    ) -> float:
        loss, _, _ = self._split_metrics(
            x,
            self.dataset.train_x,
            self.dataset.train_y,
        )
        return loss

    def metrics(self, x: np.ndarray) -> dict[str, float]:
        train_loss, _, train_predictions = self._split_metrics(
            x,
            self.dataset.train_x,
            self.dataset.train_y,
        )
        test_loss, probabilities, test_predictions = self._split_metrics(
            x,
            self.dataset.test_x,
            self.dataset.test_y,
        )
        return {
            "train_loss": train_loss,
            "train_accuracy": _binary_accuracy(
                self.dataset.train_y,
                train_predictions,
            ),
            "test_loss": test_loss,
            "test_auc": _binary_roc_auc(self.dataset.test_y, probabilities),
            "test_accuracy": _binary_accuracy(
                self.dataset.test_y,
                test_predictions,
            ),
        }
