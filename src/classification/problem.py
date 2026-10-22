from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "credit_processed.csv"
FEATURES = ["EducationLevel", "MaxBillAmountOverLast6Months", "MaxPaymentAmountOverLast6Months",
            "MonthsWithZeroBalanceOverLast6Months", "MonthsWithLowSpendingOverLast6Months",
            "MonthsWithHighSpendingOverLast6Months", "MostRecentBillAmount", "MostRecentPaymentAmount",
            "TotalOverdueCounts", "TotalMonthsOverdue", "HistoryOfOverduePayments"]


def load_dataset(spec):
    frame = pd.read_csv(DATA_FILE)
    labels = np.where(frame["NoDefaultNextMonth"].to_numpy(dtype=float) > 0.0, 1.0, 0.0)
    features = frame[FEATURES].to_numpy(dtype=float)
    scales = features.std(axis=0, ddof=0)
    features = (features - features.mean(axis=0)) / np.where(scales > 0.0, scales, 1.0)
    rng = np.random.RandomState(int(spec["dataset_seed"]))
    order = rng.permutation(len(labels))
    labels, features = labels[order], features[order]
    negative, positive = np.flatnonzero(labels == 0.0), np.flatnonzero(labels == 1.0)
    per_class = min(len(negative), len(positive))
    selected = np.concatenate((negative[:per_class], positive[:per_class]))
    selected = selected[rng.permutation(len(selected))][: int(spec["train_size"]) + int(spec["test_size"])]
    features, labels = features[selected], labels[selected]
    test = int(spec["test_size"])
    return {"train_x": features[test:], "train_y": labels[test:], "test_x": features[:test], "test_y": labels[:test]}


def sigmoid(logits):
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def roc_auc(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positives, negatives = int(np.sum(labels == 1.0)), int(np.sum(labels == 0.0))
    return float((ranks[labels == 1.0].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


class StrategicClassification:
    def __init__(self, dataset, tau, spec):
        self.data, self.tau, self.spec = dataset, float(tau), spec
        self.n = dataset["train_x"].shape[1] + 1

    @property
    def initial_x(self):
        return np.full(self.n, float(self.spec["initial_value"]), dtype=float)

    def react(self, features, x):
        weights, intercept = x[:-1], float(x[-1])
        norm_sq = float(np.dot(weights, weights))
        if norm_sq <= 1.0e-16:
            return features.copy()
        margin = features @ weights + intercept
        cost = np.square(np.minimum(margin, 0.0)) / norm_sq
        moves = (margin < 0.0) & (cost < float(self.spec["reward"]) / self.tau)
        return features + np.where(moves, -margin / norm_sq, 0.0)[:, None] * weights[None, :]

    def sample_observations(self, x, count, rng):
        indices = rng.randint(0, len(self.data["train_y"]), size=int(count))
        return {"features": self.react(self.data["train_x"][indices], np.asarray(x, dtype=float)),
                "labels": self.data["train_y"][indices]}

    def loss_from_observations(self, x, observations):
        logits = observations["features"] @ x[:-1] + x[-1]
        return np.logaddexp(0.0, logits) - observations["labels"] * logits

    def sample_losses(self, x, count, rng):
        observations = self.sample_observations(x, count, rng)
        return self.loss_from_observations(x, observations), observations

    def known_gradient(self, x, observations):
        features = observations["features"]
        residual = sigmoid(features @ x[:-1] + x[-1]) - observations["labels"]
        return (np.column_stack((features, np.ones(len(features)))) * residual[:, None]).mean(axis=0)

    def _split(self, x, features, labels):
        logits = self.react(features, x) @ x[:-1] + x[-1]
        return float(np.mean(np.logaddexp(0.0, logits) - labels * logits)), logits

    def evaluate(self, x, count=None, rng=None):
        return self._split(x, self.data["train_x"], self.data["train_y"])[0]

    def metrics(self, x):
        train_loss, train_logits = self._split(x, self.data["train_x"], self.data["train_y"])
        test_loss, test_logits = self._split(x, self.data["test_x"], self.data["test_y"])
        return {"train_loss": train_loss,
                "train_accuracy": float(np.mean(self.data["train_y"] == (train_logits >= 0.0))),
                "test_loss": test_loss,
                "test_auc": roc_auc(self.data["test_y"], sigmoid(test_logits)),
                "test_accuracy": float(np.mean(self.data["test_y"] == (test_logits >= 0.0)))}
