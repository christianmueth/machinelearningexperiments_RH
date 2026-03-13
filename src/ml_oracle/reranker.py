from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairwiseMLPReranker:
    input_dim: int
    hidden_dim: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(int(self.seed))
        scale1 = 1.0 / max(1, int(self.input_dim))
        scale2 = 1.0 / max(1, int(self.hidden_dim))
        self.W1 = rng.normal(loc=0.0, scale=scale1, size=(int(self.input_dim), int(self.hidden_dim))).astype(np.float64)
        self.b1 = np.zeros((int(self.hidden_dim),), dtype=np.float64)
        self.W2 = rng.normal(loc=0.0, scale=scale2, size=(int(self.hidden_dim), 1)).astype(np.float64)
        self.b2 = np.zeros((1,), dtype=np.float64)

    def score(self, X: np.ndarray) -> np.ndarray:
        hidden = np.tanh(np.asarray(X, dtype=np.float64) @ self.W1 + self.b1)
        return (hidden @ self.W2 + self.b2).reshape(-1).astype(np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        group_ids = np.asarray(group_ids, dtype=np.int64)
        history: list[float] = []
        for _ in range(int(epochs)):
            hidden = np.tanh(X @ self.W1 + self.b1)
            scores = (hidden @ self.W2 + self.b2).reshape(-1)
            grad_scores = np.zeros_like(scores)
            loss_value = 0.0
            pair_count = 0

            for group in np.unique(group_ids):
                idx = np.flatnonzero(group_ids == int(group))
                pos = idx[y[idx] > 0.5]
                neg = idx[y[idx] <= 0.5]
                for i in pos.tolist():
                    for j in neg.tolist():
                        delta = float(scores[i] - scores[j])
                        coeff = -1.0 / (1.0 + np.exp(delta))
                        loss_value += float(np.log1p(np.exp(-delta)))
                        grad_scores[i] += coeff
                        grad_scores[j] -= coeff
                        pair_count += 1

            if pair_count == 0:
                history.append(0.0)
                continue

            grad_scores /= float(pair_count)
            loss_value /= float(pair_count)

            grad_W2 = hidden.T @ grad_scores.reshape(-1, 1) + float(weight_decay) * self.W2
            grad_b2 = np.asarray([np.sum(grad_scores)], dtype=np.float64)
            grad_hidden = grad_scores.reshape(-1, 1) @ self.W2.T
            grad_pre = grad_hidden * (1.0 - hidden * hidden)
            grad_W1 = X.T @ grad_pre + float(weight_decay) * self.W1
            grad_b1 = np.sum(grad_pre, axis=0)

            self.W2 -= float(lr) * grad_W2
            self.b2 -= float(lr) * grad_b2
            self.W1 -= float(lr) * grad_W1
            self.b1 -= float(lr) * grad_b1

            history.append(float(loss_value))
        return history

    def save(self, path: str) -> None:
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            input_dim=np.asarray([self.input_dim], dtype=np.int64),
            hidden_dim=np.asarray([self.hidden_dim], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "PairwiseMLPReranker":
        data = np.load(path)
        model = cls(input_dim=int(data["input_dim"][0]), hidden_dim=int(data["hidden_dim"][0]))
        model.W1 = np.asarray(data["W1"], dtype=np.float64)
        model.b1 = np.asarray(data["b1"], dtype=np.float64)
        model.W2 = np.asarray(data["W2"], dtype=np.float64)
        model.b2 = np.asarray(data["b2"], dtype=np.float64)
        return model


def group_accuracy(scores: np.ndarray, labels: np.ndarray, group_ids: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    correct = 0
    total = 0
    for group in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == int(group))
        if idx.size == 0:
            continue
        pred = idx[int(np.argmax(scores[idx]))]
        gold = idx[int(np.argmax(labels[idx]))]
        correct += int(pred == gold)
        total += 1
    return float(correct) / float(total) if total else 0.0