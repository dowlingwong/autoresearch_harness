"""Frozen benchmark for the digits_mlp node (R1 L8b, noisy CPU node).

Noisy CPU node: sklearn MLPClassifier on the scikit-learn digits data (1,797 images, 64 features, 10 classes; bundled with scikit-learn, no download). The metric is validation accuracy.

Seeds (injected by the worker; never editable):
- AR_SPLIT_SEED (default 42) fixes the train/val/test split of a fixed data pool;
- AR_TRAIN_SEED (default 42) fixes the MLP weight initialisation and the
  mini-batch order.
They are independent, so the two noise sources can be measured separately.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

TRAIN_SEED = int(os.environ.get("AR_TRAIN_SEED", "42"))
SPLIT_SEED = int(os.environ.get("AR_SPLIT_SEED", "42"))
TRAIN_FRACTION = 0.60  # the rest is split evenly into val and test

def load_pool() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits
    return load_digits(return_X_y=True)


def score(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> float:
    return float(model.score(x, y))


def run_benchmark(*, hidden_units: int, alpha: float, learning_rate_init: float,
                  batch_size: int, max_iter: int) -> dict[str, float]:
    if hidden_units <= 0 or batch_size <= 0 or max_iter <= 0:
        raise ValueError("hidden_units, batch_size and max_iter must be positive")
    if alpha < 0 or learning_rate_init <= 0:
        raise ValueError("alpha must be >= 0 and learning_rate_init > 0")
    x, y = load_pool()
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, train_size=TRAIN_FRACTION, random_state=SPLIT_SEED, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, train_size=0.5, random_state=SPLIT_SEED, stratify=y_rest)
    scaler = StandardScaler().fit(x_train)
    model = MLPClassifier(hidden_layer_sizes=(int(hidden_units),), alpha=float(alpha),
                          learning_rate_init=float(learning_rate_init),
                          batch_size=int(batch_size), max_iter=int(max_iter),
                          random_state=TRAIN_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(scaler.transform(x_train), y_train)
    return {
        "val_score": score(model, scaler.transform(x_val), y_val),
        "test_score": score(model, scaler.transform(x_test), y_test),
        "train_loss": float(model.loss_),
    }
