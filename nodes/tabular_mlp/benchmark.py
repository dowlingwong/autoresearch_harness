"""Frozen benchmark for the tabular_mlp node (R1 L8b, noisy CPU node).

Noisy CPU node: sklearn MLPClassifier on a fixed synthetic tabular pool (make_classification, 2,000 rows, 20 features, 6 informative, 8% label noise, generator seed 0). The metric is validation ROC AUC.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

TRAIN_SEED = int(os.environ.get("AR_TRAIN_SEED", "42"))
SPLIT_SEED = int(os.environ.get("AR_SPLIT_SEED", "42"))
TRAIN_FRACTION = 0.60  # the rest is split evenly into val and test

POOL_SEED = 0  # the data pool is fixed; AR_SPLIT_SEED only re-splits it


def load_pool() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_classification
    return make_classification(n_samples=2000, n_features=20, n_informative=6, n_redundant=4,
                               flip_y=0.08, class_sep=0.8, random_state=POOL_SEED)


def score(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, model.predict_proba(x)[:, 1]))


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
