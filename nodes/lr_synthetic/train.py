#!/usr/bin/env python3
"""Synthetic logistic regression node for autoresearch_harness governance demonstration.

This node provides a second ML task so the governance protocol can be validated
across more than one experiment surface. The task is binary classification on
synthetic 2-D Gaussian data; training uses mini-batch gradient descent with
logistic loss implemented in pure NumPy (no scikit-learn dependency).

Editable hyperparameters (modified by the harness during ablations):
    LEARNING_RATE   -- SGD step size
    REGULARIZATION  -- L2 penalty coefficient
    N_EPOCHS        -- number of SGD passes over the training set
    BATCH_SIZE      -- SGD mini-batch size

Fixed settings are not editable and must not appear in a governance proposal.
Output format:
    val_score: 0.XXXXXX   (validation AUC, higher is better)
    train_loss: 0.XXXXXX  (final training loss, informational)
"""
import math
import random

import numpy as np

# ── editable hyperparameters ────────────────────────────────────────────────
LEARNING_RATE = 3e-4
REGULARIZATION = 0.05
N_EPOCHS = 80
BATCH_SIZE = 32

# ── fixed settings ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_TRAIN = 1600
N_VAL = 400
N_FEATURES = 10
N_INFORMATIVE = 4


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(n_samples: int, n_features: int, n_informative: int, seed: int):
    """Generate a synthetic binary classification dataset (pure NumPy)."""
    rng = np.random.default_rng(seed)
    # informative features drawn from two shifted Gaussians
    X_inf_pos = rng.normal(loc=0.5, scale=1.0, size=(n_samples // 2, n_informative))
    X_inf_neg = rng.normal(loc=-0.5, scale=1.0, size=(n_samples - n_samples // 2, n_informative))
    X_informative = np.vstack([X_inf_pos, X_inf_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples - n_samples // 2))
    # noise features
    X_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features - n_informative))
    X = np.hstack([X_informative, X_noise])
    # shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def _standardize(X_train, X_val):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression (NumPy mini-batch SGD)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _bce_loss(y_true, y_pred, weights, reg):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    l2 = 0.5 * reg * np.sum(weights ** 2)
    return bce + l2


def _roc_auc(y_true, y_score):
    """Compute ROC-AUC using the trapezoidal rule (pure NumPy)."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    auc = float(np.trapezoid(tpr, fpr))
    return max(auc, 1.0 - auc)  # always return the meaningful side


def train(X_train, y_train, X_val, y_val, lr, reg, n_epochs, batch_size, seed):
    rng = np.random.default_rng(seed + 1)
    n, d = X_train.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    final_loss = float("inf")
    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]
        for start in range(0, n, batch_size):
            Xb = X_shuf[start: start + batch_size]
            yb = y_shuf[start: start + batch_size]
            pred = _sigmoid(Xb @ w + b)
            err = pred - yb
            grad_w = Xb.T @ err / len(yb) + reg * w
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b
        preds_train = _sigmoid(X_train @ w + b)
        final_loss = _bce_loss(y_train, preds_train, w, reg)
    val_score_pred = _sigmoid(X_val @ w + b)
    val_auc = _roc_auc(y_val, val_score_pred)
    return val_auc, final_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng_state = RANDOM_SEED
    X_all, y_all = _make_dataset(
        n_samples=N_TRAIN + N_VAL,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        seed=rng_state,
    )
    X_train_raw, X_val_raw = X_all[:N_TRAIN], X_all[N_TRAIN:]
    y_train, y_val = y_all[:N_TRAIN], y_all[N_TRAIN:]
    X_train, X_val = _standardize(X_train_raw, X_val_raw)

    val_score, train_loss = train(
        X_train, y_train, X_val, y_val,
        lr=LEARNING_RATE,
        reg=REGULARIZATION,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
    )

    print(f"val_score: {val_score:.6f}")
    print(f"train_loss: {train_loss:.6f}")
    print(f"LEARNING_RATE={LEARNING_RATE}  REGULARIZATION={REGULARIZATION}")


if __name__ == "__main__":
    main()
