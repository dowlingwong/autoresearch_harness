from __future__ import annotations

import numpy as np


RANDOM_SEED = 123
N_TRAIN = 1200
N_VAL = 400
N_FEATURES = 12
NOISE_STD = 0.18


def run_mlp_benchmark(
    *,
    learning_rate: float,
    hidden_dim: int,
    regularization: float,
    n_epochs: int,
    batch_size: int,
) -> tuple[float, float]:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    x_train, y_train, x_val, y_val = _make_dataset()
    y_train_col = y_train.reshape(-1, 1)
    rng = np.random.default_rng(RANDOM_SEED + 1)

    w1 = rng.normal(0.0, np.sqrt(2.0 / x_train.shape[1]), size=(x_train.shape[1], hidden_dim))
    b1 = np.zeros((1, hidden_dim), dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1))
    b2 = np.zeros((1, 1), dtype=np.float64)

    last_loss = 0.0
    for epoch in range(n_epochs):
        order = rng.permutation(x_train.shape[0])
        for start in range(0, x_train.shape[0], batch_size):
            batch = order[start : start + batch_size]
            xb = x_train[batch]
            yb = y_train_col[batch]

            z1 = xb @ w1 + b1
            hidden = np.tanh(z1)
            logits = hidden @ w2 + b2
            pred = _sigmoid(logits)

            scale = 1.0 / xb.shape[0]
            dlogits = (pred - yb) * scale
            dw2 = hidden.T @ dlogits + regularization * w2
            db2 = np.sum(dlogits, axis=0, keepdims=True)
            dhidden = dlogits @ w2.T
            dz1 = dhidden * (1.0 - hidden**2)
            dw1 = xb.T @ dz1 + regularization * w1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2

        if epoch == n_epochs - 1:
            hidden_train = np.tanh(x_train @ w1 + b1)
            train_pred = _sigmoid(hidden_train @ w2 + b2)
            l2 = 0.5 * regularization * (float(np.sum(w1 * w1)) + float(np.sum(w2 * w2)))
            last_loss = _binary_cross_entropy(train_pred, y_train_col) + l2

    hidden_val = np.tanh(x_val @ w1 + b1)
    val_pred = _sigmoid(hidden_val @ w2 + b2).reshape(-1)
    val_score = _roc_auc(y_val, val_pred)
    return val_score, float(last_loss)


def _make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_VAL
    n_a = n_total // 2
    n_b = n_total - n_a

    theta_a = rng.uniform(0.0, np.pi, size=n_a)
    theta_b = rng.uniform(0.0, np.pi, size=n_b)
    x_a = np.column_stack([np.cos(theta_a), np.sin(theta_a)])
    x_b = np.column_stack([1.0 - np.cos(theta_b), 0.45 - np.sin(theta_b)])
    x = np.vstack([x_a, x_b])
    x += rng.normal(0.0, NOISE_STD, size=x.shape)
    y = np.concatenate([np.zeros(n_a, dtype=np.float64), np.ones(n_b, dtype=np.float64)])

    features = [
        x[:, 0],
        x[:, 1],
        x[:, 0] * x[:, 1],
        x[:, 0] ** 2,
        x[:, 1] ** 2,
        np.sin(np.pi * x[:, 0]),
        np.cos(np.pi * x[:, 1]),
    ]
    while len(features) < N_FEATURES:
        features.append(rng.normal(0.0, 1.0, size=n_total))
    x_full = np.column_stack(features[:N_FEATURES])

    order = rng.permutation(n_total)
    x_full = x_full[order]
    y = y[order]
    x_train = x_full[:N_TRAIN]
    y_train = y[:N_TRAIN]
    x_val = x_full[N_TRAIN:]
    y_val = y[N_TRAIN:]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return x_train, y_train, x_val, y_val


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.clip(pred, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred)))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    positives = float(np.sum(y_true == 1.0))
    negatives = float(np.sum(y_true == 0.0))
    if positives == 0.0 or negatives == 0.0:
        return 0.5

    tps = np.cumsum(y_true == 1.0)
    fps = np.cumsum(y_true == 0.0)
    tpr = np.concatenate([[0.0], tps / positives])
    fpr = np.concatenate([[0.0], fps / negatives])
    return float(np.trapezoid(tpr, fpr))
