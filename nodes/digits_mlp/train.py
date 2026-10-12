#!/usr/bin/env python3
"""digits_mlp: editable hyperparameters for a frozen sklearn MLP benchmark (R1 L8b).

Only the constants below are editable. Output:
    val_score: 0.XXXXXX   (validation accuracy, higher is better)
    test_score: 0.XXXXXX  (held-out test split; informational, never used to keep)
"""
from benchmark import run_benchmark

# ── editable hyperparameters ────────────────────────────────────────────────
HIDDEN_UNITS = 32
ALPHA = 1e-4
LEARNING_RATE_INIT = 1e-3
BATCH_SIZE = 64
MAX_ITER = 40


def main() -> None:
    result = run_benchmark(
        hidden_units=HIDDEN_UNITS,
        alpha=ALPHA,
        learning_rate_init=LEARNING_RATE_INIT,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
    )
    print(f"val_score: {result['val_score']:.6f}")
    print(f"test_score: {result['test_score']:.6f}")
    print(f"train_loss: {result['train_loss']:.6f}")


if __name__ == "__main__":
    main()
