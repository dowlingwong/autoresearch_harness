#!/usr/bin/env python3
from __future__ import annotations

from benchmark import run_mlp_benchmark


LEARNING_RATE = 0.02
HIDDEN_DIM = 8
REGULARIZATION = 0.001
N_EPOCHS = 80
BATCH_SIZE = 32


def main() -> None:
    val_score, train_loss = run_mlp_benchmark(
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        regularization=REGULARIZATION,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    print(f"val_score: {val_score:.6f}")
    print(f"train_loss: {train_loss:.6f}")
    print(
        "LEARNING_RATE={learning_rate} HIDDEN_DIM={hidden_dim} "
        "REGULARIZATION={regularization} N_EPOCHS={n_epochs} BATCH_SIZE={batch_size}".format(
            learning_rate=LEARNING_RATE,
            hidden_dim=HIDDEN_DIM,
            regularization=REGULARIZATION,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
        )
    )


if __name__ == "__main__":
    main()
