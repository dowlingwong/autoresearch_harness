"""Fake Track G node for tests: a CPU script that imitates a fixed-step training run."""
from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

TRAIN_PY = textwrap.dedent('''\
    """Fake fixed-step trainer; MODE selects the behaviour under test."""
    import math
    import os
    import sys
    import time

    MODE = "ok"
    LEARNING_RATE = 0.1


    def main():
        steps = int(os.environ.get("AR_BUDGET_VALUE", "5"))
        if MODE == "sleep":
            time.sleep(60)
        if MODE == "affinity":
            print("affinity:", sorted(os.sched_getaffinity(0)))
            import resource
            print("rlimit_data:", resource.getrlimit(resource.RLIMIT_DATA)[0])
        if MODE == "env":
            for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "AR_NUM_WORKERS", "CUBLAS_WORKSPACE_CONFIG",
                        "AR_TRAIN_SEED", "AR_BUDGET_KIND"):
                print(f"env {key}={os.environ.get(key)!r}")
        for step in range(1, steps + 1):
            loss = float("nan") if (MODE == "nan" and step == 3) else 1.0 / (step * LEARNING_RATE * 10)
            print(f"step {step} loss: {loss}")
            if MODE == "crash" and step == 2:
                sys.exit(1)
            if MODE == "short" and step == 2:
                break
        if MODE != "nometric":
            print(f"val_accuracy: {0.5 + LEARNING_RATE:.4f}")
            print("test_accuracy: 0.4242")


    if __name__ == "__main__":
        main()
    ''')

EVALUATE_PY = "def evaluate(preds, labels):\n    return sum(p == l for p, l in zip(preds, labels)) / len(labels)\n"


def spec_mapping(**overrides) -> dict:
    payload = {
        "schema": "trackg_node/v1",
        "name": "fake_task",
        "description": "fake fixed-step node for tests",
        "suite": "test",
        "editable_paths": ["train.py", "model/"],
        "frozen_paths": ["evaluate.py", "data/", "model/metric.py"],
        "run_command": "python3 train.py",
        "metric_name": "val_accuracy",
        "metric_direction": "maximize",
        "metric": {
            "metric_regex": r"^val_accuracy:\s*(\S+)",
            "loss_regex": r"^step\s+\d+\s+loss:\s*(\S+)",
            "steps_regex": r"^step\s+(\d+)\b",
            "test_metric_regex": r"^test_accuracy:\s*(\S+)",
        },
        "run_budget": {"kind": "steps", "value": 5},
        "workers": {"dataloader_workers": 2, "threads": 1},
        "selection": {"test_scored": True, "selection_split": "validation", "test_metric_name": "test_accuracy"},
    }
    payload.update(overrides)
    return payload


def make_node(root: Path, mode: str = "ok") -> Path:
    node = root / "node"
    (node / "data").mkdir(parents=True)
    (node / "model").mkdir()
    (node / "train.py").write_text(TRAIN_PY.replace('MODE = "ok"', f'MODE = "{mode}"'), encoding="utf-8")
    (node / "evaluate.py").write_text(EVALUATE_PY, encoding="utf-8")
    (node / "data" / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    (node / "model" / "net.py").write_text("WIDTH = 64\n# width of the hidden layer\nDEPTH = 2\n", encoding="utf-8")
    (node / "model" / "metric.py").write_text("NAME = 'acc'\n", encoding="utf-8")
    (node / "config.yaml").write_text("lr: 0.1\n", encoding="utf-8")
    return node


def write_spec(path: Path, **overrides) -> Path:
    path.write_text(yaml.safe_dump(spec_mapping(**overrides)), encoding="utf-8")
    return path
