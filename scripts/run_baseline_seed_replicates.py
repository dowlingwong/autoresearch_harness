#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.nodes.resnet_trigger.metric_parser import MetricParseError, parse_val_auc


FAST_ENV_DEFAULTS = {
    "RESNET_TRIGGER_FAST_SEARCH": "1",
    "RESNET_TRIGGER_FAST_N_SIGNAL": "1000",
    "RESNET_TRIGGER_FAST_N_NOISE": "1000",
    "RESNET_TRIGGER_FAST_TRACE_LEN": "4096",
    "RESNET_TRIGGER_FAST_BATCH_SIZE": "64",
    "RESNET_TRIGGER_FAST_EPOCHS": "3",
    "RESNET_TRIGGER_FAST_SKIP_TEST": "1",
    "RESNET_TRIGGER_EARLY_STOP_PATIENCE": "2",
    "RESNET_TRIGGER_EARLY_STOP_MIN_DELTA": "0.002",
    "RESNET_TRIGGER_DEVICE": "cpu",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clean baseline training across multiple seeds.")
    parser.add_argument("--node-root", default=str(ROOT / "nodes" / "ResNet_trigger"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125, 126, 127])
    parser.add_argument(
        "--artifacts-dir",
        default=str(ROOT / "experiments" / "artifacts" / "p8_baseline_seed_replicates"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper" / "tables" / "baseline_seed_replicates.csv"),
    )
    parser.add_argument(
        "--ci-output",
        default=str(ROOT / "paper" / "tables" / "baseline_seed_bootstrap_ci.csv"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    node_root = Path(args.node_root).resolve()
    train_path = node_root / "train.py"
    original = train_path.read_text(encoding="utf-8")
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            seed_dir = artifacts_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            _clear_node_runtime_state(node_root)
            _write_seed(train_path, original, seed)
            log_path = seed_dir / "run.log"
            env = dict(FAST_ENV_DEFAULTS)
            completed = _run_train(node_root=node_root, log_path=log_path, env_overrides=env)
            row: dict[str, object] = {
                "seed": seed,
                "returncode": completed.returncode,
                "log_ref": str(log_path),
                "val_auc": "",
                "status": "failed" if completed.returncode else "success",
            }
            if completed.returncode == 0:
                try:
                    parsed = parse_val_auc(log_path)
                    row["val_auc"] = parsed.metric_value
                except MetricParseError as exc:
                    row["status"] = "metric_missing"
                    row["error"] = str(exc)
            rows.append(row)
            print(f"seed={seed} status={row['status']} val_auc={row['val_auc']}")
    finally:
        train_path.write_text(original, encoding="utf-8")
        _clear_node_runtime_state(node_root)

    _write_rows(Path(args.output), rows)
    ci = _bootstrap_ci(
        [float(row["val_auc"]) for row in rows if row.get("val_auc") not in ("", None)],
        samples=args.bootstrap_samples,
    )
    _write_rows(Path(args.ci_output), [ci])
    print(f"wrote={args.output}")
    print(f"wrote={args.ci_output}")
    return 0 if all(row["status"] == "success" for row in rows) else 1


def _write_seed(train_path: Path, original: str, seed: int) -> None:
    updated, count = re.subn(r"(?m)^SEED\s*=\s*[0-9]+", f"SEED = {seed}", original, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace SEED assignment in {train_path}")
    train_path.write_text(updated, encoding="utf-8")


def _run_train(*, node_root: Path, log_path: Path, env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(
            ["uv", "run", "train.py"],
            cwd=node_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )


def _clear_node_runtime_state(node_root: Path) -> None:
    for rel in ("run.log", "results.tsv", "experiment_memory.jsonl"):
        path = node_root / rel
        if path.exists() and path.is_file():
            path.unlink()
    artifact_dir = node_root / "artifacts"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def _bootstrap_ci(values: list[float], *, samples: int) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean_val_auc": "",
            "std_val_auc": "",
            "bootstrap_samples": samples,
            "mean_ci_low": "",
            "mean_ci_high": "",
        }
    rng = random.Random(12345)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        means.append(statistics.fmean(draw))
    means.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return {
        "n": len(values),
        "mean_val_auc": statistics.fmean(values),
        "std_val_auc": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_val_auc": min(values),
        "max_val_auc": max(values),
        "bootstrap_samples": samples,
        "mean_ci_low": means[low_idx],
        "mean_ci_high": means[high_idx],
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
