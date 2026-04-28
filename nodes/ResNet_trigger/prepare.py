"""
Read-only data preparation and runtime utilities for the ResNet trigger node.

This file mirrors the role of `prepare.py` in autoresearch-macos:
- fixed dataset configuration
- deterministic dataset splitting
- reusable runtime helpers for `train.py`

The worker should not edit this file during autoresearch loops.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_FILE = "signal_vacuum_sum_crop_4000x8000.h5"
NOISE_FILE = "noise_traces_4000x8000.h5"


@dataclass(frozen=True)
class DataConfig:
    signal_file: str = SIGNAL_FILE
    noise_file: str = NOISE_FILE
    n_signal: int = 4000
    n_noise: int = 4000
    trace_len: int = 8000
    train_frac: float = 0.70
    val_frac: float = 0.15
    eps: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SOURCE_CACHE: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}


def resolve_device() -> torch.device:
    requested = os.environ.get("RESNET_TRIGGER_DEVICE", "").strip().lower()
    if requested:
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("RESNET_TRIGGER_DEVICE=cuda requested, but CUDA is unavailable")
            return torch.device("cuda")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("RESNET_TRIGGER_DEVICE=mps requested, but MPS is unavailable")
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        raise RuntimeError(
            "RESNET_TRIGGER_DEVICE must be one of: cpu, cuda, mps"
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_h5_traces(path: Path) -> tuple[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing H5 file: {path}")
    with h5py.File(path, "r") as handle:
        key = "traces" if "traces" in handle else list(handle.keys())[0]
        array = np.asarray(handle[key][:], dtype=np.float32)
    return key, array


def load_source_arrays(cfg: DataConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = cfg or DataConfig()
    cache_key = (config.signal_file, config.noise_file, int(config.trace_len))
    if cache_key in _SOURCE_CACHE:
        return _SOURCE_CACHE[cache_key]

    signal_path = ROOT / config.signal_file
    noise_path = ROOT / config.noise_file

    signal_key, signal_all = read_h5_traces(signal_path)
    noise_key, noise_all = read_h5_traces(noise_path)

    target_len = int(config.trace_len)
    if signal_all.shape[1] < target_len or noise_all.shape[1] < target_len:
        raise ValueError(
            f"Requested trace_len={target_len}, but got signal_len={signal_all.shape[1]}, "
            f"noise_len={noise_all.shape[1]}"
        )

    signal_all = signal_all[:, :target_len]
    noise_all = noise_all[:, :target_len]

    summary = {
        "signal_file": str(signal_path),
        "noise_file": str(noise_path),
        "signal_key": signal_key,
        "noise_key": noise_key,
        "signal_shape": list(signal_all.shape),
        "noise_shape": list(noise_all.shape),
        "trace_len": target_len,
    }
    _SOURCE_CACHE[cache_key] = (signal_all, noise_all, summary)
    return _SOURCE_CACHE[cache_key]


def split_indices(
    rng: np.random.Generator,
    n: int,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    order = rng.permutation(n)
    return (
        order[:n_train],
        order[n_train:n_train + n_val],
        order[n_train + n_val:n_train + n_val + n_test],
    )


def zscore_per_trace(x: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def make_split(
    signal_all: np.ndarray,
    noise_all: np.ndarray,
    signal_idx: np.ndarray,
    noise_idx: np.ndarray,
    sig_ids: np.ndarray,
    noi_ids: np.ndarray,
    rng: np.random.Generator,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = signal_all[signal_idx[sig_ids]]
    xn = noise_all[noise_idx[noi_ids]]
    ys = np.ones((xs.shape[0],), dtype=np.float32)
    yn = np.zeros((xn.shape[0],), dtype=np.float32)
    x = np.concatenate([xs, xn], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([ys, yn], axis=0)
    order = rng.permutation(x.shape[0])
    x = zscore_per_trace(x[order], eps=eps)
    y = y[order]
    return x[:, None, :], y


def prepare_run_arrays(
    cfg: DataConfig | None = None,
    run_seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    config = cfg or DataConfig()
    signal_all, noise_all, source_summary = load_source_arrays(config)
    if signal_all.shape[0] < config.n_signal or noise_all.shape[0] < config.n_noise:
        raise ValueError(
            f"Requested n_signal={config.n_signal}, n_noise={config.n_noise}, "
            f"but source sizes are signal={signal_all.shape[0]}, noise={noise_all.shape[0]}"
        )

    rng = np.random.default_rng(run_seed)
    signal_idx = rng.permutation(signal_all.shape[0])[:config.n_signal]
    noise_idx = rng.permutation(noise_all.shape[0])[:config.n_noise]

    sig_train, sig_val, sig_test = split_indices(rng, len(signal_idx), config.train_frac, config.val_frac)
    noi_train, noi_val, noi_test = split_indices(rng, len(noise_idx), config.train_frac, config.val_frac)

    x_train, y_train = make_split(signal_all, noise_all, signal_idx, noise_idx, sig_train, noi_train, rng, config.eps)
    x_val, y_val = make_split(signal_all, noise_all, signal_idx, noise_idx, sig_val, noi_val, rng, config.eps)
    x_test, y_test = make_split(signal_all, noise_all, signal_idx, noise_idx, sig_test, noi_test, rng, config.eps)

    split_meta = {
        "seed": run_seed,
        "config": config.to_dict(),
        "source_summary": source_summary,
        "signal_indices": signal_idx.tolist(),
        "noise_indices": noise_idx.tolist(),
        "signal_split": {"train": sig_train.tolist(), "val": sig_val.tolist(), "test": sig_test.tolist()},
        "noise_split": {"train": noi_train.tolist(), "val": noi_val.tolist(), "test": noi_test.tolist()},
    }
    return x_train, y_train, x_val, y_val, x_test, y_test, split_meta


def objective_from_val_auc(val_auc: float) -> float:
    """Compatibility objective for the current autoresearch harness.

    The harness expects a lower-is-better scalar named `val_bpb`. For this
    classification node we use:

        val_bpb = 1 - best_val_auc

    The real task metric remains ROC AUC, but this alias keeps the
    current manager/control-plane machinery usable without modification.
    """

    return 1.0 - float(val_auc)


def peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    if device.type == "mps":
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current_allocated):
            return float(current_allocated() / (1024 ** 2))
    return 0.0


def save_json_artifact(name: str, payload: dict[str, Any]) -> Path:
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    config = DataConfig()
    _, _, _, _, _, _, split_meta = prepare_run_arrays(config, run_seed=123)
    summary = {
        "device": str(resolve_device()),
        "config": config.to_dict(),
        "split_seed": split_meta["seed"],
        "source_summary": split_meta["source_summary"],
    }
    print(json.dumps(summary, indent=2))
