"""
Single-file training entrypoint for the ResNet trigger autoresearch node.

This is the only file the worker should modify in the experiment loop.
The output format intentionally mirrors autoresearch-macos so the existing
control plane can parse it:

---
val_bpb:          <lower is better compatibility objective>
training_seconds: ...
total_seconds:    ...
peak_vram_mb:     ...
...

For this classification task:
    val_bpb = 1 - best_val_auc
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    ARTIFACT_DIR,
    DataConfig,
    objective_from_val_auc,
    peak_vram_mb,
    prepare_run_arrays,
    resolve_device,
    save_json_artifact,
)


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}

SEED = 123
N_SIGNAL = 4000
N_NOISE = 4000
TRACE_LEN = 8000
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
EPS = 1e-6

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

STAGE_LAYERS = [1, 1, 1]
KERNEL_SIZE = 7
DROPOUT = 0.0


DEVICE = resolve_device()
GRAD_CLIP_NORM = 1.0
MPS_MAX_LEARNING_RATE = 5e-4
MPS_ADAM_EPS = 1e-4
LOGIT_CLAMP = 30.0
RUN_ARCHIVE_DIR = ARTIFACT_DIR / "run_archive"
RUN_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Fast-search mode trades evaluation fidelity for much faster screening loops.
FAST_SEARCH_MODE = _env_flag("RESNET_TRIGGER_FAST_SEARCH", False)
FAST_SEARCH_N_SIGNAL = int(os.environ.get("RESNET_TRIGGER_FAST_N_SIGNAL", "1000"))
FAST_SEARCH_N_NOISE = int(os.environ.get("RESNET_TRIGGER_FAST_N_NOISE", "1000"))
FAST_SEARCH_TRACE_LEN = int(os.environ.get("RESNET_TRIGGER_FAST_TRACE_LEN", "4096"))
FAST_SEARCH_BATCH_SIZE = int(os.environ.get("RESNET_TRIGGER_FAST_BATCH_SIZE", "64"))
FAST_SEARCH_EPOCHS = int(os.environ.get("RESNET_TRIGGER_FAST_EPOCHS", "3"))
FAST_SEARCH_SKIP_TEST = _env_flag("RESNET_TRIGGER_FAST_SKIP_TEST", True)
EARLY_STOPPING_PATIENCE = int(os.environ.get("RESNET_TRIGGER_EARLY_STOP_PATIENCE", "2"))
EARLY_STOPPING_MIN_DELTA = float(os.environ.get("RESNET_TRIGGER_EARLY_STOP_MIN_DELTA", "0.002"))


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    data: DataConfig
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    stage_layers: list[int]
    kernel_size: int
    dropout: float
    fast_search_mode: bool
    skip_test_eval: bool
    early_stopping_patience: int
    early_stopping_min_delta: float

    @property
    def depth(self) -> int:
        return int(sum(self.stage_layers))


def build_config() -> TrainConfig:
    n_signal = FAST_SEARCH_N_SIGNAL if FAST_SEARCH_MODE else N_SIGNAL
    n_noise = FAST_SEARCH_N_NOISE if FAST_SEARCH_MODE else N_NOISE
    trace_len = FAST_SEARCH_TRACE_LEN if FAST_SEARCH_MODE else TRACE_LEN
    batch_size = FAST_SEARCH_BATCH_SIZE if FAST_SEARCH_MODE else BATCH_SIZE
    epochs = FAST_SEARCH_EPOCHS if FAST_SEARCH_MODE else EPOCHS
    return TrainConfig(
        seed=SEED,
        data=DataConfig(
            n_signal=n_signal,
            n_noise=n_noise,
            trace_len=trace_len,
            train_frac=TRAIN_FRACTION,
            val_frac=VAL_FRACTION,
            eps=EPS,
        ),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        stage_layers=list(STAGE_LAYERS),
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        fast_search_mode=FAST_SEARCH_MODE,
        skip_test_eval=FAST_SEARCH_SKIP_TEST if FAST_SEARCH_MODE else False,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def load_resnet1d_class() -> type[nn.Module]:
    try:
        from resnet_1d import ResNet1D  # type: ignore
        return ResNet1D
    except Exception:
        code = (Path(__file__).resolve().parent / "resnet_1d.py").read_text(encoding="utf-8")
        replacement = (
            "from typing import Any\n"
            "NormLayer = Any\n\n"
            "def GroupNorm1DGetter(num_groups: int = 8):\n"
            "    def _mk(channels: int):\n"
            "        g = min(num_groups, channels)\n"
            "        while g > 1 and channels % g != 0:\n"
            "            g -= 1\n"
            "        return nn.GroupNorm(g, channels)\n"
            "    return _mk\n"
        )
        code = code.replace("from ..norm import GroupNorm1DGetter, NormLayer", replacement)
        module = types.ModuleType("resnet_1d_local")
        module.__dict__["nn"] = nn
        module.__dict__["torch"] = torch
        exec(code, module.__dict__)
        return module.ResNet1D


class ResNetTrigger1D(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        ResNet1D = load_resnet1d_class()
        norm_layer = nn.BatchNorm1d if DEVICE.type == "mps" else None
        self.backbone = ResNet1D(
            in_channels=1,
            layers=config.stage_layers,
            classes=1,
            kernel_size=config.kernel_size,
            norm_layer=norm_layer,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return self.dropout(logits)


def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True).view(-1, 1)
            logits = model(xb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits detected during evaluation")
            logits_for_loss = logits.float().clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
            loss = criterion(logits_for_loss, yb)
            losses.append(float(loss.item()))
            ys.append(yb.detach().cpu().numpy().ravel())
            probs.append(torch.sigmoid(logits_for_loss).detach().cpu().numpy().ravel())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    return float(np.mean(losses)), y_true, y_prob


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    best_epoch_row: dict[str, float | int],
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "stage_layers": config.stage_layers,
            "kernel_size": config.kernel_size,
            "dropout": config.dropout,
            "fast_search_mode": config.fast_search_mode,
            "skip_test_eval": config.skip_test_eval,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "data": config.data.to_dict(),
        },
        "best_epoch": int(best_epoch_row["epoch"]),
        "best_val_auc": float(best_epoch_row["val_auc"]),
        "best_val_loss": float(best_epoch_row["val_loss"]),
        "best_val_pr_auc": float(best_epoch_row["val_pr_auc"]),
    }


def archive_run_bundle(
    config: TrainConfig,
    split_meta: dict,
    history: list[dict[str, float | int]],
    metrics: dict,
    timing: dict,
) -> Path:
    archive_id = f"run_{int(time.time() * 1000)}"
    archive_path = RUN_ARCHIVE_DIR / f"{archive_id}.json"
    archive_payload = {
        "archive_id": archive_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "stage_layers": config.stage_layers,
            "kernel_size": config.kernel_size,
            "dropout": config.dropout,
            "fast_search_mode": config.fast_search_mode,
            "skip_test_eval": config.skip_test_eval,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "data": config.data.to_dict(),
        },
        "split_meta": split_meta,
        "history": history,
        "metrics": metrics,
        "timing": timing,
    }
    archive_path.write_text(json.dumps(archive_payload, indent=2), encoding="utf-8")
    return archive_path


def main() -> None:
    config = build_config()
    set_seed(config.seed)
    reset_memory_stats(DEVICE)

    prep_t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, x_test, y_test, split_meta = prepare_run_arrays(config.data, run_seed=config.seed)
    dataset_prep_seconds = time.perf_counter() - prep_t0

    train_loader = make_loader(x_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, config.batch_size, shuffle=False)
    test_loader = None if config.skip_test_eval else make_loader(x_test, y_test, config.batch_size, shuffle=False)

    model = ResNetTrigger1D(config).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_lr = min(config.learning_rate, MPS_MAX_LEARNING_RATE) if DEVICE.type == "mps" else config.learning_rate
    optimizer_eps = MPS_ADAM_EPS if DEVICE.type == "mps" else 1e-8
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_lr,
        weight_decay=config.weight_decay,
        eps=optimizer_eps,
    )

    param_count = int(sum(p.numel() for p in model.parameters()))
    history: list[dict[str, float | int]] = []
    total_t0 = time.perf_counter()
    train_time_sum = 0.0
    val_time_sum = 0.0
    total_examples_seen = 0
    best_epoch_row: dict[str, float | int] | None = None
    early_stop_counter = 0
    stopped_early = False
    best_checkpoint_path = ARTIFACT_DIR / "best_model.pt"
    best_metrics_path = ARTIFACT_DIR / "best_performance.json"

    print("device:", DEVICE)
    print(
        json.dumps(
            {
                "seed": config.seed,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "optimizer_learning_rate": optimizer_lr,
                "optimizer_eps": optimizer_eps,
                "weight_decay": config.weight_decay,
                "stage_layers": config.stage_layers,
                "kernel_size": config.kernel_size,
                "dropout": config.dropout,
                "fast_search_mode": config.fast_search_mode,
                "skip_test_eval": config.skip_test_eval,
                "early_stopping_patience": config.early_stopping_patience,
                "early_stopping_min_delta": config.early_stopping_min_delta,
                "n_signal": config.data.n_signal,
                "n_noise": config.data.n_noise,
                "trace_len": config.data.trace_len,
            },
            indent=2,
        )
    )
    print(f"train/val/test: {x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]}")
    print(f"parameter_count: {param_count}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_t0 = time.perf_counter()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, dtype=torch.float32, non_blocking=True)
            yb = yb.to(DEVICE, dtype=torch.float32, non_blocking=True).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits detected during training")
            logits_for_loss = logits.float().clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
            loss = criterion(logits_for_loss, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected during training")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(float(loss.item()))
            total_examples_seen += int(xb.shape[0])
        train_time_sum += time.perf_counter() - epoch_t0

        val_t0 = time.perf_counter()
        val_loss, y_val_true, y_val_prob = eval_loader(model, val_loader, criterion, DEVICE)
        val_time_sum += time.perf_counter() - val_t0

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_auc": float(roc_auc_score(y_val_true, y_val_prob)),
            "val_roc_auc": float(roc_auc_score(y_val_true, y_val_prob)),
            "val_pr_auc": float(average_precision_score(y_val_true, y_val_prob)),
        }
        history.append(row)
        print(row)

        if best_epoch_row is None or (
            row["val_auc"], row["val_pr_auc"], -row["val_loss"]
        ) > (
            best_epoch_row["val_auc"],
            best_epoch_row["val_pr_auc"],
            -best_epoch_row["val_loss"],
        ):
            best_epoch_row = row
            torch.save(checkpoint_payload(model, optimizer, config, best_epoch_row), best_checkpoint_path)
            best_metrics_path.write_text(
                json.dumps(
                    {
                        "best_epoch": int(best_epoch_row["epoch"]),
                        "best_val_auc": float(best_epoch_row["val_auc"]),
                        "best_val_roc_auc": float(best_epoch_row["val_roc_auc"]),
                        "best_val_pr_auc": float(best_epoch_row["val_pr_auc"]),
                        "best_val_loss": float(best_epoch_row["val_loss"]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            early_stop_counter = 0
        else:
            val_auc_delta = float(row["val_auc"]) - float(best_epoch_row["val_auc"])
            if val_auc_delta < config.early_stopping_min_delta:
                early_stop_counter += 1
            else:
                early_stop_counter = 0

        if config.early_stopping_patience > 0 and early_stop_counter >= config.early_stopping_patience:
            stopped_early = True
            print(
                "early_stopping_triggered:",
                json.dumps(
                    {
                        "epoch": epoch,
                        "best_val_auc": float(best_epoch_row["val_auc"]),
                        "current_val_auc": float(row["val_auc"]),
                        "patience": config.early_stopping_patience,
                        "min_delta": config.early_stopping_min_delta,
                    }
                ),
            )
            break

    assert best_epoch_row is not None

    test_loss: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_confusion_matrix: list[list[int]] | None = None
    if test_loader is not None:
        test_loss, y_test_true, y_test_prob = eval_loader(model, test_loader, criterion, DEVICE)
        y_test_pred = (y_test_prob >= 0.5).astype(np.int64)
        test_roc_auc = float(roc_auc_score(y_test_true, y_test_prob))
        test_pr_auc = float(average_precision_score(y_test_true, y_test_prob))
        test_confusion_matrix = confusion_matrix(
            y_test_true.astype(np.int64),
            y_test_pred,
        ).tolist()
    total_wall_seconds = time.perf_counter() - total_t0
    epochs_completed = len(history)

    metrics = {
        "best_epoch": int(best_epoch_row["epoch"]),
        "best_val_auc": float(best_epoch_row["val_auc"]),
        "best_val_loss": float(best_epoch_row["val_loss"]),
        "best_val_roc_auc": float(best_epoch_row["val_roc_auc"]),
        "best_val_pr_auc": float(best_epoch_row["val_pr_auc"]),
        "test_loss": test_loss,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_confusion_matrix_threshold_0_5": test_confusion_matrix,
        "parameter_count": param_count,
        "device": str(DEVICE),
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
        "best_model_path": str(best_checkpoint_path),
        "best_metrics_path": str(best_metrics_path),
        "fast_search_mode": config.fast_search_mode,
        "skip_test_eval": config.skip_test_eval,
        "stopped_early": stopped_early,
        "epochs_completed": epochs_completed,
    }

    timing = {
        "dataset_prep_seconds": float(dataset_prep_seconds),
        "training_seconds": float(train_time_sum),
        "validation_seconds": float(val_time_sum),
        "total_wall_seconds": float(total_wall_seconds),
        "epochs_completed": epochs_completed,
    }

    archive_path = archive_run_bundle(config, split_meta, history, metrics, timing)
    metrics["archive_run_path"] = str(archive_path)
    metrics["archive_run_id"] = archive_path.stem
    timing["archive_run_path"] = str(archive_path)

    save_json_artifact("config_latest.json", {
        "seed": config.seed,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "stage_layers": config.stage_layers,
        "kernel_size": config.kernel_size,
        "dropout": config.dropout,
        "fast_search_mode": config.fast_search_mode,
        "skip_test_eval": config.skip_test_eval,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "data": config.data.to_dict(),
    })
    save_json_artifact("split_latest.json", split_meta)
    save_json_artifact("history_latest.json", {"history": history})
    save_json_artifact("metrics_latest.json", metrics)
    save_json_artifact("timing_latest.json", timing)

    val_bpb = objective_from_val_auc(metrics["best_val_auc"])
    peak_memory = peak_vram_mb(DEVICE)
    total_tokens_m = float(total_examples_seen * config.data.trace_len / 1_000_000)
    num_steps = len(train_loader) * epochs_completed

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {train_time_sum:.1f}")
    print(f"total_seconds:    {total_wall_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_memory:.1f}")
    print("mfu_percent:      0.0")
    print(f"total_tokens_M:   {total_tokens_m:.3f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {param_count / 1_000_000:.3f}")
    print(f"depth:            {config.depth}")
    print(f"fast_search_mode: {config.fast_search_mode}")
    print(f"skip_test_eval:   {config.skip_test_eval}")
    print(f"epochs_completed: {epochs_completed}")
    print(f"val_auc:          {metrics['best_val_auc']:.6f}")
    print(f"val_pr_auc:       {metrics['best_val_pr_auc']:.6f}")
    print(f"val_roc_auc:      {metrics['best_val_roc_auc']:.6f}")
    if metrics["test_pr_auc"] is None:
        print("test_pr_auc:      skipped")
        print("test_roc_auc:     skipped")
    else:
        print(f"test_pr_auc:      {metrics['test_pr_auc']:.6f}")
        print(f"test_roc_auc:     {metrics['test_roc_auc']:.6f}")
    print(f"best_model_path:  {best_checkpoint_path}")
    print(f"best_metrics:     {best_metrics_path}")
    print(f"archive_run_path: {archive_path}")
    print(f"artifacts_dir:    {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
