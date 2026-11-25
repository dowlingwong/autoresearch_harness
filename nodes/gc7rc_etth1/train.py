"""Editable baseline of node gc7rc_etth1 (1GC-7RC Task 6, ETTh1 96-step forecasting; Track G fixed-step port).

The task description states the standard LTSF setup: multivariate 96-step-ahead forecasting on ETTh1 with 7
features, given 336 historical steps, standardised on the training split; the paper's baseline is DLinear
(moving-average decomposition with kernel 25 into trend and seasonal components, each projected by a linear
layer), the metric is MSE.  The 1GC-7RC suite repository is not public, so this is the port's own implementation
of that described baseline, not suite code.  The training loop, the step budget, the data, the split, the
windowing and the metric are in the frozen files (ar_fixed_loop.py, run_fixed_budget.py, ar_worker.py); this
file only provides the model, optimizer, loss and prediction functions they call.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "batch_size": 128,
    "grad_clip": None,
    "amp_bf16": False,
}
LEARNING_RATE = 1e-3
KERNEL_SIZE = 25  # DLinear's moving-average kernel, as stated for the 1GC-7RC Task 6 baseline


class DLinear(nn.Module):
    """DLinear (Zeng et al. 2023): channel-independent linear projections of the trend and seasonal parts."""

    def __init__(self, seq_len, horizon, n_features, kernel_size=KERNEL_SIZE):
        super().__init__()
        self.kernel_size = kernel_size
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.linear_trend = nn.Linear(seq_len, horizon)

    def decomposition(self, x):
        """x (B, L, C) -> (seasonal, trend), each (B, L, C); the moving average is replicate-padded to L."""
        channels = x.permute(0, 2, 1)  # (B, C, L)
        pad = (self.kernel_size - 1) // 2
        padded = F.pad(channels, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(padded, self.kernel_size, stride=1)
        return x - trend.permute(0, 2, 1), trend.permute(0, 2, 1)

    def forward(self, x):
        seasonal, trend = self.decomposition(x)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1))  # shared over channels
        trend = self.linear_trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)  # (B, horizon, C)


def build_model(info):
    return DLinear(info["seq_len"], info["horizon"], info["n_features"], KERNEL_SIZE).to(info["device"])


def build_optimizer(model, total_steps):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), None


def training_step(model, batch, step):
    return F.mse_loss(model(batch["x"]), batch["y"])


def predict(model, batch):
    return model(batch["x"])
