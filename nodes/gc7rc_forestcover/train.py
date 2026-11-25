"""Editable baseline of node gc7rc_forestcover (1GC-7RC Task 5, Forest Cover Type; Track G fixed-step port).

The task description states the setup: seven-class tabular prediction on Forest Cover Type (UCI Covertype,
581,012 rows, 54 features: 10 continuous + 44 binary one-hot), stratified 80/10/10 split and StandardScaler
normalisation; the paper's baseline is a 3-layer MLP (128 -> 256 -> 128) with BatchNorm, ReLU and dropout 0.3,
scored by accuracy.  The 1GC-7RC suite repository is not public, so this is the port's own implementation of
that described baseline, not suite code.  The data, the stratified split, the standardisation, the training
loop, the step budget and the accuracies are in the frozen files (run_fixed_budget.py, ar_worker.py,
ar_fixed_loop.py, ar_parent.py); this file only provides the model, optimizer, loss and prediction functions
they call.

The baseline: three hidden layers of 128, 256 and 128 units, each ``Linear -> BatchNorm1d -> ReLU -> Dropout``
(dropout 0.3, as stated for the paper's baseline), then a linear seven-logit head; Adam at lr 1e-3 and
cross-entropy.  The frozen parent computes the argmax validation accuracy (selection) and the sealed test
accuracy (record).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "batch_size": 4096,  # the paper's Task 5 batch size
    "grad_clip": None,
    "amp_bf16": False,
}
LEARNING_RATE = 1e-3

HIDDEN = (128, 256, 128)  # the paper's 3-layer baseline widths
DROPOUT = 0.3             # the paper's dropout


class TabularMLP(nn.Module):
    """The paper's 3-layer tabular MLP: BatchNorm + ReLU + dropout after every hidden layer, n-class logits."""

    def __init__(self, n_features: int, n_classes: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        width = n_features
        for hidden_width in hidden:
            layers += [nn.Linear(width, hidden_width), nn.BatchNorm1d(hidden_width), nn.ReLU(),
                       nn.Dropout(dropout)]
            width = hidden_width
        layers += [nn.Linear(width, n_classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def build_model(info):
    """The unfitted classifier for the training features (``info`` from the frozen worker)."""
    return TabularMLP(int(info["n_features"]), int(info["num_classes"]), HIDDEN, DROPOUT).to(info["device"])


def build_optimizer(model, total_steps):
    """Adam at the baseline learning rate; the paper states no schedule for its Task 5 baseline."""
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), None


def training_step(model, batch, step):
    return F.cross_entropy(model(batch["x"]), batch["y"])


def predict_logits(model, batch):
    return model(batch["x"])
