"""Editable baseline of node mlgym_titanic (MLGym titanicSurvival, Track G fixed-step port).

MLGym's ``data/titanic`` task (``configs/tasks/titanic.yaml``, task id ``titanicSurvival``) ships the Kaggle
"Titanic - Machine Learning from Disaster" tables (``train.csv`` with the ``Survived`` label, unlabelled
``test.csv`` and the test labels in ``ground_truth.csv``), a sample submission and the task's
``evaluate.py``, which scores a submission's accuracy.  MLGym ships no baseline script for this task (the
task asks the agent to write one); the baseline below is written by the port.  The port keeps the task's
surface -- predict ``Survived`` from passenger attributes, scored by accuracy -- but fixes the training
loop: the frozen parent and worker hold the data, the split, the step budget and the accuracies; this file
holds the feature engineering, the model, the optimizer, the schedule and the loss.

The baseline: seventeen engineered features (ticket class, sex, age with a missing-value flag, siblings and
parents aboard, family size, an alone flag, log fare, three embarked indicators, four name-title indicators
and a cabin-known flag), standardised with the training rows' mean and standard deviation, and a
two-hidden-layer ReLU MLP with dropout, trained with AdamW and cross-entropy for the fixed step budget.  The
model returns two logits; the frozen parent computes the argmax accuracy.  Nothing is selected on the test
set: the frozen parent scores the validation split's accuracy as the selection metric and the sealed
test accuracy for the record (nothing is selected on the test set).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "batch_size": 128,   # at most this node's cap in the frozen loop
    "grad_clip": 1.0,
    "amp_bf16": False,
}
HIDDEN = (256, 128)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500

TITLES = ("Mr", "Mrs", "Miss", "Master")
# Columns whose training median fills their missing values (the flag columns keep the fact that they were
# missing); the remaining columns are never missing in the task tables.
MEDIAN_FILL = ("Age", "Fare")


def _title(names):
    """The name title (``Mr``, ``Mrs``, ``Miss``, ``Master``); every other title is ``Rare``."""
    title = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    return title.where(title.isin(TITLES), "Rare")


def _raw_columns(frame):
    """The engineered columns before imputation and standardisation, in a fixed order."""
    out = pd.DataFrame(index=frame.index)
    out["Pclass"] = frame["Pclass"].astype(np.float64)
    out["Sex"] = (frame["Sex"] == "female").astype(np.float64)
    out["Age"] = frame["Age"].astype(np.float64)
    out["AgeMissing"] = frame["Age"].isna().astype(np.float64)
    out["SibSp"] = frame["SibSp"].astype(np.float64)
    out["Parch"] = frame["Parch"].astype(np.float64)
    out["FamilySize"] = (frame["SibSp"] + frame["Parch"] + 1).astype(np.float64)
    out["IsAlone"] = (out["FamilySize"] == 1).astype(np.float64)
    out["Fare"] = np.log1p(frame["Fare"].astype(np.float64))
    out["CabinKnown"] = frame["Cabin"].notna().astype(np.float64)
    embarked = frame["Embarked"].fillna("Missing")
    out["Embarked_C"] = (embarked == "C").astype(np.float64)
    out["Embarked_Q"] = (embarked == "Q").astype(np.float64)
    out["Embarked_S"] = (embarked == "S").astype(np.float64)
    title = _title(frame["Name"])
    for name in TITLES:
        out["Title_" + name] = (title == name).astype(np.float64)
    return out


def build_features(train_frame):
    """Fit the feature encoding on the training rows only and return ``encode(frame) -> float32 array``.

    *train_frame* holds the training rows' raw task columns (``train.csv`` long form, ``Survived``
    included).  The returned encoder maps a frame with the same raw columns to a (rows, features) float32
    array; the frozen worker applies it to the training, validation and test rows.  The imputation values
    and the standardisation statistics come from *train_frame*, never from the validation or test rows.
    """
    raw_train = _raw_columns(train_frame)
    fill = {name: float(raw_train[name].median()) for name in MEDIAN_FILL}

    def _prepare(frame):
        columns = _raw_columns(frame).copy()
        for name, value in fill.items():
            columns[name] = columns[name].fillna(value)
        return columns

    fitted = _prepare(train_frame)
    mean = fitted.mean()
    scale = fitted.std(ddof=0).replace(0.0, 1.0)

    def encode(frame):
        values = (_prepare(frame) - mean) / scale
        return np.ascontiguousarray(values.to_numpy(dtype=np.float32))

    return encode


class TabularMLP(nn.Module):
    """A ReLU MLP with dropout after every hidden layer and two output logits."""

    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        width = n_features
        for hidden_width in hidden:
            layers += [nn.Linear(width, hidden_width), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_width
        layers += [nn.Linear(width, 2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def build_model(info):
    """The unfitted classifier for the training features (``info`` from the frozen worker)."""
    return TabularMLP(int(info["n_features"]), HIDDEN, DROPOUT).to(info["device"])


def build_optimizer(model, total_steps):
    """AdamW with weight decay on the weight matrices only, a short warmup then linear decay."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    def schedule(step):
        if step < WARMUP_STEPS:
            return (step + 1) / max(1, WARMUP_STEPS)
        return max(0.0, (total_steps - step) / max(1, total_steps - WARMUP_STEPS))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def training_step(model, batch, step):
    return F.cross_entropy(model(batch["x"]), batch["y"])


def predict_logits(model, batch):
    return model(batch["x"])
