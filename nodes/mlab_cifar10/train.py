"""Editable baseline of node mlab_cifar10 (MLAgentBench cifar10, Track G fixed-step port).

The model, optimizer and loss are those of MLAgentBench's
``benchmarks/cifar10/env/train.py`` (MIT licence): the small LeNet-style CNN,
SGD with learning rate 0.1 and momentum 0.9, batch size 128, cross-entropy,
inputs normalised with mean 0.5 and std 0.5.  The training loop, the step
budget, the data and the metric are in the frozen files (ar_fixed_loop.py,
run_fixed_budget.py); this file only provides the pieces they call.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "batch_size": 128,
    "grad_clip": None,
    "amp_bf16": False,
}
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocess(x):
    """uint8 images (N, 3, 32, 32) -> normalised floats."""
    return (x.float() / 255.0 - 0.5) / 0.5


def build_model(info):
    return Net().to(info["device"])


def build_optimizer(model, total_steps):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    return optimizer, None


def training_step(model, batch, step):
    logits = model(preprocess(batch["x"]))
    return F.cross_entropy(logits, batch["y"])


def predict_logits(model, batch):
    return model(preprocess(batch["x"]))
