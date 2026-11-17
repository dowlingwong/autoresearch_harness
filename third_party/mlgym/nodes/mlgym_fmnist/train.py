"""Editable baseline of node mlgym_fmnist (MLGym imageClassificationFMnist, Track G fixed-step port).

The model and the optimizer settings are those of MLGym's
``data/imageClassificationFMnist/baseline.py`` (CC BY-NC 4.0, Meta
Platforms): the two-layer CNN (32 and 64 channels, max pooling, one linear
layer), inputs scaled to [0, 1], batch size 64, and the Hugging Face Trainer
defaults that baseline relies on, written out here: AdamW (lr 5e-5, betas
0.9/0.999, eps 1e-8, weight decay 0.01 on weights, none on biases), 500 warmup
steps then linear decay to zero, gradient clipping at 1.0.  The Trainer's
per-epoch evaluation on the test set and its "best checkpoint" reload are
not ported (they select on the test set).  The training loop, the step
budget, the data and the metric are in the frozen files.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "batch_size": 64,
    "grad_clip": 1.0,
    "amp_bf16": False,
}
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        x = self.pool(self.relu(self.conv1(pixel_values)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        return self.fc(x)


def preprocess(x):
    """uint8 images (N, 1, 28, 28) -> floats in [0, 1]."""
    return x.float() / 255.0


def build_model(info):
    return Net().to(info["device"])


def build_optimizer(model, total_steps):
    decay = [p for n, p in model.named_parameters() if not n.endswith("bias")]
    no_decay = [p for n, p in model.named_parameters() if n.endswith("bias")]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    def schedule(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return max(0.0, (total_steps - step) / max(1, total_steps - WARMUP_STEPS))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def training_step(model, batch, step):
    return F.cross_entropy(model(preprocess(batch["x"])), batch["y"])


def predict_logits(model, batch):
    return model(preprocess(batch["x"]))
