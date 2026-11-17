"""Editable baseline of node mlab_imdb (MLAgentBench imdb, Track G fixed-step port).

MLAgentBench's imdb task ships an unfinished skeleton (``model = None``) and
asks for DistilBERT fine-tuning; it has no runnable baseline.  This baseline
is written by the port: ``distilbert-base-uncased`` with a fresh 2-class
head (AutoModelForSequenceClassification), reviews truncated to 256 tokens,
AdamW (lr 2e-5, weight decay 0.01 except on biases and on every LayerNorm
weight: ``embeddings.LayerNorm`` and each block's ``sa_layer_norm`` and
``output_layer_norm``) with a linear decay to zero, gradient
clipping at 1.0, bf16 autocast on CUDA, batch size 16.  The training loop,
the step budget, the data and the metric are in the frozen files.
"""
import torch
import torch.nn.functional as F

CONFIG = {
    "batch_size": 16,
    "grad_clip": 1.0,
    "amp_bf16": True,
}

# Optimizer.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def build_model(info):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(info["pretrained_dir"], num_labels=2)
    return model.to(info["device"])


def build_optimizer(model, total_steps):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_steps))
    return optimizer, scheduler


# Input length.
MAX_LEN = 256  # the stored reviews have 512 tokens; this baseline uses the first 256


def _inputs(batch):
    return {"input_ids": batch["input_ids"][:, :MAX_LEN].long(),
            "attention_mask": batch["attention_mask"][:, :MAX_LEN].long()}


def training_step(model, batch, step):
    logits = model(**_inputs(batch)).logits
    return F.cross_entropy(logits.float(), batch["y"])


def predict_logits(model, batch):
    return model(**_inputs(batch)).logits
