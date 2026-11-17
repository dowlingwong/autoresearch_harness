"""Editable baseline of node mlab_ogbn_arxiv (MLAgentBench ogbn-arxiv, Track G fixed-step port).

The model, optimizer and loss are those of MLAgentBench's
``benchmarks/ogbn-arxiv/env/train.py`` (MIT licence): a 2-layer MLP on the
node features (hidden size 16, batch norm, ReLU, dropout 0, log-softmax),
negative log-likelihood loss, Adam with learning rate 1, batches of 1,024
training nodes.  The suite's script builds a ReduceLROnPlateau scheduler but
never steps it, so the baseline has no schedule.  The suite's MLP has 172
outputs (ogbn-arxiv has 40 classes); the port uses 40, which the frozen
evaluation requires.  The MLP uses only node
features; the graph is available in ``info`` for models that use it.  The
training loop, the step budget, the data and the metric are in the frozen
files.
"""
import torch
import torch.nn.functional as F

CONFIG = {
    "batch_size": 1024,
    "grad_clip": None,
    "amp_bf16": False,
}
HIDDEN = 16
NUM_LAYERS = 2
DROPOUT = 0.0
LEARNING_RATE = 1.0


class MLP(torch.nn.Module):
    def __init__(self, x, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.x = x
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_channels, hidden_channels)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, idx):
        h = self.x[idx]
        for lin, bn in zip(self.lins[:-1], self.bns):
            h = F.dropout(F.relu(bn(lin(h))), p=self.dropout, training=self.training)
        return torch.log_softmax(self.lins[-1](h), dim=-1)


def build_model(info):
    model = MLP(info["x"], info["x"].shape[1], HIDDEN, info["num_classes"], NUM_LAYERS, DROPOUT)
    return model.to(info["device"])


def build_optimizer(model, total_steps):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), None


def training_step(model, batch, step):
    return F.nll_loss(model(batch["idx"]), batch["y"])


def predict_logits(model, batch):
    return model(batch["idx"])
