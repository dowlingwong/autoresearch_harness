# Node mlab_cifar10 — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §4 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

## Code

| Item | Value |
|---|---|
| Upstream | https://github.com/snap-stanford/MLAgentBench, HEAD `5d71205cc20a8e95d43aa7cb7120e89ca3323e31` (U1 §4), task `MLAgentBench/benchmarks/cifar10` |
| Licence | **MIT, verified** in the repository's `LICENSE`. Its copyright line is the unfilled template `Copyright (c) [year] [fullname]`, so no copyright holder is named (U1 §4) |
| Attribution | "Task cifar10 from MLAgentBench (Huang et al., arXiv:2310.03302), MIT licence" |

**Changes made by the port.** The suite's `train.py` was restructured into the Track G fixed-step API. The editable `train.py` keeps the suite's model, optimizer, loss and preprocessing (its docstring lists every difference). The loop, step budget, data handling, validation split and metric moved into the frozen files.

**Written by the port** (Track G, Paper B, R1): `run_fixed_budget.py` (the frozen parent), `ar_worker.py` (the frozen worker entry), `ar_parent.py`, `ar_isolated.py`, `ar_fixed_loop.py`, `ar_data.py`,
`prepare_data.py`, `requirements.txt`, this file.

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | CIFAR-10 archive fetched by `torchvision.datasets.CIFAR10(download=True)` from the canonical `cs.toronto.edu` URL, as the suite's `scripts/prepare.py` does (U1 §4) |
| Licence | **unknown / not stated** (U1 §4). Citation: Krizhevsky 2009, *Learning Multiple Layers of Features from Tiny Images* |
| Hash | torchvision checks the archive's md5 (`c58f30108f718f92721af3b95e74349a`, torchvision's published value). `prepare_data.py` checks that torchvision's recorded md5 is this value, writes the sha256 of the four arrays to `data/data_manifest.json`, and every run checks them. The census records the manifest's sha256 |

The node and its branch contain no task data. The port's tests use synthetic random data.
