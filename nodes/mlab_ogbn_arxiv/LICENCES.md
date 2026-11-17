# Node mlab_ogbn_arxiv — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §4 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

## Code

| Item | Value |
|---|---|
| Upstream | https://github.com/snap-stanford/MLAgentBench, HEAD `5d71205cc20a8e95d43aa7cb7120e89ca3323e31` (U1 §4), task `MLAgentBench/benchmarks/ogbn-arxiv` |
| Licence | **MIT, verified** in the repository's `LICENSE`. Its copyright line is the unfilled template `Copyright (c) [year] [fullname]`, so no copyright holder is named (U1 §4) |
| Attribution | "Task ogbn-arxiv from MLAgentBench (Huang et al., arXiv:2310.03302), MIT licence" |

**Changes made by the port.** The suite's `train.py` was restructured into the Track G fixed-step API. The editable `train.py` keeps the suite's MLP, optimizer and loss, with the number of outputs corrected from 172 to the 40 classes. Two differences come from the frozen loop: batch norm sees only a batch's 1,024 training seeds (the suite's MLP runs on a sampled 2-hop subgraph of them, so its batch statistics include sampled neighbours), and each epoch's last partial batch is dropped (88 full batches per epoch instead of 89; 356 steps are about 4.05 epochs). The `train.py` docstring lists every difference. The loop, step budget, data handling and metric moved into the frozen files.

**Written by the port** (Track G, Paper B, R1): `run_fixed_budget.py` (the frozen parent), `ar_worker.py` (the frozen worker entry), `ar_parent.py`, `ar_isolated.py`, `ar_fixed_loop.py`, `ar_data.py`,
`prepare_data.py`, `requirements.txt`, this file.

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | ogbn-arxiv from the OGB download host, through `ogb.nodeproppred.NodePropPredDataset`. This is the same dataset as the suite's `PygNodePropPredDataset`; the HF mirror `OGB/ogbn-arxiv` returned HTTP 401 to U1, so the OGB host is used (U1 §4) |
| Licence | **ODC-BY** per OGB's node-property page (https://ogb.stanford.edu/docs/nodeprop/); attribution required (U1 §4). Attribution: "ogbn-arxiv, Open Graph Benchmark (Hu et al., 2020), derived from the Microsoft Academic Graph" |
| Hash | the sha256 of the six arrays is written to `data/data_manifest.json` at setup and checked by every run; the census records the manifest's sha256 |

The node and its branch contain no task data. The port's tests use synthetic random data.
