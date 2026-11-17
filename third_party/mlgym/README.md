# third_party/mlgym — Track G nodes ported from MLGym (CC BY-NC 4.0)

R1-TRACKG-PORT-DECISIONS (1): everything under this directory is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International licence (`LICENSE` here, the text of MLGym's own `LICENSE` at commit
9d40c1b5035202018cd7091fb4e83a9c68b377c0), not under the repository's MIT licence. It holds the four Track G nodes
ported from MLGym (https://github.com/facebookresearch/MLGym, Meta Platforms) and their node specs:

| node | spec | MLGym task |
|---|---|---|
| `nodes/mlgym_cifar10` | `configs/trackg/nodes/mlgym_cifar10.yaml` | imageClassificationCifar10 |
| `nodes/mlgym_fmnist` | `configs/trackg/nodes/mlgym_fmnist.yaml` | imageClassificationFMnist |
| `nodes/mlgym_houseprice` | `configs/trackg/nodes/mlgym_houseprice.yaml` | regressionKaggleHousePrice |
| `nodes/mlgym_breakout_minatar` | `configs/trackg/nodes/mlgym_breakout_minatar.yaml` | rlBreakoutMinAtar |

Each node's `LICENCES.md` records its sources and the data's own terms (the data are never committed; setup
writes them outside the repository). The House Prices table is the user's own Kaggle download, passed to setup with
`--from-file` (R1-TRACKG-PORT-DECISIONS (2)). The nodes run isolated (R1-HARNESS-ISOLATION): the runner starts each
node's frozen parent `run_fixed_budget.py`, which starts `ar_worker.py`, the only process that imports the node's
editable code.
