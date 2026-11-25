# Node mlgym_titanic — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §3 and §7, on the paper
repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a legal
opinion.

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates; HEAD at U1 §3 and at the port), task `titanicSurvival` (`configs/tasks/titanic.yaml`), dataset
`titanic` (`configs/datasets/titanic.yaml`), commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** Per the repository's `LICENSE` file (U1 §3), MLGym is released under the Creative Commons
Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0,
https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is taken or adapted from
MLGym is under the same licence: it may be used and shared for non-commercial purposes with attribution, and
changes must be indicated (below). MLGym's README notes that SWE-Agent and Modded-NanoGPT are MIT-licensed
and Gymnax and gymnax-blines are Apache-2.0; which other files those cover is not itemised upstream (U1 §3).
CC BY-NC 4.0 grants no patent rights (U1 §3). Attribution: "Adapted from MLGym (Nathani et al., 2025),
CC BY-NC 4.0; modified as listed below".

**Scope within the harness repository.** R1-TRACKG-PORT-DECISIONS (1): this node lives in
`third_party/mlgym/`, whose `LICENSE` is MLGym's CC BY-NC 4.0, and the root `LICENSE` and `README` state the
scope. The repository's MIT licence does not cover this node's MLGym-derived files: they stay under
CC BY-NC 4.0. The files written by the port (below) are the repository's own, under MIT.

**Changes made by the port.**
- The MLGym assets read are the task definition (`titanicSurvival`: read `train.csv`, predict `Survived` on
  `test.csv`, score a submission's accuracy with the task's `evaluate.py`) and the three task tables. MLGym
  ships **no baseline script** for this task (the task prompt asks the agent to write one); the editable
  baseline `train.py` (feature engineering and a small tabular MLP) is **written by the port**, not adapted
  from MLGym.
- The training loop, the split, the accuracies and the result channel are the port's frozen code
  (`run_fixed_budget.py`, `ar_worker.py`, `prepare_data.py`), modelled on the other Track G MLGym nodes; the
  task's `evaluate.py` is **not copied or run**: the frozen parent recomputes the accuracy (argmax over two
  logits against the loaded labels) itself. The shared frozen helpers (`ar_parent.py`, `ar_isolated.py`,
  `ar_fixed_loop.py`, `ar_data.py`) are the harness's own MIT code, copied byte-identically.
- `prepare_data.py` downloads the three tables from MLGym's **pinned commit** at setup, checks the pinned
  sha256 values and records them in the data manifest; the node and the branch hold no task data.

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | The Kaggle competition **"Titanic - Machine Learning from Disaster"** tables MLGym forwards in its repository: `data/titanic/data/train.csv` (891 rows, `Survived` label), `test.csv` (418 unlabelled rows) and `ground_truth.csv` (418 rows, the test labels). The port downloads them from `https://raw.githubusercontent.com/facebookresearch/MLGym/9d40c1b5035202018cd7091fb4e83a9c68b377c0/data/titanic/data/` in the frozen `setup_command`; nothing is committed or redistributed |
| Pinned hashes (checked at setup) | `train.csv` sha256 `4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`; `test.csv` sha256 `5c78320a80dc159b35c408363be46a5d93f078183d539db75ee42a796d6a4e95`; `ground_truth.csv` sha256 `af6609cdf1dad0b699bd3414c13e999b2c2f651972f0938254a835985d584306` (the bytes of the pinned commit; recorded in the data manifest as well) |
| Origin and terms | Kaggle competition "Titanic - Machine Learning from Disaster" (https://www.kaggle.com/competitions/titanic). Kaggle's competition rules and terms apply; MLGym's CC BY-NC 4.0 does not relicense the data. **Unknown / not stated**: no explicit licence is declared for the competition data (the U1 audit's rule for Kaggle data is to record the URLs and not claim a licence); MLGym's forwarded copy's identity with the Kaggle download was not checked row by row |
| Shape | `train.csv` 891 × 12 (`PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`); `test.csv` 418 × 11 (no `Survived`); `ground_truth.csv` 418 × 2 (`PassengerId`, `Survived`) with the same `PassengerId` set as `test.csv` |
| Redistribution | none: the branch holds no data; the tests use synthetic tables only; setup downloads the bytes at deployment time and verifies the pinned sha256 |

No container is used (native port): no image digest applies.
