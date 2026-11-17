# Node mlgym_fmnist — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §3 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates; HEAD at U1 §3 and at the port), task `imageClassificationFMnist`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** Per the repository's `LICENSE` file (U1 §3), MLGym is released under the Creative Commons Attribution-NonCommercial 4.0 International
licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is
taken or adapted from MLGym is under the same licence: it may be used and shared for non-commercial
purposes with attribution, and changes must be indicated (below). The files written by the port (Track G,
Paper B, R1) are listed below. MLGym's README notes that SWE-Agent and Modded-NanoGPT are MIT-licensed and Gymnax and
gymnax-blines are Apache-2.0; CC BY-NC 4.0 grants no patent rights (U1 §3). Attribution: "Adapted from MLGym
(Nathani et al., 2025), CC BY-NC 4.0; modified as listed below".

**Scope within the harness repository.** R1-TRACKG-PORT-DECISIONS (1): this node lives in `third_party/mlgym/`, whose `LICENSE` is MLGym's CC BY-NC 4.0, and the root `LICENSE` and `README` state the scope. The repository's MIT licence does not cover this node's MLGym-derived files: they stay under CC BY-NC 4.0 (parts that MLGym took from gymnax or gymnax-blines under Apache-2.0). The files written by the port (below) are the repository's own, under MIT.

**Changes made by the port.**
- `train.py` (editable) is adapted from MLGym's `baseline.py`: the same CNN and the Hugging Face Trainer
  defaults the baseline relies on, written out as an explicit AdamW optimizer and linear schedule; the
  Trainer, its per-epoch test evaluation and its best-checkpoint reload are removed.
- Written by the port: `run_fixed_budget.py` (the frozen parent), `ar_worker.py` (the frozen worker entry), `ar_parent.py`, `ar_isolated.py`, `ar_fixed_loop.py`, `ar_data.py`, `prepare_data.py`
  (fixed step budget, validation split carved from the training set, metric lines, hashed data manifest).
  MLGym's `evaluate.py` (test accuracy of a submission file) is replaced by the frozen test-accuracy line.

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | https://huggingface.co/datasets/zalando-datasets/fashion_mnist (MLGym `configs/datasets/fashionMnistHF.yaml`) |
| Licence | **MIT**, declared on the HF card (U1 §3). Attribution: "Fashion-MNIST, Zalando Research (Xiao, Rasul and Vollgraf, 2017), MIT" |
| Hub revision | U1 observed card revision `531be5e`. `prepare_data.py` STOPs unless the resolved revision starts with it (or with `--accept-revision`, after the author accepts another), and loads exactly that revision |
| Hash | the sha256 of the four arrays is written to `data/data_manifest.json` and checked by every run; the census records the manifest's sha256 |

No container is used (native ports): no image digest applies. The node and its branch contain no task data; the
port's tests use synthetic data.
