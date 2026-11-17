# Node mlab_imdb — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §4 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

## Code

| Item | Value |
|---|---|
| Upstream | https://github.com/snap-stanford/MLAgentBench, HEAD `5d71205cc20a8e95d43aa7cb7120e89ca3323e31` (U1 §4), task `MLAgentBench/benchmarks/imdb` |
| Licence | **MIT, verified** in the repository's `LICENSE`. Its copyright line is the unfilled template `Copyright (c) [year] [fullname]`, so no copyright holder is named (U1 §4) |
| Attribution | "Task imdb from MLAgentBench (Huang et al., arXiv:2310.03302), MIT licence" |

**Changes made by the port.** The suite ships an unfinished skeleton (`model = None`), so the editable `train.py` is a baseline written by the port for the task's stated goal of fine-tuning DistilBERT (see its docstring). The loop, step budget, data handling, validation split and metric are in the frozen files.

**Written by the port** (Track G, Paper B, R1): `run_fixed_budget.py`, `ar_fixed_loop.py`, `ar_data.py`,
`prepare_data.py`, `requirements.txt`, this file.

## Data and pretrained model (never redistributed)

| Item | Value |
|---|---|
| Dataset | https://huggingface.co/datasets/stanfordnlp/imdb (the suite's `load_dataset("imdb")`) |
| Dataset licence | card metadata **`license: other`**, Licensing Information "More Information Needed": **terms unspecified upstream; no licence is claimed** (U1 §4). Citation: Maas et al. 2011 |
| Dataset revision | U1 observed card revision `e628166`. `prepare_data.py` STOPs unless the resolved revision starts with it (or with the revision passed with `--accept-dataset-revision`, after the author accepts it), and loads exactly that revision |
| Pretrained model | https://huggingface.co/distilbert-base-uncased, named by the task's research problem. **Not covered by U1**; the model card states Apache-2.0, which U1 should confirm. Only the five files listed in `prepare_data.py` are downloaded, unchanged, from the revision resolved at setup and recorded in the manifest |
| Hash | the sha256 of every array and model file is written to `data/data_manifest.json` and checked by every run; the census records the manifest's sha256 |

The node and its branch contain no task data. The port's tests use synthetic random data.
