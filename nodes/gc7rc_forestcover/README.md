# Node gc7rc_forestcover (Track G)

1GC-7RC **Task 5** (tabular prediction): seven-class classification on **Forest Cover Type** (UCI Covertype,
581,012 rows, 54 features: 10 continuous + 44 binary one-hot), stratified 80/10/10 split, StandardScaler
normalisation, scored by accuracy. The task is the paper's; the 1GC-7RC suite repository is not public (404 in
the U1 audit and again on 25 Nov 2026), so this node implements the **paper's stated public dataset/task
semantics** with the port's own baseline, not suite code (see `LICENCES.md`).

**Editable** `train.py`: the paper's baseline as described — a 3-layer MLP (128 -> 256 -> 128, BatchNorm, ReLU,
dropout 0.3) with a seven-logit head, Adam at lr 1e-3 and cross-entropy. It provides `CONFIG`, `build_model`,
`build_optimizer`, `training_step` and `predict_logits`.

**Frozen**: `run_fixed_budget.py` (the isolated parent: data, stratified split, training-set standardisation,
accuracies), `ar_worker.py` (the only process that imports the editable code), `prepare_data.py` (one-off
setup), `ar_parent.py`, `ar_isolated.py`, `ar_fixed_loop.py`, `ar_data.py`, `requirements.txt`, `LICENCES.md`,
`data/` and `representative_edits/`. The shared frozen loop owns the step budget, the seeds, the batch order,
the loss log and the evaluation loop (R1-HARNESS-ISOLATION).

**Data.** The UCI Covertype zip is downloaded by the frozen `setup_command` outside the node, hashed, parsed
into `X.npy` (float32, 581,012 x 54) and `y.npy` (int8, classes 0-6), and recorded in the data manifest. Nothing
is redistributed.

```
export AR_TRACKG_DATA_ROOT=<a directory outside the repository>
python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/gc7rc_forestcover"
```

**Split and metrics.** The paper's stratified 80/10/10 split: per class a seeded permutation (fixed split seed
42), 10 % validation and 10 % test, the remaining 80 % training — 464,810 / 58,101 / 58,101 rows, the same split
in every run. Features are standardised with the training rows' mean and population std only. Selection is the
validation accuracy (`AR_METRIC val_accuracy=`); the sealed test accuracy is `AR_TEST test_accuracy=`.

**Budget.** A fixed number of optimizer steps (`run_budget` in the spec: 8,000 steps at batch 4,096, about 71
epochs of the 464,810 training rows; R1-BX-D5), enforced by the frozen loop's post-step hook; an edit can change
the compute per step but not the number of steps. The loop picks CUDA when the census assigns a GPU and the CPU
otherwise.

**Environment.** `requirements.txt` (numpy, pandas, torch); the census records the versions it uses.

**Tests.** `tests/trackg/test_port_1gc.py` (static checks and a synthetic-data smoke; the smoke needs numpy,
pandas and torch in the test interpreter), `tests/trackg/test_trackg_isolation.py` for the shared isolation
behaviour, and `tests/trackg/port_smoke_gpu.py` for the checklist-E smoke with the real data.
