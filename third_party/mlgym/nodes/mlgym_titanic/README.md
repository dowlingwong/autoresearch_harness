# Node mlgym_titanic (Track G)

MLGym task `titanicSurvival` (`configs/tasks/titanic.yaml`, dataset `titanic`), commit
`9d40c1b5035202018cd7091fb4e83a9c68b377c0` (CC BY-NC 4.0; see `LICENCES.md`): predict `Survived` from the
Kaggle "Titanic - Machine Learning from Disaster" passenger tables, scored by accuracy.

**Editable** `train.py`: the port's own baseline (MLGym ships no baseline for this task) — seventeen
engineered features standardised on the training rows and a small tabular ReLU MLP with dropout, AdamW
warmup/linear decay and cross-entropy. It provides `CONFIG`, `build_features`, `build_model`,
`build_optimizer`, `training_step` and `predict_logits`.

**Frozen**: `run_fixed_budget.py` (the isolated parent), `ar_worker.py` (the only process that imports the
editable code), `prepare_data.py` (one-off setup), `ar_parent.py`, `ar_isolated.py`, `ar_fixed_loop.py`,
`ar_data.py`, `requirements.txt`, `LICENCES.md`, `data/` and `representative_edits/`. The parent holds the
data, the split and the accuracies; the shared frozen loop owns the step budget, the seeds, the batch order,
the loss log and the evaluation loop (R1-HARNESS-ISOLATION).

**Data.** The three task tables come from MLGym's pinned commit and are downloaded by the frozen
`setup_command` outside the node; the script checks the pinned sha256 values and writes the data manifest.
Nothing is redistributed.

```
export AR_TRACKG_DATA_ROOT=<a directory outside the repository>
python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_titanic"
```

**Split and metrics.** 20 % of `train.csv` is the validation split, drawn by `AR_SPLIT_SEED`; the rest
trains (the frozen loop drops each epoch's last incomplete batch); the task's own `test.csv` is the sealed
test set, its labels aligned by `PassengerId` from `ground_truth.csv`. Selection is the validation accuracy
(`AR_METRIC val_accuracy=`); the sealed test accuracy is `AR_TEST test_accuracy=`.

**Budget.** A fixed number of optimizer steps (`run_budget` in the spec; R1-BX-D5), enforced by the frozen
loop's post-step hook; an edit can change the compute per step but not the number of steps. This node is
CPU-only: the worker pins the model to the CPU.

**Environment.** `requirements.txt` (numpy, pandas, torch); the census records the versions it uses.

**Tests.** `tests/trackg/test_port_mlgym.py` (static checks and a synthetic-data smoke; the smoke needs
numpy, pandas and torch in the test interpreter), `tests/trackg/test_trackg_isolation.py` for the shared
isolation behaviour, and `tests/trackg/port_smoke_gpu.py` for the checklist-E smoke with the real data (no
`--gpu-uuid` for a CPU-only node).
