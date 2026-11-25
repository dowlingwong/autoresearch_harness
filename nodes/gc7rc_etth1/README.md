# Node gc7rc_etth1 — 1GC-7RC Task 6, ETTh1 96-step forecasting (Track G isolated port)

One node of the Track G T1 expansion (`R1-T1-NODE-EXPANSION`, candidate 2: 1GC-7RC ETTh1).  The 1GC-7RC suite
repository is not public (U1 audit, 17 Nov 2026), so this node ports the **public dataset and task semantics the
paper states**, not suite code:

- ETTh1 (hourly electricity-transformer temperatures, 7 features, 17,420 rows), downloaded from the pinned
  ETDataset commit; never committed.
- Multivariate **96-step-ahead** forecasting given **336** historical steps, standard LTSF 12/4/4-month split
  (8,640 / 2,880 / 2,880 rows), standardised on the training rows, metric **MSE** (test split sealed).
- Baseline **DLinear** (moving-average decomposition, kernel 25, two linear projections), batch 128,
  Adam, MSE, written by the port from the DLinear paper because the suite's own baseline is unavailable.
- Fixed budget: **2,000 optimizer steps** (`run_budget.kind: steps`; no wall-clock budget, `R1-BX-D5`).

## Layout

| File | Role |
|---|---|
| `train.py` | **editable**; model, optimizer, loss, prediction (`DLinear` baseline) |
| `run_fixed_budget.py` | frozen parent: data check, split, windowing, standardisation, worker calls, MSEs, result channel |
| `ar_worker.py` | frozen worker entry: the only process that imports `train.py` |
| `ar_fixed_loop.py` | frozen fixed-step loop (step budget, seeds, window order, loss log, prediction conversion) |
| `ar_parent.py`, `ar_data.py`, `ar_isolated.py` | frozen shared helpers, byte-identical to the T1 harness copies |
| `prepare_data.py` | frozen one-off setup: pinned download + hash, `series.npy`, data manifest |
| `LICENCES.md` | licence and provenance record (CC BY-ND 4.0 governing, data not redistributed) |
| `representative_edits/` | four declared edits (E1–E4) and their manifest, committed before any measurement |

## Setup and run

```sh
export AR_TRACKG_DATA_ROOT=/path/outside/the/repo/data   # the node links data/ into it
python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/gc7rc_etth1"
python3 run_fixed_budget.py --steps 2000
```

The runner starts `run_fixed_budget.py` directly (isolation block in `configs/trackg/nodes/gc7rc_etth1.yaml`);
the parent never imports editable code and writes `AR_METRIC val_mse=…` / `AR_TEST test_mse=…` to the result
channel only.

## Threat-model notes for the review (same class as the T1 torch nodes)

- The worker can read `data/series.npy`, which holds the validation and test targets; a proposal could build
  predictions from them.  No scored target is sent to the worker, but the file is readable (the standard limit
  of the non-Breakout nodes).  Declared-edit review and the S1 audit read the diffs for label access.
- The budget fixes optimizer steps, not FLOPs: an edit can make each step heavier (as E4 does) and runs longer.
  The runner timeout bounds the time.
- The split seed is recorded but the split is fixed and chronological, so it changes nothing (as `mlab_ogbn_arxiv`).
