# Current Readiness and Next Steps

Date: 2026-05-11

This note records the current state before final paper writing. The short version is: the framework implementation is mostly in place, but the current real experimental evidence is not yet paper-ready. We can draft stable sections now, but final Results, Abstract, and Introduction should wait until the real campaigns are rerun cleanly.

## Current Problems

### 1. Main real campaign is not usable as the primary result

`kdd_main_5trial` completed with 5 real records, complete provenance, and no pending guard. However, all 5 trials are `failed_invalid`:

- 3 `runtime_error`
- 2 `no_op_patch`
- 0 `kept`
- 0 `discarded`

This demonstrates governance and failure capture, but it is not a convincing main campaign result. The paper needs at least some valid lifecycle diversity: kept, discarded, and failed-invalid outcomes.

### 2. Memory ablation is still incomplete

The memory ablation is the most important experiment for the paper claim about rationale memory reducing repeated poor proposals.

Current state:

- `ablation_none`: 5 real trials, but all invalid.
- `ablation_append_only_summary`: only 2 real trials, with a stale pending guard.
- `ablation_append_only_summary_with_rationale`: 5 dry-run trials, not real.

This means the core memory claim is still a hypothesis, not a result.

### 3. Likely false `no_op_patch` classification

Some trials were marked `no_op_patch` even though training ran and `val_auc` was parsed. This suggests the worker/result extraction path is losing the patch after the legacy loop discards/restores `train.py`.

The likely issue: `ClawWorker` tries to capture `patch.diff` after the legacy worker has already restored the edit, so the diff can be empty even though the worker did make an edit and training ran.

This must be fixed before interpreting no-op rates or failure taxonomy results.

### 4. Reset is not fully clearing legacy node state

`reset_node_state.py` restores editable files and removes campaign ledgers/artifacts, but the legacy node can still retain local state such as:

- `nodes/ResNet_trigger/.autoresearch_state.json`
- `nodes/ResNet_trigger/results.tsv`
- `nodes/ResNet_trigger/experiment_memory.jsonl`
- `nodes/ResNet_trigger/artifacts/`

That state can contaminate fresh campaigns, especially because the legacy loop uses its own baseline/best state internally.

### 5. Current paper tables are not final evidence

The exported tables are structurally valid, but they reflect the current flawed real run:

- `governance_metrics.csv`: invalid rate is 1.0 for `kdd_main_5trial`.
- `main_campaign_summary.csv`: no final accepted metric.
- `memory_ablation_summary.csv`: still reflects dry-run or incomplete evidence.

Do not write final numerical claims from these tables yet.

## Immediate Fixes

### A. Fix reset hygiene

Update `scripts/reset_node_state.py` so a real campaign reset can also clear node-local legacy state:

- `.autoresearch_state.json`
- `results.tsv`
- `experiment_memory.jsonl`
- node-local `artifacts/`
- stale `run.log`

Keep this behavior explicit and documented. If needed, add a flag such as `--clear-node-runtime-state`, but for final campaign runs the clean-reset path should be used every time.

### B. Fix patch capture before legacy discard

Update the legacy/Stage 2 worker path so `patch.diff` is captured before any discard/restore removes the edit.

Acceptance for this fix:

- a successful edit produces a non-empty `patch.diff`;
- a true byte-identical edit is still classified as `no_op_patch`;
- a discarded but valid metric trial is classified as `discarded`, not `failed_invalid / no_op_patch`;
- tests cover this behavior.

### C. Clear stale pending guard

Resolve:

`experiments/ledgers/ablation_append_only_summary_pending.json`

For final experiments, prefer resetting the whole campaign rather than marking this partial interrupted run as paper evidence.

## Rerun Plan

### Step 1: One-trial real smoke

Run a fresh 1-trial smoke after fixing reset and patch capture.

Acceptance:

- ledger has 1 record;
- no pending guard remains;
- provenance is complete;
- if the worker edits and training runs, `patch.diff` exists and is non-empty;
- no false `no_op_patch`.

### Step 2: Main 5-trial real campaign

Rerun:

`kdd_main_5trial`

Acceptance:

- 5 records;
- complete provenance;
- artifact completeness passes;
- at least one non-kept decision;
- at least one valid kept or discarded trial;
- no stale pending guard;
- patch artifacts exist for real edits.

### Step 3: Full real memory ablation

Rerun all three arms from the same clean baseline:

- `ablation_none`, 5 real trials
- `ablation_append_only_summary`, 5 real trials
- `ablation_append_only_summary_with_rationale`, 5 real trials

Then run:

```bash
uv run --extra dev python scripts/check_kdd_memory_ablation.py --require-real
```

Acceptance:

- all three ledgers have 5 real records;
- no pending guards;
- `memory_ablation_summary.csv` is regenerated;
- repeated-bad rates are based on real runs, not dry-run profiles.

### Step 4: Stress trials

Keep both stress trials:

- `kdd_stress_scope`
- `kdd_stress_noop`

Rerun only if the artifact paths or taxonomy code changed.

### Step 5: Regenerate paper artifacts

After final real campaigns:

```bash
uv run --extra dev python scripts/export_kdd_tables.py
uv run --extra dev python scripts/export_kdd_figures.py --figure all
uv run --extra dev python scripts/check_kdd_artifact_completeness.py
uv run --extra dev python scripts/generate_artifact_manifest.py --output artifact_manifest.json
```

Then inspect:

- `paper/tables/main_campaign_summary.csv`
- `paper/tables/governance_metrics.csv`
- `paper/tables/memory_ablation_summary.csv`
- `paper/tables/failure_taxonomy.csv`
- `paper/tables/artifact_completeness_report.txt`

## Writing Guidance

### Safe to write now

These sections can be drafted before final numbers:

- Related Work
- System Design
- Experiment Setup
- Failure taxonomy definitions
- Governance protocol
- Artifact/provenance schema
- Limitations framework

### Do not finalize yet

Wait for clean real campaign results before finalizing:

- Results
- Abstract
- Introduction
- final Discussion claims
- any quantitative claim about rationale memory
- any claim about improvement or acceptance rate

## Paper Claim Status

Current honest status:

- Governed control plane: implemented.
- No-op guard: implemented, but current false-positive behavior must be fixed before final taxonomy claims.
- Event stream: implemented.
- Artifact manifest/completeness tooling: implemented.
- LangChain/provider packaging: implemented.
- Main real campaign evidence: not yet usable.
- Real memory ablation evidence: not yet complete.

The next milestone is not more writing. The next milestone is a clean rerun that produces interpretable real ledgers.
