# Stage 2 Current Structure

This document describes the current Stage 2 framework structure, the technology stack, what was added beyond Stage 1, and how to test the available full-loop paths.

## What We Have Now

Stage 2 adds a paper-facing governed framework around the existing worker
backend in `harness/claw-code`.

The current system has two supported campaign paths:

1. **Stage 2 dry-run full loop**
   - deterministic
   - does not invoke Ollama
   - does not edit the ResNet node
   - exercises the new Stage 2 framework contracts end to end

2. **Stage 2 real campaign full loop**
   - invokes `run_real_campaign`
   - uses `ClawWorker` as the Stage 2 worker implementation
   - generates each worker packet from the Stage 2 manager proposal
   - calls the existing `harness/claw-code` autoresearch loop as a worker backend
   - can call local Ollama / Qwen
   - can create experiment commits and run the ResNet training command
   - returns a structured `WorkerResult`
   - lets the Stage 2 control plane own final validity and keep/discard decisions

The legacy harness is no longer described as a separate real execution control
plane. It is the current real worker backend behind the Stage 2 control plane.

## Technology Stack

Core language:

- Python 3.11+

Experiment node:

- PyTorch-based ResNet-trigger benchmark under `nodes/ResNet_trigger`
- `uv` for node environment management

Worker/runtime substrate:

- existing `harness/claw-code` autoresearch CLI
- local Ollama-compatible worker model path for real execution
- deterministic dry-run worker for Stage 2 smoke tests

Stage 2 framework:

- standard-library Python dataclasses and enums
- LangGraph / LangChain Core for the optional `langgraph_manager`
- LangChain Ollama for real local LLM manager calls
- append-only JSONL trial ledgers
- CSV exports for paper tables and figure data
- JSON-compatible node/campaign/manager/worker configs

LangGraph is intentionally scoped to proposal generation. It does not own
budget, lifecycle, worker execution, records, or decisions.

## Deployed Structure

```text
autoresearch_harness/
  configs/
    campaigns/
    managers/
    nodes/
    workers/
  docs/
    architecture.md
    experiment_protocol.md
    memory_architecture.md
    metrics.md
    node_spec.md
    provenance.md
    stage_2_current_structure.md
    trial_schema.md
  experiments/
    artifacts/
    ledgers/
    runs/
    summaries/
  harness/
    claw-code/
  nodes/
    ResNet_trigger/
  notebooks/
    stage2_development_smoke_tests/
  paper/
    figures/
    notes/
    tables/
  scripts/
    export_paper_figures.py
    export_paper_tables.py
    inspect_node.py
    parse_node_metric.py
    recover_pending.py
    run_campaign.py
    run_memory_ablation.py
    summarize_campaign.py
  src/
    autoresearch/
      control_plane/
      evaluation/
      legacy/
      manager/
      memory/
      nodes/
      reporting/
      worker/
```

## Why This Structure

The structure separates the research system into explicit layers:

- `configs/`: inspectable experiment, manager, worker, and node settings
- `src/autoresearch/control_plane/`: lifecycle, budget, and decision logic
- `src/autoresearch/manager/`: swappable proposal generators
- `src/autoresearch/worker/`: bounded worker interface
- `src/autoresearch/nodes/`: node contracts and metric parsing
- `src/autoresearch/memory/`: append-only records and memory summaries
- `src/autoresearch/evaluation/`: deterministic metrics and ablation summaries
- `src/autoresearch/reporting/`: paper table and figure CSV export
- `experiments/`: generated campaign ledgers and artifacts
- `paper/`: generated paper-ready tables, figures, and notes

This is good for the KDD AAE framing because the core contribution is visible: the governed control plane and audit/evaluation layer, not a coding-agent wrapper.

## What Changed Compared To Stage 1

Stage 1 proved that the governed loop can improve the ResNet-trigger node.

Stage 2 adds:

- formal node spec
- formal trial schema
- lifecycle state machine
- editable-scope validation
- budget accounting
- decision module
- append-only Stage 2 trial ledgers
- memory modes:
  - `none`
  - `append_only_summary`
  - `append_only_summary_with_rationale`
- repeated-bad-proposal detection
- manager interfaces:
  - `baseline_manager`
  - `prompt_manager`
- generic worker interface
- dry-run worker for deterministic testing
- real campaign runner through `ClawWorker`
- generated packet, patch, raw log, parsed metric, and legacy loop artifact capture
- pending-trial recovery commands
- node registry and inspection CLI
- dry-run fixed-budget campaign runner
- dry-run memory ablation runner
- campaign/governance table exports
- paper figure CSV exports
- documentation and paper-note scaffolding

The remaining Stage 3 gap is empirical, not structural: real campaigns should be
run through the Stage 2 path at enough budget to produce paper evidence. The
real worker still uses `harness/claw-code` internally, but Stage 2 now owns the
campaign lifecycle and record.

## Full-Loop Tests

### Stage 2 Dry-Run Full Loop

This is the recommended development full-loop test.

It checks:

- node registry
- manager proposal
- memory mode
- budget enforcement
- dry-run worker
- editable-scope validation
- decision logic
- append-only trial ledger
- campaign summary
- paper table CSV export
- paper figure CSV export

Command:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

python3 scripts/run_campaign.py \
  --node resnet_trigger \
  --campaign-id smoke \
  --budget 3 \
  --manager baseline_manager \
  --memory-mode none \
  --dry-run
```

Expected outputs:

- `experiments/ledgers/smoke_trials.jsonl`
- `paper/tables/main_campaign_summary.csv`
- `paper/tables/governance_metrics.csv`

Then export figure CSVs:

```bash
python3 scripts/export_paper_figures.py \
  --campaign-id smoke \
  --records experiments/ledgers/smoke_trials.jsonl
```

Expected outputs:

- `paper/figures/campaign_trajectory.csv`
- `paper/figures/accepted_discarded_invalid_counts.csv`
- `paper/figures/repeated_bad_idea_rates.csv`
- `paper/figures/gain_per_budget_unit.csv`

### Stage 2 Memory Ablation Dry Loop

This checks all three memory modes under equal budget.

Command:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

python3 scripts/run_memory_ablation.py \
  --node resnet_trigger \
  --budget 3 \
  --execute-dry-campaigns
```

Expected outputs:

- `experiments/ledgers/resnet_trigger_none_ablation_trials.jsonl`
- `experiments/ledgers/resnet_trigger_append_only_summary_ablation_trials.jsonl`
- `experiments/ledgers/resnet_trigger_append_only_summary_with_rationale_ablation_trials.jsonl`
- `paper/tables/memory_ablation_summary.csv`

### Stage 2 Real Campaign Full Loop

This is the canonical real execution path.

Use this only when you are ready for the worker to edit `nodes/ResNet_trigger/train.py`, invoke the local model, and run training.

Command:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

RESNET_TRIGGER_FAST_SEARCH=1 \
RESNET_TRIGGER_FAST_N_SIGNAL=1000 \
RESNET_TRIGGER_FAST_N_NOISE=1000 \
RESNET_TRIGGER_FAST_TRACE_LEN=4096 \
RESNET_TRIGGER_FAST_BATCH_SIZE=64 \
RESNET_TRIGGER_FAST_EPOCHS=3 \
RESNET_TRIGGER_FAST_SKIP_TEST=1 \
RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 \
RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 \
RESNET_TRIGGER_DEVICE=cpu \
python3 scripts/run_campaign.py \
  --node resnet_trigger \
  --campaign-id main_smoke \
  --budget 1 \
  --manager prompt_manager \
  --memory-mode append_only_summary_with_rationale \
  --node-root nodes/ResNet_trigger \
  --packet-defaults tests/stage_1_sprint_deliverable/loop_packet.json \
  --model qwen2.5-coder:7b \
  --host http://localhost:11434
```

Afterward inspect:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

tail -n 1 experiments/ledgers/main_smoke_trials.jsonl
find experiments/artifacts/trial-001 -maxdepth 1 -type f -print
```

Expected behavior:

- one candidate run is produced
- Stage 2 records keep, discard, or failed_invalid
- the ledger contains one authoritative `TrialRecord`
- artifact refs include generated packet, raw legacy result, parsed metric JSON, raw log if present, and patch diff if available
- no Stage 2 pending guard remains after the loop completes

### Pending-Guard Recovery

If a real campaign crashes while a trial is active, Stage 2 leaves a
`*_pending.json` guard next to the ledger. Recover it explicitly:

```bash
python3 scripts/recover_pending.py list
python3 scripts/recover_pending.py inspect experiments/ledgers/main_smoke_trials_pending.json
python3 scripts/recover_pending.py fail experiments/ledgers/main_smoke_trials_pending.json \
  --node resnet_trigger \
  --manager-mode prompt_manager \
  --worker-mode claw_style_worker \
  --memory-mode append_only_summary_with_rationale \
  --message "worker interrupted during local smoke"
```

Use `clear` only when you intentionally want to remove the guard without writing
a failed trial record.

## Development Smoke Tests

Run all current Stage 2 development smoke tests:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

python3 -m unittest discover tests
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_0_1_contracts_smoke.py
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_2_3_integration_smoke.py
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_4_memory_and_manager_smoke.py
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_6_16_system_smoke.py
```

Run the legacy harness regression:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m unittest tests.test_autoresearch_integration
```
