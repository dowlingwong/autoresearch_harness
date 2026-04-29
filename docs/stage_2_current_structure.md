# Stage 2 Current Structure

This document describes the current Stage 2 framework structure, the technology stack, what was added beyond Stage 1, and how to test the available full-loop paths.

## What We Have Now

Stage 2 adds a paper-facing framework layer around the working Stage 1 harness.

The current system has two loop paths:

1. **Stage 2 dry-run full loop**
   - deterministic
   - does not invoke Ollama
   - does not edit the ResNet node
   - exercises the new Stage 2 framework contracts end to end

2. **Stage 1 real worker full loop**
   - invokes the existing `harness/claw-code` autoresearch loop
   - can call local Ollama / Qwen
   - can create experiment commits and run the ResNet training command
   - remains the current real execution backend

The Stage 2 framework is intentionally built around the Stage 1 loop instead of replacing it immediately.

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
- append-only JSONL trial ledgers
- CSV exports for paper tables and figure data
- JSON-compatible node/campaign/manager/worker configs

No heavy framework dependency has been introduced yet. LangGraph remains optional future manager orchestration, not a current dependency.

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
- node registry and inspection CLI
- dry-run fixed-budget campaign runner
- dry-run memory ablation runner
- campaign/governance table exports
- paper figure CSV exports
- documentation and paper-note scaffolding

The main remaining gap is that Stage 2 real campaign execution is not yet wired as the default real worker path. Real execution still goes through the Stage 1 harness loop, while Stage 2 dry-run execution proves the framework contracts.

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

### Real Legacy Worker Full Loop

This is the current real execution path.

Use this only when you are ready for the worker to edit `nodes/ResNet_trigger/train.py`, invoke the local model, and run training.

Command:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code

PYTHONPATH=$PWD \
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
python3 -m src.main autoresearch loop \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger \
  --packet /Users/wongdowling/Documents/autoresearch_harness/tests/stage_1_sprint_deliverable/loop_packet.json \
  --model qwen2.5-coder:7b \
  --host http://localhost:11434 \
  --iterations 1 \
  --retry-limit 1
```

Afterward inspect:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger

git status --short
tail -n 20 results.tsv
tail -n 20 experiment_memory.jsonl
cat .autoresearch_state.json
```

Expected behavior:

- one candidate run is produced
- decision is keep, discard, or crash
- memory/result files are updated
- no pending experiment remains after the loop completes

## Development Smoke Tests

Run all current Stage 2 development smoke tests:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

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

