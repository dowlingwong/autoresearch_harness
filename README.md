# autoresearch_harness

**Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control**

A governed autonomous experimentation framework for evaluating agentic optimization under bounded execution constraints.

---

## What this is

A research system in which an LLM manager proposes bounded hyperparameter changes, a controlled worker executes them, and a formal control plane decides what to keep — with every trial recorded in an append-only ledger for full auditability.

The core contribution is not the LLM or the coding agent. It is the **governed control plane**: explicit trial lifecycle states, editable-scope enforcement, append-only memory, keep/discard decisions owned by the framework (not the agent), and reproducible paper-facing metrics.

## What this is not

- Not a general autonomous scientist
- Not a LangChain/LangGraph demo — LangGraph is one optional manager backend
- Not a wrapper around a coding agent
- Not proof of scientific discovery
- Not a universal optimizer

---

## Stage 1 result

A 3 × 5 governed campaign on the ResNet-trigger near-threshold detector node produced a measurable improvement under bounded execution:

| Metric | Value |
|---|---|
| Baseline val_auc | 0.7799 |
| Best val_auc | 0.7876 |
| Net improvement | +0.0076 |
| Campaign shape | 3 manager rounds × 5 trials |
| Worker edit surface | `train.py` only |

---

## Stage 2 architecture

Six explicit layers, each with a bounded responsibility.

```
Manager / Planner
  ↓  (ManagerProposal — objective, rationale, target files)
Stage 2 Control Plane
  ↓  (lifecycle, budget, scope check, pending-trial guard)
Worker Interface
  ↓  (generates packet from proposal, calls legacy loop)
Experiment Node (ResNet_trigger)
  ↓  (runs train.py, writes run.log)
Metric Parser
  ↓  (val_bpb → val_auc, validity checks)
Decision (keep / discard / failed_invalid)
  ↓
Append-only Trial Ledger  →  Memory Context  →  (back to Manager)
  ↓
Paper Metrics + Table Export
```

### Layer responsibilities

| Layer | Owns | Does not own |
|---|---|---|
| Manager | proposal generation | budget, lifecycle, records |
| Control plane | trial state, budget, decision | worker execution details |
| Worker | code edit + run | state transitions, trial records |
| Node | experiment environment | management logic |
| Memory | append-only records, summaries | proposals or decisions |
| Evaluation | paper metrics from records | live experiment state |

---

## Repository layout

```
autoresearch_harness/
├── README.md
├── pyproject.toml
├── src/autoresearch/               # Stage 2 governed framework
│   ├── control_plane/
│   │   ├── campaign.py             # run_dry_campaign / run_real_campaign
│   │   ├── decision.py             # keep / discard / failed_invalid logic
│   │   ├── budget.py               # fixed-trial budget enforcement
│   │   ├── lifecycle.py            # trial state machine + legacy adapter
│   │   ├── permissions.py          # editable-scope validation
│   │   └── state_machine.py        # formal trial state transitions
│   ├── manager/
│   │   ├── base.py                 # ManagerProposal / ManagerStatus protocol
│   │   ├── baseline_manager.py     # deterministic heuristic proposals
│   │   ├── prompt_manager.py       # memory-aware proposals
│   │   └── langgraph_manager.py    # LangGraph planner (optional backend)
│   ├── worker/
│   │   ├── base.py                 # WorkerResult protocol + DryRunWorker
│   │   └── claw_worker.py          # ClawWorker — generates packet from proposal
│   ├── nodes/
│   │   ├── spec.py                 # NodeSpec schema
│   │   ├── registry.py             # resolve node name → YAML config
│   │   └── resnet_trigger/
│   │       ├── metric_parser.py    # val_bpb → val_auc
│   │       └── validity.py
│   ├── memory/
│   │   ├── schemas.py              # TrialRecord dataclass
│   │   ├── append_store.py         # append-only JSONL ledger
│   │   ├── summarizer.py           # memory modes + context builder
│   │   ├── similarity.py           # repeated-bad-idea detection
│   │   └── provenance.py           # stable ID generation
│   ├── evaluation/
│   │   ├── metrics.py              # all paper metrics from trial records
│   │   ├── campaign_summary.py     # load + summarize a ledger
│   │   └── ablations.py            # memory ablation plan rows
│   └── reporting/
│       ├── export_tables.py        # write paper CSVs
│       └── export_figures.py       # write figure-data CSVs
│
├── configs/
│   ├── nodes/resnet_trigger.yaml   # formal node contract
│   ├── campaigns/resnet_trigger_smoke.json
│   └── managers/langgraph_manager.json
│
├── scripts/
│   ├── run_campaign.py             # dry-run + real campaign entrypoint
│   ├── run_memory_ablation.py      # equal-budget memory ablation
│   ├── summarize_campaign.py       # summarize a ledger
│   ├── export_paper_tables.py      # write paper/tables/
│   ├── export_paper_figures.py     # write paper/figures/
│   ├── inspect_node.py             # print node spec as JSON
│   └── parse_node_metric.py        # parse a real run.log
│
├── experiments/
│   ├── ledgers/                    # append-only trial JSONL files
│   ├── artifacts/                  # generated packets + run logs
│   └── summaries/
│
├── paper/
│   ├── tables/                     # main_campaign_summary.csv, governance_metrics.csv, …
│   └── figures/                    # trajectory CSVs for plotting
│
├── notebooks/
│   ├── stage2_overview_and_demo.ipynb     # ← start here for Stage 2
│   └── stage2_development_smoke_tests/    # unit/smoke test scripts
│
├── harness/claw-code/              # current real worker backend
└── nodes/ResNet_trigger/           # active experiment node
```

---

## Quickstart

### Install

```bash
uv venv
uv pip install -e ".[dev]"
```

### Inspect the node contract

```bash
python3 scripts/inspect_node.py --node resnet_trigger
```

### Dry-run campaign (no worker, synthetic metrics)

```bash
python3 scripts/run_campaign.py \
    --node resnet_trigger \
    --campaign-id smoke \
    --budget 3 \
    --manager baseline_manager \
    --memory-mode none \
    --dry-run
```

### Dry-run with LangGraph manager (no Ollama needed)

```bash
python3 scripts/run_campaign.py \
    --node resnet_trigger \
    --campaign-id langgraph_smoke \
    --budget 3 \
    --manager langgraph_manager \
    --memory-mode append_only_summary_with_rationale \
    --dry-run --llm-stub
```

### Summarize a ledger

```bash
python3 scripts/summarize_campaign.py \
    --campaign-id smoke \
    --records experiments/ledgers/smoke_trials.jsonl
```

### Dry-run memory ablation (all three modes)

```bash
python3 scripts/run_memory_ablation.py \
    --node resnet_trigger \
    --budget 3 \
    --execute-dry-campaigns
```

### Real campaign quickstart

This is the canonical real execution path. Stage 2 owns the campaign lifecycle,
pending guard, validity checks, decision, artifact capture, and append-only
ledger. `harness/claw-code` is called only as the current worker backend.

Requires Ollama running at `http://localhost:11434` with the configured model.

```bash
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
    --campaign-id real_smoke \
    --budget 1 \
    --manager prompt_manager \
    --memory-mode append_only_summary_with_rationale \
    --node-root nodes/ResNet_trigger \
    --packet-defaults tests/stage_1_sprint_deliverable/loop_packet.json \
    --model qwen2.5-coder:7b \
    --host http://localhost:11434 \
    --allow-any-branch
```

Expected Stage 2 outputs:

- `experiments/ledgers/real_smoke_trials.jsonl`
- `experiments/artifacts/trial-001/generated_packet.json`
- `experiments/artifacts/trial-001/legacy_loop_result.json`
- `experiments/artifacts/trial-001/parsed_metrics.json`
- `experiments/artifacts/trial-001/run.log`, if the worker produced a log
- `experiments/artifacts/trial-001/patch.diff`, if a diff can be captured

Recover a stale pending guard after an interrupted real run:

```bash
python3 scripts/recover_pending.py list
python3 scripts/recover_pending.py inspect experiments/ledgers/real_smoke_trials_pending.json
python3 scripts/recover_pending.py fail experiments/ledgers/real_smoke_trials_pending.json \
    --node resnet_trigger \
    --manager-mode prompt_manager \
    --worker-mode claw_style_worker \
    --memory-mode append_only_summary_with_rationale \
    --message "worker interrupted during smoke run"
```

### Real memory ablation (requires Ollama)

```bash
python3 scripts/run_memory_ablation.py \
    --node resnet_trigger \
    --budget 5 \
    --execute-real-campaigns \
    --node-root nodes/ResNet_trigger \
    --packet-defaults tests/stage_1_sprint_deliverable/loop_packet.json
```

---

## Trial record schema

Every trial produces one append-only JSON record. Key fields:

```json
{
  "trial_id": "main-trial-007",
  "campaign_id": "main",
  "node_id": "resnet_trigger",
  "budget_index": 7,
  "manager_mode": "prompt_manager",
  "memory_mode": "append_only_summary_with_rationale",
  "proposal_summary": "reduce-lr-5e-4",
  "proposal_rationale": "...",
  "targeted_files": ["train.py"],
  "patch_ref": "experiments/artifacts/trial-007/patch.diff",
  "raw_log_ref": "experiments/artifacts/trial-007/run.log",
  "execution_status": "success",
  "validity_status": "valid",
  "parsed_metrics": {"val_auc": 0.7876},
  "current_best_before": 0.7841,
  "delta_vs_best": 0.0035,
  "decision": "kept",
  "decision_rationale": "Candidate improved maximize objective by 0.003500.",
  "timestamp_start": "2026-04-29T14:00:00Z",
  "timestamp_end": "2026-04-29T14:18:22Z",
  "provenance": {
    "proposal_id": "...", "patch_id": "...",
    "run_id": "...", "metric_id": "...", "decision_id": "..."
  },
  "extra": {
    "manager": {"context_sha256": "..."},
    "worker": {
      "generated_packet_ref": "experiments/artifacts/trial-007/generated_packet.json",
      "parsed_metrics_ref": "experiments/artifacts/trial-007/parsed_metrics.json",
      "legacy_loop_result_ref": "experiments/artifacts/trial-007/legacy_loop_result.json"
    }
  }
}
```

---

## Memory modes

Three modes define the primary ablation experiment:

| Mode | Manager receives |
|---|---|
| `none` | node spec, budget index, current best only |
| `append_only_summary` | above + per-trial summary (proposal, delta, decision) |
| `append_only_summary_with_rationale` | above + rationale, failure categories, repeated-bad warnings, best strategy |

---

## Manager options

| Manager | Description |
|---|---|
| `baseline_manager` | Deterministic heuristic proposals, no LLM |
| `prompt_manager` | Memory-aware proposals, no LLM (text construction) |
| `langgraph_manager` | LangGraph `prepare_context → generate_proposal → validate_proposal` graph; real LLM or injected stub |

All three implement the same interface: `propose_next_trial(status, memory_context, node_spec) → ManagerProposal`.

The control plane never calls a manager directly to write trial state. Managers only return proposals.

---

## Smoke tests

```bash
# CI-friendly unittest suite
python3 -m unittest discover tests

# Contracts and lifecycle
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_0_1_contracts_smoke.py

# ResNet node integration and trial records
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_2_3_integration_smoke.py

# Memory modes and manager interfaces
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_4_memory_and_manager_smoke.py

# System integration (budget, decision, campaign)
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_6_16_system_smoke.py

# Proposal-to-TrialRecord chain (ClawWorker mock)
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_proposal_to_trial_smoke.py

# LangGraph manager (FakeListChatModel, no Ollama)
python3 notebooks/stage2_development_smoke_tests/stage2_langgraph_manager_smoke.py
```

The `tests/` suite is intended for CI. The notebook smoke scripts are retained
as development checks and documentation for the Stage 2 milestones.

---

## Paper outputs

```bash
python3 scripts/export_paper_tables.py --campaign-id <id>
python3 scripts/export_paper_figures.py --campaign-id <id>
```

Outputs written to `paper/tables/` and `paper/figures/`:

| File | Contents |
|---|---|
| `main_campaign_summary.csv` | optimization metrics per campaign |
| `governance_metrics.csv` | acceptance rate, invalid rate, provenance completeness |
| `memory_ablation_summary.csv` | repeated-bad counts, compression ratio by memory mode |
| `campaign_trajectory.csv` | val_auc per trial |
| `accepted_discarded_invalid_counts.csv` | decision breakdown |
| `repeated_bad_idea_rates.csv` | memory-mode comparison |
| `gain_per_budget_unit.csv` | budget efficiency |

---

## Key design decisions

**The control plane owns all state transitions.** Managers and workers request actions; only the control plane commits trial state. A worker that crashes or produces an invalid result still gets a `failed_invalid` record in the ledger — nothing is silently lost.

**Proposals drive worker packets.** `ClawWorker.run_trial()` generates a temporary `AutoresearchExperimentPacket` JSON from the `ManagerProposal`. The packet's `objective` always comes from the proposal, never from a static file. This is what makes memory ablation meaningful: different memory modes produce different objectives, which flow into different worker instructions.

**Pending-trial guard.** Before calling the worker, the control plane writes a `*_pending.json` guard next to the ledger. If the process crashes mid-trial, the next run raises `PendingTrialError` rather than silently overwriting state. Use `scripts/recover_pending.py` to list, inspect, mark failed, or clear stale guards.

**LangGraph is scoped to the manager layer only.** The `langgraph_manager` graph (`prepare_context → generate_proposal → validate_proposal`) may only produce a `ManagerProposal`. It has no access to budget, lifecycle, trial records, worker execution, or append-only memory writes.

---

## Worker backend

`harness/claw-code/` is the current real worker backend. The Stage 2 `ClawWorker` calls it as a subprocess via `ClawCodeAutoresearchAdapter`, then converts the result into `WorkerResult`. The legacy loop's keep/discard recommendation is ignored as an authority; the Stage 2 control plane makes the authoritative decision from the parsed metric, edit scope, node contract, and current campaign state.

---

## Contributing / extending

To add a new manager:

1. Implement `propose_next_trial(status, memory_context, node_spec) -> ManagerProposal` in `src/autoresearch/manager/`.
2. Add a case to `_manager()` in `src/autoresearch/control_plane/campaign.py`.
3. Add a smoke test under `notebooks/stage2_development_smoke_tests/`.

To add a new experiment node:

1. Create a YAML spec in `configs/nodes/`.
2. Add a metric parser under `src/autoresearch/nodes/<node_name>/metric_parser.py`.
3. Register the node name in `src/autoresearch/nodes/registry.py`.
