# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Purpose

This is a **governed autonomous experimentation framework** targeting the KDD 2026 Agentic AI Evaluation (AAE) workshop. The core contribution is not the LLM or the coding agent — it is the **governed control plane**: explicit trial lifecycle states, editable-scope enforcement, append-only memory, keep/discard decisions owned by the framework (not the agent), and reproducible paper-facing metrics.

The project is framed as: **Agent = Model + Harness**. The control plane *is* the harness. The LLM manager is a replaceable component.

The active work plan lives in `plan/KDD_AAE_refinement_plan_v2.md`. Execution is chunked in `plan/KDD_AAE_execution_chunks.md`. When implementing anything, consult those files first.

---

## Commands

### Install
```bash
pip install -e ".[dev]"
# or from the repo root with uv:
uv sync
```

### Run all tests
```bash
pytest -p no:cacheprovider
```

### Run a single test file
```bash
pytest tests/test_trial_schema.py -p no:cacheprovider
pytest tests/test_state_machine.py -p no:cacheprovider -v
```

### Dry-run campaign (no Ollama required)
```bash
python3 scripts/run_campaign.py \
    --node resnet_trigger --campaign-id smoke --budget 3 --dry-run
```

### Real campaign (requires Ollama + node directory)
```bash
python3 scripts/run_campaign.py \
    --node resnet_trigger --campaign-id main \
    --budget 5 --manager prompt_manager \
    --memory-mode append_only_summary_with_rationale \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b --host http://localhost:11434
```

### Memory ablation (all three modes, dry or real)
```bash
# Dry (plan only, no execution):
python3 scripts/run_memory_ablation.py --node resnet_trigger --budget 3 --dry-run

# Dry campaigns (mock metrics):
python3 scripts/run_memory_ablation.py --node resnet_trigger --budget 3 --execute-dry-campaigns

# Real campaigns:
python3 scripts/run_memory_ablation.py \
    --node resnet_trigger --budget 5 --execute-real-campaigns \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b --host http://localhost:11434
```

### LangGraph manager smoke test (no Ollama — uses stub LLM)
```bash
python3 scripts/run_campaign.py \
    --node resnet_trigger --campaign-id lg_smoke --budget 2 \
    --manager langgraph_manager --dry-run --llm-stub
```

### Export paper tables and figures
```bash
python3 scripts/export_paper_tables.py --campaign-id <id>
python3 scripts/export_paper_figures.py --campaign-id <id>
```

### Recover a stale pending trial
```bash
python3 scripts/recover_pending.py --list
python3 scripts/recover_pending.py --inspect experiments/ledgers/<id>_pending.json
python3 scripts/recover_pending.py --mark-failed <id> --reason "manual recovery"
```

### Inspect a node spec
```bash
python3 scripts/inspect_node.py --node resnet_trigger
```

---

## Architecture

### Six-Layer Flow

```
Manager (proposes next trial)
    ↓
Control Plane (owns lifecycle, decisions, ledger)
    ↓
Worker (executes the edit + training run)
    ↓
Node (defines editable scope, run command, metric)
    ↓
Memory / Audit (append-only JSONL ledger + memory context)
    ↓
Evaluation / Reporting (paper-facing CSVs, figures)
```

**Critical invariant:** The manager cannot commit trial state. The control plane owns every state transition and the keep/discard/failed_invalid decision.

### Control Plane (`src/autoresearch/control_plane/`)

- `campaign.py` — `run_real_campaign()` and `run_dry_campaign()`. The main loop: build memory context → acquire pending guard → call worker → release guard → build `TrialRecord` → append to store. Also owns pending-trial guard helpers (`list_pending_guards`, `inspect_pending_guard`, `mark_pending_failed`, `clear_pending_guard`).
- `decision.py` — `decide_trial()`: purely functional, no side effects. Returns `kept` if candidate metric beats current best, `discarded` if valid but worse, `failed_invalid` if invalid.
- `permissions.py` — `validate_edit_scope()`: checks changed files against `editable_paths` and `frozen_paths` from the node spec. Returns a `ScopeValidationResult`.
- `state_machine.py` — Trial state enum and allowed transitions.
- `budget.py` — `BudgetState`: immutable counter.

### Memory (`src/autoresearch/memory/`)

- `schemas.py` — All core dataclasses: `TrialRecord`, `TrialProvenance`, and the `StrEnum` types (`ExecutionStatus`, `ValidityStatus`, `TrialDecision`, `FailureCategory`). Requires Python ≥ 3.11.
- `append_store.py` — `TrialAppendStore`: the append-only JSONL ledger. Never overwrites. `append()` and `read_all()`.
- `summarizer.py` — `build_memory_context()`: produces the `MemoryContext` injected into the manager. Three modes: `none` (node params only), `append_only_summary` (one line per trial), `append_only_summary_with_rationale` (line + rationale + repeated-bad warnings).
- `similarity.py` — `compute_repeated_bad_stats()`: Jaccard similarity + parameter-direction extraction to detect semantically repeated bad proposals.

### Nodes (`src/autoresearch/nodes/`)

- `spec.py` — `NodeSpec` (frozen dataclass) and `BudgetSpec`. Loaded from JSON-compatible YAML via `load_node_spec()`.
- `registry.py` — `load_registered_node(name)`: maps node names to YAML paths under `configs/nodes/`.
- `resnet_trigger/metric_parser.py` — `parse_val_auc()`: reads run log, converts `val_bpb` → `val_auc` (`1 - val_bpb`), prefers `val_bpb` over direct `val_auc` when both present.

### Managers (`src/autoresearch/manager/`)

All managers implement the `Manager` Protocol: `propose_next_trial(status, memory_context, node_spec) → ManagerProposal`.

- `baseline_manager.py` — Template-based, no LLM. Deterministic proposals for dry-run testing.
- `prompt_manager.py` — LLM-backed via Ollama direct call.
- `langgraph_manager.py` — LangGraph graph: `prepare_context → generate_proposal → validate_proposal`. Scoped to proposal generation only; never touches the control plane or ledger. Accepts an injected `llm` for stub testing (`--llm-stub`).

### Workers (`src/autoresearch/worker/`)

All workers implement the `Worker` Protocol: `run_trial(proposal, node_spec, budget_index) → WorkerResult`.

- `base.py` — `WorkerResult` dataclass + `DryRunWorker` (synthetic metrics, no filesystem changes).
- `claw_worker.py` — `ClawWorker`: the real worker. Generates a packet JSON, invokes the legacy claw-code harness via subprocess, captures patch diff, run log, git commits, and parsed metrics.
- `local_worker.py` — `LocalWorker`: direct-edit worker that parses "Change PARAM from X to Y" directives and applies them without the claw-code harness.

### Evaluation & Reporting

- `evaluation/campaign_summary.py` — `load_campaign_summary()`: loads a JSONL ledger and computes `CampaignMetrics`.
- `evaluation/metrics.py` — `CampaignMetrics`: acceptance rate, invalid rate, repeated-bad rate, net gain, provenance completeness, artifact completeness.
- `evaluation/ablations.py` — `build_memory_ablation_plan()`, `export_memory_ablation_summary()`.
- `reporting/export_tables.py` — `export_campaign_tables()`: writes paper-facing CSVs to `paper/tables/`.
- `reporting/write_report.py` — `write_campaign_report()`: markdown narrative report.
- `common/paths.py` — `REPO_ROOT`, `LEDGERS_DIR`, `ARTIFACTS_DIR`, `PAPER_TABLES_DIR` constants; `ledger_path(campaign_id)`.

---

## Key Data Contracts

### TrialRecord (the atomic unit of the ledger)

Every trial — kept, discarded, or failed — produces exactly one immutable `TrialRecord` appended to the JSONL ledger. Required fields include `trial_id`, `campaign_id`, `budget_index`, `validity_status`, `failure_category` (non-null when invalid), `decision`, and `provenance`. Validation is enforced in `__post_init__`.

### FailureCategory enum
`syntax_error | runtime_error | metric_missing | invalid_edit_scope | degraded_metric`

Adding a new category: add to `FailureCategory` in `schemas.py`, add detection logic in `campaign._record_from_worker_result()`, add a test in `tests/test_trial_schema.py`.

### Pending-trial guard
A `*_pending.json` file is written before calling the worker and deleted after. If it exists when a new campaign starts, `PendingTrialError` is raised. Use `scripts/recover_pending.py` to resolve stale guards — never manually delete them without appending a failure record first.

### Node spec YAML
Lives under `configs/nodes/<name>.yaml`. `editable_paths` is the whitelist; `frozen_paths` is the blocklist. The scope validator in `permissions.py` enforces both. To add a new node: add a YAML file and register the name in `nodes/registry.py`.

---

## Test Layout

```
tests/
  test_trial_schema.py          # TrialRecord validation rules
  test_state_machine.py         # State transitions
  test_permissions.py           # Scope enforcement
  test_node_spec.py             # NodeSpec / BudgetSpec loading
  test_append_only_memory.py    # TrialAppendStore
  test_metric_parser.py         # val_bpb → val_auc conversion
  test_repeated_bad_idea_detection.py  # Jaccard similarity + RepeatedBadStats
  stage2/test_stage2_control_plane.py  # Integration: full campaign loop
```

Tests use `unittest.TestCase`. No mocking of the filesystem — all store tests use `tempfile.TemporaryDirectory`. The `stage2/` integration tests use `FakeWorker` and `RaisingWorker` patterns.

---

## Active KDD AAE Work

The paper targets KDD 2026 Agentic AI Evaluation workshop. The contribution is framed as an **evaluation methodology** (governed harness), not an ML optimization result. Governance metrics (acceptance rate, repeated-bad rate, provenance completeness, failure taxonomy) are the primary results; val_AUC is secondary evidence.

**Plan files (read before implementing anything):**
- `plan/KDD_AAE_refinement_plan_v2.md` — authoritative merged plan (skeleton + harness literature synthesis + checklist)
- `plan/KDD_AAE_execution_chunks.md` — 23 ordered chunks across 5 phases with exact steps and acceptance criteria

**Current status: Level 1 (one real baseline + one agent trial).** Target: Level 2 minimum (5-trial campaign + real memory ablation + stress trial + paper tables).

**Priority order:** Chunks 1.1–1.5 (infra) → 2.1 (ablation smoke) → 2.2 (main campaign) → 2.3 (stress trial) → 2.4 (full ablation) → 3.1–3.2 (export) → 4.1–4.6 (paper writing).
