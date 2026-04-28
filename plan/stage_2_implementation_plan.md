# Stage 2 Implementation Plan

This plan turns `docs/plans/stage_2_kdd_aae_instruction_aligned.md` into an implementation sequence we can work through step by step.

The guiding rule is: keep the current Stage 1 loop working, then add the Stage 2 framework contracts, smoke tests, summaries, and ablation support around it.

## Implementation Strategy

Do not start with a large repo move.

First, create a new framework package under `src/autoresearch/` and wrap the existing working logic from:

- `harness/claw-code/src/autoresearch_worker.py`
- `harness/claw-code/src/autoresearch_runner.py`

The old loop remains usable while the new Stage 2 contracts become the authoritative paper-facing layer.

## Phase 1: Contracts and Lifecycle

Goal: define the framework contracts before changing campaign behavior.

Create:

- `src/autoresearch/nodes/spec.py`
- `src/autoresearch/memory/schemas.py`
- `src/autoresearch/control_plane/state_machine.py`
- `src/autoresearch/control_plane/permissions.py`
- `configs/nodes/resnet_trigger.yaml`

Implement:

- serializable node spec
- formal trial record schema
- lifecycle states
- valid state transitions
- editable-scope validation
- failure categories

Smoke tests:

```bash
python -m pytest tests/test_node_spec.py tests/test_trial_schema.py tests/test_state_machine.py tests/test_permissions.py
```

Pass conditions:

- `configs/nodes/resnet_trigger.yaml` loads.
- required node fields exist.
- trial records validate.
- invalid lifecycle jumps are rejected.
- `train.py` edits are valid.
- `prepare.py`, data files, and dependency edits are invalid.

## Phase 2: ResNet Node Integration

Goal: make the current ResNet-trigger benchmark conform to the Stage 2 node contract.

Create:

- `src/autoresearch/nodes/resnet_trigger/metric_parser.py`
- `src/autoresearch/nodes/resnet_trigger/validity.py`

Implement:

- parse `val_bpb` from `run.log`
- convert to `val_auc = 1 - val_bpb`
- validate finite metrics
- categorize missing metrics as `metric_missing`
- keep `train.py` as the only editable path

Smoke test:

```bash
python -m pytest tests/test_metric_parser.py
```

Real node smoke:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger
RESNET_TRIGGER_FAST_SEARCH=1 \
RESNET_TRIGGER_FAST_EPOCHS=1 \
RESNET_TRIGGER_FAST_SKIP_TEST=1 \
RESNET_TRIGGER_DEVICE=cpu \
uv run train.py > run.log 2>&1
```

Then parse:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
python scripts/parse_node_metric.py \
  --node resnet_trigger \
  --log nodes/ResNet_trigger/run.log
```

Pass conditions:

- parser returns `val_auc`
- parser returns source log path
- missing metrics fail cleanly

## Phase 3: Append-Only Trial Store

Goal: make Stage 2 trial records authoritative and append-only.

Create:

- `src/autoresearch/memory/append_store.py`
- `src/autoresearch/memory/provenance.py`

Implement:

- append trial records as JSONL
- never overwrite raw records
- store proposal, patch, run, metric, and decision provenance ids
- support campaign-level record loading

Smoke test:

```bash
python -m pytest tests/test_append_only_memory.py
```

Pass conditions:

- appending two records preserves both.
- records are read back in order.
- provenance fields exist.
- summaries do not mutate raw records.

## Phase 4: Control-Plane Lifecycle Wrapper

Goal: connect the current working Stage 1 loop to the Stage 2 lifecycle and trial schema.

Create:

- `src/autoresearch/control_plane/lifecycle.py`
- `src/autoresearch/control_plane/decision.py`
- `src/autoresearch/control_plane/budget.py`

Implement:

- begin trial
- validate patch scope
- execute trial through existing harness adapter
- parse metric
- evaluate keep/discard
- append final trial record
- enforce one active pending trial per campaign

Important: the current CLI loop can remain in `harness/claw-code`; this phase adds a Stage 2 wrapper and record writer around the same behavior.

Smoke test:

```bash
python -m pytest tests/test_control_plane_lifecycle.py
```

Pass conditions:

- trial cannot execute before scope validation.
- trial cannot be decided before metric parsing.
- invalid scope becomes `failed_invalid`.
- degraded metric becomes `discarded`.
- improved metric becomes `kept`.

## Phase 5: Campaign Summary and Paper Metrics

Goal: produce deterministic paper-facing summary files.

Create:

- `src/autoresearch/evaluation/metrics.py`
- `src/autoresearch/evaluation/campaign_summary.py`
- `src/autoresearch/reporting/export_tables.py`
- `scripts/summarize_campaign.py`
- `scripts/export_paper_tables.py`

Implement metrics:

- initial metric
- best metric
- final accepted metric
- net gain
- gain per trial
- gain per accepted trial
- kept count
- discarded count
- failed-invalid count
- acceptance rate
- invalid rate
- wall-clock totals
- complete provenance rate

Smoke test:

```bash
python scripts/summarize_campaign.py --campaign-id smoke
python scripts/export_paper_tables.py --campaign-id smoke
```

Expected outputs:

- `paper/tables/main_campaign_summary.csv`
- `paper/tables/governance_metrics.csv`

Pass conditions:

- CSV files exist.
- counts match input trial records.
- gain metrics are deterministic.

## Phase 6: Repeated-Bad-Idea Detection

Goal: implement the memory/governance metric needed for the KDD AAE ablation.

Create:

- `src/autoresearch/memory/similarity.py`
- `tests/test_repeated_bad_idea_detection.py`

Implement approximate repeated-bad detection using:

- normalized proposal text similarity
- targeted hyperparameter extraction
- same target parameter and same direction
- same failure category
- same patch signature class if available

Report:

- repeated bad proposal count
- repeated bad proposal rate
- repeated invalid proposal count
- repeated degraded proposal count

Smoke test:

```bash
python -m pytest tests/test_repeated_bad_idea_detection.py
```

Pass conditions:

- duplicate failed learning-rate proposals are flagged.
- unrelated proposals are not flagged.
- invalid repeat count and degraded repeat count are separated.

## Phase 7: Memory Modes and Ablation Runner

Goal: implement the Stage 2 memory/governance ablation.

Create:

- `src/autoresearch/memory/summarizer.py`
- `scripts/run_memory_ablation.py`

Implement memory modes:

- `none`
- `append_only_summary`
- `append_only_summary_with_rationale`

Manager context rules:

- `none`: current baseline, budget index, node spec, allowed edit scope only.
- `append_only_summary`: proposal summary, changed parameters, metric delta, decision status.
- `append_only_summary_with_rationale`: summary plus rationale, failure categories, repeated-bad warnings, current best strategy summary.

Smoke test:

```bash
python scripts/run_memory_ablation.py \
  --node resnet_trigger \
  --budget 3 \
  --dry-run
```

Pass conditions:

- creates three equal-budget ablation plans.
- records selected memory mode in every planned trial.
- exports `paper/tables/memory_ablation_summary.csv`.

## Phase 8: Manager Interfaces

Goal: make manager behavior swappable without changing the control plane.

Create:

- `src/autoresearch/manager/base.py`
- `src/autoresearch/manager/baseline_manager.py`
- `src/autoresearch/manager/prompt_manager.py`

Implement:

- common `propose_next_trial(status, memory_context)` interface
- simple heuristic baseline manager
- prompt manager adapter for the current manager style

Do not add LangGraph yet unless the previous phases are stable.

Smoke test:

```bash
python -m pytest tests/test_manager_interfaces.py
```

Pass conditions:

- both managers return structured proposals.
- proposals include summary, rationale, target files, and bounded objective.
- manager cannot directly write trial state.

## Phase 9: Full Loop Smoke Test

Goal: verify that the real local worker path still works after Stage 2 instrumentation.

Use the existing Stage 1 packet:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m src.main autoresearch loop \
  --root /Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger \
  --packet /Users/wongdowling/Documents/autoresearch_harness/tests/stage_1_sprint_deliverable/loop_packet.json \
  --model qwen2.5-coder:7b \
  --host http://localhost:11434 \
  --iterations 1 \
  --retry-limit 1
```

Pass conditions:

- loop exits successfully.
- worker edits only `train.py`.
- `results.tsv` receives a candidate row.
- `experiment_memory.jsonl` receives candidate and decision events.
- Stage 2 trial JSONL receives a final trial record.
- no pending experiment remains in `.autoresearch_state.json`.

## Phase 10: Documentation and Paper Outputs

Goal: make the repo credible as a KDD AAE research artifact.

Create or update:

- `README.md`
- `docs/architecture.md`
- `docs/node_spec.md`
- `docs/trial_schema.md`
- `docs/experiment_protocol.md`
- `docs/metrics.md`
- `docs/memory_architecture.md`
- `docs/provenance.md`
- `paper/notes/contribution_claims.md`

Minimum paper outputs:

- `paper/tables/main_campaign_summary.csv`
- `paper/tables/memory_ablation_summary.csv`
- `paper/tables/governance_metrics.csv`
- `paper/figures/campaign_trajectory.csv`
- `paper/figures/accepted_discarded_invalid_counts.csv`
- `paper/figures/repeated_bad_idea_rates.csv`
- `paper/figures/gain_per_budget_unit.csv`

## What We Will Not Do Yet

- frontend/dashboard
- many benchmark domains
- cloud deployment
- broad multi-agent system
- full Hermes-style assistant shell
- tight LangGraph dependency
- tight claw-code internals dependency
- direct ml-intern or multiautoresearch integration

## Final Stage 2 Acceptance Question

The implementation is aligned when this repository can credibly support a workshop paper titled:

> Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control

That requires:

- one clean architecture description
- one reproducible fixed-budget campaign
- one equal-budget memory/governance ablation
- paper-ready governance and optimization metrics
- complete provenance for every accepted result

