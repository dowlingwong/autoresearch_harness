# Stage 2 Raw Implementation Plan

This plan is the working roadmap for turning the Stage 1 autoresearch prototype into a Stage 2 KDD AAE-aligned research artifact.

The implementation style is incremental: keep the existing `harness/claw-code` loop working, add the Stage 2 framework layer around it, then replace ad hoc paths with explicit control-plane contracts only after each layer has smoke tests.

## Core Rule

Do not start with a large repo move.

The current Stage 1 loop already works. Stage 2 should first make that loop:

- auditable
- typed
- append-only
- summarizable
- paper-measurable
- reproducible enough for a fixed-budget campaign

Only after those contracts are stable should old harness behavior be refactored behind new Stage 2 interfaces.

## Current Status

Completed thin-slice work:

- New package exists under `src/autoresearch/`.
- Existing `harness/claw-code` code was not moved.
- Legacy wrapper exists in `src/autoresearch/legacy/claw_code.py`.
- Node and trial contracts exist.
- ResNet-trigger metric parsing exists.
- Append-only trial store exists.
- Campaign metrics and basic table export exist.
- Memory modes and repeated-bad detection exist.
- Baseline and prompt manager interfaces exist.
- Development smoke tests live under `notebooks/stage2_development_smoke_tests/`.

Important limitation:

The Stage 2 framework exists as a contract and wrapper layer. It does not yet provide a full real fixed-budget campaign runner or a real equal-budget memory ablation runner.

## Alignment Notes

### `stage_2_implementation_plan.md`

There is no conceptual conflict between `stage_2_implementation_plan.md` and this raw plan.

The implementation plan expands this raw plan into more phases:

- raw Priority 1 maps to implementation Phase 1
- raw Priority 2 maps to implementation Phases 2 and 4
- raw Priority 3 maps to implementation Phase 5, with repeated-bad detection split into Phase 6
- raw Priority 4 maps to implementation Phase 7
- raw manager layer maps to implementation Phase 8
- real worker validation maps to implementation Phase 9
- documentation and paper outputs map to implementation Phase 10

Unresolved implementation-plan items:

- `src/autoresearch/control_plane/decision.py` is not implemented.
- `src/autoresearch/control_plane/budget.py` is not implemented.
- `scripts/export_paper_tables.py` is not implemented as a standalone script.
- smoke tests are currently development scripts under `notebooks/stage2_development_smoke_tests/`, not final CI tests under `tests/`.

### `stage_2_kdd_aae_instruction_aligned.md`

There is no major conflict between the KDD-aligned instruction and this plan.

The KDD-aligned instruction is broader and more paper-complete. It adds requirements for:

- repo packaging and standard output folders
- generic worker interfaces
- real fixed-budget campaign execution
- real memory/governance ablation execution
- more complete governance, runtime, and reporting metrics
- deterministic paper figure CSVs
- architecture and paper documentation
- final real-loop validation

Those requirements are integrated below as new priorities.

## Completed Priorities

### Priority 0 — Framework Package

Status: implemented.

Purpose:

Create the Stage 2 package without moving old Stage 1 code.

Implemented files:

- `src/autoresearch/__init__.py`
- `src/autoresearch/legacy/claw_code.py`

Notes:

- `ClawCodeAutoresearchAdapter` wraps the existing legacy CLI.
- `loop_and_record(...)` can run the legacy loop and append Stage 2 trial records.
- The old `harness/claw-code` CLI is not directly instrumented yet.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_0_1_contracts_smoke.py
```

### Priority 1 — Contracts and Lifecycle

Status: implemented.

Purpose:

Define the formal node spec, trial schema, lifecycle states, and editable-scope validation.

Implemented files:

- `src/autoresearch/nodes/spec.py`
- `src/autoresearch/memory/schemas.py`
- `src/autoresearch/control_plane/state_machine.py`
- `src/autoresearch/control_plane/permissions.py`
- `configs/nodes/resnet_trigger.yaml`

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_0_1_contracts_smoke.py
```

Acceptance criteria:

- `configs/nodes/resnet_trigger.yaml` loads.
- required node fields exist.
- trial records validate.
- invalid lifecycle transitions are rejected.
- `train.py` edits are accepted.
- `prepare.py`, data files, and dependency edits are rejected.

### Priority 2 — ResNet Node Integration and Stage 2 Trial Records

Status: implemented as a framework wrapper, not yet as direct legacy CLI instrumentation.

Purpose:

Make the current ResNet-trigger loop conform to Stage 2 contracts.

Implemented files:

- `src/autoresearch/nodes/resnet_trigger/metric_parser.py`
- `src/autoresearch/nodes/resnet_trigger/validity.py`
- `src/autoresearch/control_plane/lifecycle.py`
- `src/autoresearch/memory/append_store.py`
- `src/autoresearch/memory/provenance.py`
- `scripts/parse_node_metric.py`

Current behavior:

- `val_bpb` is parsed and converted to `val_auc = 1 - val_bpb`.
- legacy loop result payloads can be converted into Stage 2 `TrialRecord`s.
- records can be appended to JSONL without overwriting previous records.

Remaining work:

- make the real campaign runner call this path consistently.
- decide whether to instrument the legacy CLI directly or keep all Stage 2 record writing in the wrapper layer.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_2_3_integration_smoke.py
```

Optional real-node parser check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
python3 scripts/parse_node_metric.py \
  --node resnet_trigger \
  --log nodes/ResNet_trigger/run.log
```

### Priority 3 — Campaign Summary and Basic Paper Metrics

Status: implemented for basic campaign and governance tables.

Purpose:

Compute paper-facing campaign metrics from Stage 2 trial ledgers.

Implemented files:

- `src/autoresearch/evaluation/metrics.py`
- `src/autoresearch/evaluation/campaign_summary.py`
- `src/autoresearch/reporting/export_tables.py`
- `scripts/summarize_campaign.py`

Current outputs:

- `main_campaign_summary.csv`
- `governance_metrics.csv`

Remaining work:

- add standalone `scripts/export_paper_tables.py`.
- add all KDD-required metrics that are not yet exported.
- add paper figure CSV exporters.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_2_3_integration_smoke.py
```

### Priority 4 — Memory Modes, Repeated-Bad Detection, and Dry-Run Ablation

Status: implemented as dry-run planning and summary.

Purpose:

Support the KDD memory/governance ablation modes.

Implemented files:

- `src/autoresearch/memory/summarizer.py`
- `src/autoresearch/memory/similarity.py`
- `src/autoresearch/evaluation/ablations.py`
- `scripts/run_memory_ablation.py`

Implemented memory modes:

- `none`
- `append_only_summary`
- `append_only_summary_with_rationale`

Current behavior:

- builds manager contexts for all memory modes.
- computes repeated-bad proposal metrics.
- exports a dry-run equal-budget memory ablation summary.

Remaining work:

- run real campaigns for each memory mode.
- feed each memory context into the real manager/worker loop.
- compute recovery-after-failure metrics from actual ablation runs.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_4_memory_and_manager_smoke.py
```

Direct dry-run check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
python3 scripts/run_memory_ablation.py \
  --node resnet_trigger \
  --budget 3 \
  --dry-run \
  --out /tmp/memory_ablation_summary.csv
```

### Priority 5 — Minimal Manager Interfaces

Status: implemented as proposal generators, not yet connected to a full campaign runner.

Purpose:

Make manager behavior swappable without allowing managers to own trial state.

Implemented files:

- `src/autoresearch/manager/base.py`
- `src/autoresearch/manager/baseline_manager.py`
- `src/autoresearch/manager/prompt_manager.py`

Current behavior:

- baseline manager returns deterministic bounded proposals.
- prompt manager returns memory-aware bounded proposals.
- both target only the node editable paths.

Remaining work:

- connect managers to a real fixed-budget campaign runner.
- add optional manager comparison only after the main campaign and memory ablation are working.
- keep LangGraph optional until the core pipeline is stable.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_4_memory_and_manager_smoke.py
```

## Next Implementation Priorities

### Priority 6 — Control-Plane Decision and Budget Modules

Status: implemented.

Purpose:

Move decision rules and budget accounting into explicit Stage 2 control-plane modules.

Implemented files:

- `src/autoresearch/control_plane/decision.py`
- `src/autoresearch/control_plane/budget.py`

`decision.py` owns:

- improved valid run -> `kept`
- valid degraded run -> `discarded`
- invalid scope, missing metric, runtime failure -> `failed_invalid`

`budget.py` owns:

- fixed trial budget
- current budget index
- cumulative budget consumed
- stop condition when budget is exhausted

Both modules are independent from any worker backend.

Acceptance criteria:

- decision logic is deterministic and unit-testable. ✅
- budget state rejects campaigns that exceed the configured trial budget. ✅
- lifecycle code can call decision and budget modules. ✅

Smoke test target:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_6_16_system_smoke.py
```

### Priority 7 — Generic Worker Interface

Purpose:

Prevent the project identity from collapsing into `claw-code`.

Add:

- `src/autoresearch/worker/base.py`
- `src/autoresearch/worker/claw_worker.py`
- optional later: `src/autoresearch/worker/local_worker.py`

Implementation instructions:

- define a worker protocol around:
  - apply/propose bounded change
  - run one experiment
  - collect changed files
  - collect logs/artifacts
  - return structured result
- `claw_worker.py` should wrap `ClawCodeAutoresearchAdapter`.
- do not move code from `harness/claw-code`.
- do not let worker code append authoritative trial state directly.

Acceptance criteria:

- worker returns structured output that the control plane can validate.
- changed files are exposed for editable-scope validation.
- worker interface can be mocked for dry-run campaign tests.

Smoke test target:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_7_worker_interface_smoke.py
```

### Priority 8 — Node Registry and Inspection CLI

Purpose:

Make benchmark contracts discoverable without reading implementation files.

Add:

- `src/autoresearch/nodes/registry.py`
- `scripts/inspect_node.py`

Implementation instructions:

- registry should resolve `resnet_trigger` to `configs/nodes/resnet_trigger.yaml`.
- `inspect_node.py --node resnet_trigger` should print the node spec as JSON.
- keep config source of truth clear. Either keep `configs/nodes/resnet_trigger.yaml`, or add a node-local copy and document the registry resolution.

Acceptance criteria:

- node can be inspected from CLI.
- output includes editable paths, frozen paths, metric, parser, acceptance rule, and budget.
- invalid node names fail cleanly.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
python3 scripts/inspect_node.py --node resnet_trigger
```

### Priority 9 — Fixed-Budget Campaign Runner

Status: dry-run implemented; real campaign runner not yet implemented.

Purpose:

Create the full Stage 2 control-plane executable path.

Implemented (dry-run):

- `scripts/run_campaign.py` — dry-run only; errors on non-dry-run.
- `src/autoresearch/control_plane/campaign.py::run_dry_campaign` — dry-run with DryRunWorker.
- `configs/campaigns/resnet_trigger_smoke.json` — smoke campaign config.
- `experiments/ledgers/`, `experiments/artifacts/`, `experiments/summaries/` — output dirs.

Remaining work (real path):

- Add `run_real_campaign()` to `campaign.py` that calls a real Worker (ClawWorker).
- Add a pending-trial guard (file lock) so crashes cannot leave orphaned pending state.
- Write real timestamps (start/end) per trial from datetime.now().
- Update `scripts/run_campaign.py` to accept `--node-root`, `--packet`, `--model`, `--host`
  and call ClawWorker when not `--dry-run`.
- Ensure `keep/discard` decision is always owned by Stage 2, not the legacy harness.

Acceptance criteria:

- dry-run writes a valid trial JSONL under `experiments/ledgers/`. ✅
- dry-run summary exports table CSVs. ✅
- one-iteration real smoke wraps the legacy loop and produces a Stage 2 trial record.
- pending-trial guard rejects a second trial start while one is active.
- no two pending trials can overwrite each other.

Smoke checks:

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

Then:

```bash
python3 scripts/summarize_campaign.py \
  --campaign-id smoke \
  --records experiments/ledgers/smoke_trials.jsonl
```

### Priority 10 — Complete Paper Metrics and Table Export

Purpose:

Make metrics complete enough for the KDD AAE paper.

Add:

- `scripts/export_paper_tables.py`
- additional fields in `src/autoresearch/evaluation/metrics.py`

Implementation instructions:

Add/export:

- gain per budget unit
- editable-scope violation count
- number of trials with recorded rationale
- milestone summary count
- wall-clock time per trial
- command failure rate
- metric parsing failure rate
- retry count
- artifact capture completeness
- number of generated tables
- reproducibility package completeness

Acceptance criteria:

- `scripts/export_paper_tables.py --campaign-id <id>` writes deterministic CSVs.
- metrics are computed only from trial records and artifacts, not manager state.
- empty or partial ledgers fail cleanly or report `null` where appropriate.

Expected outputs:

- `paper/tables/main_campaign_summary.csv`
- `paper/tables/memory_ablation_summary.csv`
- `paper/tables/governance_metrics.csv`

### Priority 11 — Paper Figure CSV Export

Purpose:

Export deterministic figure data before doing plotting.

Add:

- `scripts/export_paper_figures.py`
- `paper/figures/`

Implementation instructions:

Export CSVs:

- `paper/figures/campaign_trajectory.csv`
- `paper/figures/accepted_discarded_invalid_counts.csv`
- `paper/figures/repeated_bad_idea_rates.csv`
- `paper/figures/gain_per_budget_unit.csv`

Acceptance criteria:

- figure CSVs are reproducible from the trial ledger.
- no plotting dependency is required initially.
- each CSV has a stable schema documented in `docs/metrics.md`.

### Priority 12 — Real Memory/Governance Ablation Runner

Status: dry-run plan implemented; real equal-budget runner not yet wired.

Purpose:

Upgrade the current dry-run ablation into the main KDD experiment.

Implemented:

- `scripts/run_memory_ablation.py` — `--execute-dry-campaigns` calls `run_dry_campaign()` for each mode.
- `src/autoresearch/evaluation/ablations.py` — builds ablation plan rows.

Remaining work:

- After Priority 9 is stable, update `run_memory_ablation.py` to call `run_real_campaign()`
  for each of the three memory modes.
- Feed the actual memory context (built by `build_memory_context()`) into the manager before
  each real trial — this is the current gap even in dry-run mode (context is built but not
  passed to the manager's proposal text).
- Compute recovery-after-failed-run from actual records.
- Export `paper/tables/memory_ablation_summary.csv` from real ledgers.

Acceptance criteria:

- all three memory modes run under equal budget.
- output includes `paper/tables/memory_ablation_summary.csv`.
- each row links to the source campaign ledger.
- repeated-bad counts differ by mode if behavior differs.

Smoke check:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness
python3 scripts/run_memory_ablation.py \
  --node resnet_trigger \
  --budget 3 \
  --dry-run
```

Real execution should only be run after Priority 9 is stable.

### Priority 13 — Optional Manager Comparison

Purpose:

Compare manager strategies only after the main campaign and memory ablation work.

Add later:

- `scripts/run_manager_comparison.py`
- optional `src/autoresearch/manager/langgraph_manager_optional.py`

Implementation instructions:

- compare `baseline_manager` and `prompt_manager` under equal budget.
- use identical worker constraints and node spec.
- keep LangGraph optional.
- LangGraph manager, if added, must call the same control-plane APIs and produce the same trial schema.

Acceptance criteria:

- manager comparison does not bypass lifecycle, memory, or provenance contracts.
- results are exported as a secondary table, not the main paper claim.

### Priority 14 — Repository Packaging and Stable Layout

Purpose:

Make the repository easier to run and review as a research artifact.

Add:

- root `pyproject.toml`
- `configs/campaigns/`
- `configs/managers/`
- `configs/workers/`
- `experiments/runs/`
- `experiments/ledgers/`
- `experiments/summaries/`
- `experiments/artifacts/`
- `paper/figures/`
- `paper/tables/`
- `paper/notes/`

Implementation instructions:

- keep package dependencies minimal.
- avoid moving `harness/claw-code` until the Stage 2 wrapper path is stable.
- add `.gitkeep` files only where empty directories need to exist.

Acceptance criteria:

- package imports work from repo root.
- scripts can be run with documented commands.
- generated artifacts have predictable paths.

### Priority 15 — Documentation and Paper Notes

Purpose:

Make the repo support a KDD AAE workshop paper.

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
- `paper/notes/limitations.md`
- `paper/notes/related_work.md`

Implementation instructions:

- document what the project is and is not.
- state that the core contribution is the governed control plane.
- document the ResNet-trigger benchmark contract.
- document fixed-budget campaign reproduction.
- document memory ablation reproduction.
- document provenance from proposal to decision.

Acceptance criteria:

- a reader can understand the architecture in one page.
- a reader can reproduce one smoke campaign.
- a reader can reproduce the memory ablation path.
- contribution claims and non-claims are explicit.

### Priority 16 — Final Real-Loop Validation

Purpose:

Validate that Stage 2 is a real research artifact, not only a framework scaffold.

Run:

- one fixed-budget ResNet-trigger campaign using Stage 2 records
- one equal-budget memory/governance ablation
- paper table exports
- paper figure CSV exports

Acceptance criteria:

- every accepted result has complete provenance:
  - proposal
  - patch
  - scope check
  - run log
  - parsed metric
  - keep/discard decision
- final tables are generated from real ledgers.
- final figure CSVs are generated from real ledgers.
- the repo can credibly support the title:

> Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control

## Things To Avoid During Stage 2

Avoid until the above path is stable:

- frontend/dashboard work
- many benchmark domains
- cloud deployment
- broad multi-agent complexity
- full Hermes-style assistant shell
- tight LangGraph dependency
- tight `claw-code` internals dependency
- direct `ml-intern` or `multiautoresearch` integration
- overclaiming autonomous science

Reference projects can be discussed as related work or future backend inspiration, but they should not define the Stage 2 implementation.

## Current Development Smoke Commands

Run all current development smoke checks:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness

PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_0_1_contracts_smoke.py
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_2_3_integration_smoke.py
PYTHONPATH=$PWD/src python3 notebooks/stage2_development_smoke_tests/stage2_priority_4_memory_and_manager_smoke.py
```

Run legacy harness regression:

```bash
cd /Users/wongdowling/Documents/autoresearch_harness/harness/claw-code
PYTHONPATH=$PWD python3 -m unittest tests.test_autoresearch_integration
```
