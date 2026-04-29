# Stage 2 Instruction — KDD AAE-Aligned Autoresearch Harness

## Objective

Refactor and extend `autoresearch_harness` from a successful Stage 1 prototype into a publishable Stage 2 research artifact for a KDD Agentic AI Evaluation workshop submission.

The goal is **not** to build a general autonomous scientist, a flashy multi-agent demo, or a wrapper around an existing coding agent.

The goal is to build and evaluate a **governed autonomous experimentation framework** with:

- bounded execution
- explicit trial lifecycle control
- append-only memory
- reproducible experiment ledgers
- measurable governance metrics
- fixed-budget evaluation
- at least one memory/governance ablation

The paper-facing claim should be:

> Autonomous experimentation becomes scientifically useful only when the agent loop is governed by explicit lifecycle control, bounded execution, durable memory, and auditable decision records.

---

## Core framing

Position the repository as:

**A governed autonomous experimentation framework for evaluating agentic optimization under bounded execution constraints.**

Alternative title:

**Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control.**

The repository should not present itself as:

- a general AI scientist
- a LangChain/LangGraph demo
- a claw-code wrapper
- a generic coding-agent benchmark
- a collection of experiment scripts
- a broad AI-for-science platform yet

The intellectual contribution is the **governed experiment control plane** and its evaluation.

---

## Stage 2 target

Stage 1 already demonstrated a 3 × 5 governed campaign on a ResNet-trigger node.

Stage 2 should produce:

1. a cleaner framework implementation
2. one stable benchmark node
3. one fixed-budget main campaign
4. one memory/governance ablation
5. paper-ready metrics, tables, and plots
6. clear documentation of architecture and evaluation protocol

Do not prioritize adding many domains, many agents, a dashboard, cloud deployment, or deep LangGraph integration until these are complete.

---

## System architecture

Refactor the system around six explicit layers.

### 1. Manager / Planner

Responsible for:

- proposing candidate experiment modifications
- reading permitted memory summaries
- selecting the next trial under a fixed budget
- producing a structured proposal
- issuing bounded objectives to the control plane

The manager should be swappable:

- `baseline_manager`: simple heuristic or minimal prompt baseline
- `prompt_manager`: current stronger prompt-driven manager
- `langgraph_manager`: optional later backend, not required for the core system

LangGraph may be used later for clearer stateful orchestration, but it must not replace the control plane.

### 2. Control Plane

This is the core contribution.

Responsible for:

- campaign lifecycle
- trial state machine
- budget accounting
- run/keep/discard enforcement
- invalid-run handling
- editable-scope enforcement
- append-only authoritative record writing
- experiment ledger consistency

The control plane owns state transitions. Managers and workers may request actions, but only the control plane commits state.

### 3. Bounded Worker Runtime

Responsible for:

- applying limited code changes
- respecting editable-path allowlists
- running the experiment command
- collecting logs and artifacts
- returning structured results
- never committing authoritative state directly

A claw-style worker can be used as the local runtime substrate, but the repo identity must not depend on claw internals.

Define a generic worker interface so future backends can be added:

- local Qwen/Ollama worker
- claw-style worker
- OpenCode-style worker
- Slurm worker
- Hugging Face Jobs worker

Only the first one or two are needed for Stage 2.

### 4. Experiment Node

Responsible for defining a concrete task environment.

The primary Stage 2 node should remain the existing scientific ML / ResNet-trigger optimization task.

Each node must declare:

- node name
- description
- editable paths
- frozen paths
- setup command
- run command
- metric name
- metric direction
- metric parser
- acceptance rule
- validity checks
- default budget
- expected runtime
- failure categories

Do not add additional benchmark domains until this node is stable and reproducible.

### 5. Memory / Audit Layer

Responsible for:

- append-only trial records
- proposal summaries
- patch references
- metric records
- keep/discard decisions
- decision rationales
- failure causes
- milestone summaries
- compressed manager context
- provenance links

Memory should serve two purposes:

1. **auditability**: preserve raw records for replay and inspection
2. **agent context**: provide compressed summaries to the manager

Do not feed raw unbounded history to the manager by default.

### 6. Evaluation / Reporting Layer

Responsible for computing paper-facing metrics and exporting:

- campaign summary tables
- memory ablation tables
- trajectory plot data
- accepted/discarded/invalid counts
- repeated-bad-proposal statistics
- wall-clock and budget-efficiency metrics
- provenance-completeness metrics

This layer should be deterministic and independent from the manager.

---

## Required repository structure

Refactor toward this structure:

```text
autoresearch_harness/
  README.md
  pyproject.toml
  configs/
    campaigns/
    managers/
    workers/
    nodes/
  docs/
    architecture.md
    node_spec.md
    trial_schema.md
    experiment_protocol.md
    metrics.md
    memory_architecture.md
    provenance.md
  paper/
    figures/
    tables/
    notes/
      contribution_claims.md
      limitations.md
      related_work.md
  src/
    autoresearch/
      manager/
        base.py
        baseline_manager.py
        prompt_manager.py
        langgraph_manager_optional.py
      control_plane/
        state_machine.py
        lifecycle.py
        budget.py
        permissions.py
        decision.py
      worker/
        base.py
        local_worker.py
        claw_worker.py
      nodes/
        spec.py
        registry.py
        resnet_trigger/
          node.yaml
          metric_parser.py
          validity.py
      memory/
        schemas.py
        append_store.py
        summarizer.py
        provenance.py
        similarity.py
      evaluation/
        metrics.py
        ablations.py
        campaign_summary.py
      reporting/
        export_tables.py
        export_figures.py
        write_report.py
      common/
        types.py
        logging.py
        paths.py
  experiments/
    runs/
    ledgers/
    summaries/
    artifacts/
  scripts/
    run_campaign.py
    run_memory_ablation.py
    summarize_campaign.py
    export_paper_tables.py
    export_paper_figures.py
  tests/
    test_node_spec.py
    test_trial_schema.py
    test_state_machine.py
    test_permissions.py
    test_append_only_memory.py
    test_metric_parser.py
    test_repeated_bad_idea_detection.py
```

This exact layout may be adjusted, but the boundaries must remain clear.

---

## Formal node specification

Create a serializable node specification, preferably YAML plus a typed Python schema.

Minimum fields:

```yaml
name: resnet_trigger
description: Near-threshold detector waveform binary-classification benchmark
editable_paths:
  - train.py
frozen_paths:
  - prepare.py
  - data/
setup_command: "python prepare.py"
run_command: "python train.py --config config.yaml"
metric_name: val_auc
metric_direction: maximize
metric_parser: "metric_parser.py:parse_val_auc"
acceptance_rule: "candidate_metric > current_best_metric"
validity_checks:
  - metric_present
  - finite_metric
  - editable_scope_only
  - no_data_pipeline_modification
  - command_exit_zero
default_budget:
  trials: 50
  max_wall_clock_hours: 12
failure_categories:
  - syntax_error
  - runtime_error
  - metric_missing
  - invalid_edit_scope
  - degraded_metric
```

The node spec should be inspectable from CLI:

```bash
python scripts/inspect_node.py --node resnet_trigger
```

---

## Formal trial record schema

Every trial must produce an append-only structured record.

Minimum fields:

```json
{
  "trial_id": "...",
  "campaign_id": "...",
  "node_id": "resnet_trigger",
  "budget_index": 7,
  "timestamp_start": "...",
  "timestamp_end": "...",
  "manager_mode": "prompt_manager",
  "worker_mode": "local_worker",
  "memory_mode": "append_only_with_rationale",
  "proposal_summary": "...",
  "proposal_rationale": "...",
  "targeted_files": ["train.py"],
  "patch_ref": "experiments/artifacts/.../patch.diff",
  "git_commit_before": "...",
  "git_commit_after": "...",
  "execution_status": "success | failed",
  "validity_status": "valid | invalid",
  "failure_category": null,
  "raw_log_ref": "experiments/artifacts/.../run.log",
  "parsed_metrics": {"val_auc": 0.7876},
  "current_best_before": 0.7841,
  "delta_vs_best": 0.0035,
  "decision": "kept | discarded | failed_invalid",
  "decision_rationale": "...",
  "wall_clock_seconds": 1234.5,
  "cumulative_budget_consumed": 7,
  "provenance": {
    "proposal_id": "...",
    "patch_id": "...",
    "run_id": "...",
    "metric_id": "...",
    "decision_id": "..."
  }
}
```

Records must be append-only. Do not overwrite old records when updating summaries.

---

## Trial lifecycle

Implement a machine-readable state machine:

```text
initialized
  -> proposed
  -> patch_generated
  -> scope_validated
  -> executed
  -> metric_parsed
  -> evaluated
  -> kept | discarded | failed_invalid
```

Rules:

- A trial cannot execute before editable-scope validation.
- A trial cannot be kept/discarded before metric parsing.
- Invalid edits must become `failed_invalid`.
- Broken runs must be categorized and logged.
- A campaign must not allow two active pending trials to overwrite each other.
- Only the control plane writes authoritative lifecycle state.

---

## Memory modes for ablation

Implement exactly these Stage 2 memory modes first.

### `memory_mode = none`

The manager receives only:

- current baseline
- current budget index
- node spec
- allowed edit scope

No prior trial summaries are provided.

### `memory_mode = append_only_summary`

The manager receives compressed summaries of prior trials:

- proposal summary
- changed parameters
- metric delta
- kept/discarded/invalid status

No detailed rationale is included.

### `memory_mode = append_only_summary_with_rationale`

The manager receives the above plus:

- keep/discard rationale
- failure categories
- repeated-bad-idea warnings
- current best strategy summary

These three modes define the primary KDD AAE ablation.

---

## Repeated-bad-idea detection

Implement a lightweight detector for repeated bad proposals.

A proposal should be flagged as repeated-bad if it is similar to an earlier discarded or invalid proposal by any of:

- same targeted parameter and same direction
- same failure category and similar proposal summary
- high textual similarity to prior discarded proposal
- same patch signature class

For Stage 2, approximate methods are acceptable:

- normalized proposal text similarity
- regex extraction of hyperparameter names
- targeted-file overlap
- failure-category matching

Report:

- repeated bad proposal count
- repeated bad proposal rate
- repeated invalid proposal count
- repeated degraded proposal count

This metric is central to the memory ablation.

---

## Main Stage 2 experiments

### Experiment 1 — governed main campaign

Run one fixed-budget governed campaign on the ResNet-trigger node.

Recommended budget:

- minimum: 3 rounds × 5 trials, matching Stage 1
- target: 5 rounds × 10 trials if runtime allows

Report:

- initial metric
- best metric
- final accepted metric
- net improvement
- total trials
- kept / discarded / invalid counts
- acceptance rate
- invalid rate
- wall-clock time
- gain per trial
- gain per hour
- best accepted modification summary
- complete provenance rate

### Experiment 2 — memory/governance ablation

Compare under equal budget:

1. `memory_mode = none`
2. `memory_mode = append_only_summary`
3. `memory_mode = append_only_summary_with_rationale`

Report:

- best metric
- final accepted metric
- acceptance rate
- invalid rate
- repeated bad proposal count
- recovery after failed run
- gain per trial
- gain per hour
- context length fed to manager
- compression ratio from raw memory to manager context

This is the most important Stage 2 paper experiment.

### Experiment 3 — optional manager comparison

Only if Experiments 1 and 2 are complete.

Compare:

- baseline manager
- prompt manager
- optional LangGraph manager

under equal budget and identical worker constraints.

Do not make this required for the first KDD AAE submission unless time permits.

---

## Metrics to implement

### Optimization metrics

- initial metric
- best metric
- final accepted metric
- net gain
- gain per trial
- gain per accepted trial
- gain per wall-clock hour
- gain per budget unit

### Governance metrics

- total proposals
- kept proposals
- discarded proposals
- failed-invalid proposals
- acceptance rate
- invalid rate
- editable-scope violation count
- complete-provenance rate
- number of trials with recorded rationale

### Memory metrics

- repeated bad proposal count
- repeated bad proposal rate
- repeated invalid proposal count
- repeated degraded proposal count
- manager context length
- raw memory size
- compressed summary size
- compression ratio
- milestone summary count

### Runtime and reliability metrics

- wall-clock time per trial
- total wall-clock time
- command failure rate
- metric parsing failure rate
- retry count if retries are implemented
- artifact capture completeness

### Reporting metrics

- number of generated tables
- number of generated plot-data files
- reproducibility package completeness

---

## Paper-ready outputs

Implement scripts that export deterministic paper artifacts:

```bash
python scripts/summarize_campaign.py --campaign-id <id>
python scripts/run_memory_ablation.py --node resnet_trigger --budget 15
python scripts/export_paper_tables.py --campaign-id <id>
python scripts/export_paper_figures.py --campaign-id <id>
```

Outputs:

```text
paper/tables/main_campaign_summary.csv
paper/tables/memory_ablation_summary.csv
paper/tables/governance_metrics.csv
paper/figures/campaign_trajectory.csv
paper/figures/accepted_discarded_invalid_counts.csv
paper/figures/repeated_bad_idea_rates.csv
paper/figures/gain_per_budget_unit.csv
```

The figure scripts may export CSV first; plotting can be done later.

---

## Documentation deliverables

Create or rewrite these documents.

### `README.md`

Must clearly state:

- what the project is
- what it is not
- core architecture
- current benchmark
- Stage 1 result
- Stage 2 plan
- how to reproduce one campaign
- how to reproduce the memory ablation

### `docs/architecture.md`

Explain:

- manager
- control plane
- worker runtime
- experiment node
- memory/audit layer
- evaluation/reporting layer
- control flow diagram

### `docs/node_spec.md`

Define the node contract and include the ResNet-trigger example.

### `docs/trial_schema.md`

Define the trial record schema and lifecycle states.

### `docs/experiment_protocol.md`

Define:

- fixed budget
- metric direction
- acceptance rule
- validity criteria
- repeated-bad-idea definition
- ablation design

### `docs/metrics.md`

Define every reported metric precisely.

### `docs/memory_architecture.md`

Explain:

- raw append-only memory
- milestone summaries
- manager-fed compressed context
- memory ablation modes

### `docs/provenance.md`

Explain how a final accepted result traces back to:

- proposal
- patch
- scope check
- run log
- parsed metric
- keep/discard decision

### `paper/notes/contribution_claims.md`

State what the paper claims and does not claim.

Claims:

- bounded execution improves auditability
- explicit lifecycle control makes autonomous experiments inspectable
- memory and rationale can reduce repeated poor proposals under fixed budget
- governed experimentation can produce measurable improvements on a real scientific ML node

Non-claims:

- not a general autonomous scientist
- not proof of scientific discovery
- not a universal optimization algorithm
- not dependent on a specific coding agent backend

---

## Implementation priority order

Follow this order strictly.

### Priority 1 — Contracts and lifecycle

Implement:

- node spec
- trial schema
- state machine
- append-only trial store
- editable-scope validation
- metric parser contract

### Priority 2 — Stable benchmark campaign

Implement:

- ResNet-trigger node spec
- fixed-budget campaign runner
- deterministic metric extraction
- main campaign summary
- reproducible artifact paths

### Priority 3 — Paper metrics and reporting

Implement:

- governance metrics
- optimization metrics
- runtime metrics
- repeated-bad-idea detector
- table/figure CSV exporters

### Priority 4 — Memory ablation

Implement:

- `none`
- `append_only_summary`
- `append_only_summary_with_rationale`
- equal-budget ablation runner
- ablation summary table

### Priority 5 — Minimal manager baseline comparison

Implement:

- simple baseline manager
- current prompt manager

Only compare managers if the memory ablation is already complete.

### Priority 6 — Optional LangGraph integration

Add LangGraph only as an optional manager backend after the core framework and ablation are stable.

The LangGraph implementation should call the same control-plane APIs and produce the same trial records. It must not bypass lifecycle or memory contracts.

---

## What to avoid in Stage 2

Avoid:

- frontend/dashboard work
- many benchmark domains
- cloud deployment
- broad multi-agent complexity
- full Hermes-style assistant shell
- tight dependence on LangChain/LangGraph
- tight dependence on claw internals
- integrating multiautoresearch or ml-intern directly
- overclaiming autonomous science
- adding features that do not improve evaluation, auditability, or paper clarity

Reference projects can be discussed as related work or future backend inspiration, but they should not define the Stage 2 implementation.

---

## Success criteria

Stage 2 is successful if the repository can support:

1. a one-page architecture description
2. one reproducible fixed-budget campaign on the ResNet-trigger node
3. one memory/governance ablation under equal budget
4. paper-ready tables for optimization, governance, and memory metrics
5. a clear explanation of why bounded execution and explicit keep/discard control matter
6. complete provenance for every accepted result
7. a conservative KDD AAE framing around evaluation and governance

A good final test:

> Could this repository credibly support a workshop paper titled “Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control”?

If yes, the Stage 2 refinement is aligned.

