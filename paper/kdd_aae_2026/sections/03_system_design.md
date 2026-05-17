# 3. Governed Control Plane

The system is a governed harness for autonomous ML experimentation. The manager proposes an experimental change; the worker materialises a patch and run artifacts; the control plane validates scope, executes the lifecycle, parses metrics, decides keep/discard/failed-invalid, updates memory, and appends the authoritative ledger record. Managers and workers are replaceable, but managers cannot commit trial state.

## Manager/Worker Separation and Replacement Principle

The manager is responsible for proposal generation: a summary, rationale, target file, and intended experimental effect. The worker is responsible for bounded execution: turning that proposal into a patch, running the configured command, and returning artifact references. The control plane is responsible for authority: validity, state transitions, budget accounting, memory updates, provenance IDs, and final decision.

This separation is the replacement principle. A baseline heuristic manager, prompt manager, LangGraph manager, AIDE-style tree-search manager, local worker, Claw-style worker, or future cloud worker can be swapped in without changing the paper-facing governance contract. The append-only ledger remains the source of truth.

**Worker replication paths.** Two worker implementations cover the full reproducibility spectrum:

- `LocalWorker` (`local_worker.py`): parses "Change PARAM from X to Y" directives from proposal text and applies edits directly with no external dependencies. Suitable for protocol verification without a running LLM or coding agent.
- `ClawWorker` (`claw_worker.py`): generates a packet JSON and invokes the legacy claw-harness loop, which requires a live AI coding agent (such as Claude Code) for free-form LLM proposals. For structured proposals, `ClawWorker` routes through a deterministic constant-patch path that requires only Ollama, not a coding agent.

Both workers implement the same `Worker` protocol and produce ledger records under the same schema; the control plane and paper-facing governance contract are identical regardless of which worker is used.

## Trial Lifecycle

Each trial follows a fixed lifecycle:

```mermaid
flowchart LR
    A["Budget slot opened"] --> B["Manager proposes change"]
    B --> C["Pending guard written"]
    C --> D["Worker produces patch and log"]
    D --> E["Scope and no-op validation"]
    E --> F["Training command and metric parser"]
    F --> G["Control-plane decision"]
    G --> H["Append-only ledger record"]
    H --> I["Memory update and guard removal"]
```

In pseudocode:

```text
for budget_index in campaign_budget:
    proposal = manager.propose(memory_context, node_spec)
    write_pending_guard(campaign_id, trial_id, proposal)
    worker_result = worker.run(proposal, node_spec)
    validity = validate_patch_scope_and_effect(worker_result.patch_ref, node_spec)
    if not validity.ok:
        decision = failed_invalid(validity.failure_category)
    else:
        metrics = parse_metric(worker_result.raw_log_ref)
        decision = keep_if_metric_improves(metrics, current_best)
    append_trial_record(decision, provenance, artifacts)
    update_memory(decision)
    remove_pending_guard()
```

## Scope Enforcement

The ResNet-trigger `NodeSpec` declares `train.py` as the only editable file. Frozen files include `prepare.py`, `program.md`, node dependencies, data files, and artifact directories. The scope validator checks the generated patch before the candidate can be accepted. A patch touching a frozen path is marked `failed_invalid / invalid_edit_scope`, written to the ledger, and prevented from corrupting node state.

## Pending-Trial Guard

Before worker execution, the control plane writes a pending guard under `experiments/ledgers/`. The guard makes mid-trial crashes auditable instead of invisible. Recovery tooling can list, inspect, and resolve stale guards by appending a failed-invalid ledger record before deleting the guard. This preserves the invariant that every opened budget slot has a terminal audit object.

## Append-Only Ledger

The ledger is an append-only JSONL file per campaign. A trial record contains campaign identity, budget index, proposal summary and rationale, patch and raw-log references, parsed metrics, validity status, decision, failure category, provenance IDs, reproducibility hashes, and timing. The ledger is the durable system of record; generated tables and figures are derived artifacts.

## Memory Modes

The memory interface has three modes:

| Mode | Manager context |
|---|---|
| `none` | No prior-trial context is injected. |
| `append_only_summary` | Prior trial summaries, decisions, metrics, and failure categories are injected. |
| `append_only_summary_with_rationale` | The summary context is augmented with decision rationales, giving the manager targeted failure signals. |

The ablation asks whether rationale-linked memory reduces repeated poor proposals more than raw history alone.

## Decision Authority

We keep the keep/discard decision in the control plane, not delegated to the manager. Rajasekaran et al. (2026) found that LLM self-evaluation is systematically lenient -- agents confidently praise mediocre work. A deterministic, externally-owned decision criterion addresses this.

The decision rule is intentionally simple: a valid candidate is kept only if the configured metric improves over the current best in the configured direction. A valid but non-improving candidate is `discarded / degraded_metric`; an invalid candidate is `failed_invalid` with a specific failure category. The manager may explain a proposal, but it cannot declare it successful.

## Guides and Sensors

**Table S1: Harness guides and sensors for autonomous ML experimentation.**

| | **Guide (feedforward -- steers before act)** | **Sensor (feedback -- corrects after act)** |
|---|---|---|
| **Computational** (deterministic, fast) | Node spec defining valid scope; editable-path whitelist; budget cap; scope constraints | Metric parser (`val_bpb` to `val_auc`); state machine enforcing transition legality; pending-trial guard detecting crashes |
| **Inferential** (semantic, LLM-based) | Manager system prompt; memory injection; proposal rationale template | Repeated-bad detector using edit-target and mechanism similarity to flag redundant proposals |

This table instantiates Böckeler's guides/sensors taxonomy in the ML experimentation setting.

## Failure Mode Motivation

**Table S2: Agent failure modes and harness controls.**

| Agent failure mode | ML experimentation analogue | Harness fix |
|---|---|---|
| Declares victory too early | Manager stops after one apparently good trial | Budget enforcer plus fixed trial count |
| Leaves broken state | Failed patch leaves repository or node dirty | Pending-trial guard, git before/after checks, reset script |
| Marks feature done prematurely | Manager claims improvement without parseable metric | `FAILED_INVALID` state plus metric parser |
| Does not know how to run the app | Manager proposes an out-of-scope edit | Node spec plus editable-path whitelist |

The design treats these failures as expected lifecycle outcomes. Failed trials are audit records, not hidden exceptions.
