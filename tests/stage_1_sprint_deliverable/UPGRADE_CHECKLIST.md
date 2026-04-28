# Stage 1 Sprint Upgrade Checklist

This checklist is based on `instruction.md` and the Stage 1 director report.

The sprint deliverable has two acceptance parts:

- Part 1: check that the harness and framework work as expected.
- Part 2: run a complete governed full loop on the `ResNet_trigger` node.

## 1. Make the Architecture Explicit

- Upgrade: document and enforce the separation between manager, control plane, bounded worker, experiment node, memory, and evaluation.
- Why meaningful: the paper needs a clean system description, and the repo should not read as a wrapper around `claw-code`.
- Implementation: add `docs/architecture.md`, move or wrap project-specific orchestration behind stable interfaces, and keep `harness/claw-code` as the worker substrate rather than the repo identity.

## 2. Formalize the Node Contract

- Upgrade: define a serializable node spec for `ResNet_trigger`.
- Why meaningful: this turns the current demo node into a reproducible benchmark node.
- Implementation: add a config such as `configs/nodes/resnet_trigger.json` with editable paths, setup command, run command, metric parser, objective direction, acceptance rule, validity checks, and default budget.

## 3. Formalize Trial Records

- Upgrade: replace ad hoc state fields with a documented append-only trial schema.
- Why meaningful: every paper claim about auditability depends on consistent trial records.
- Implementation: define a trial record containing trial id, campaign id, node id, manager mode, worker mode, proposal, targeted files, diff or commit, execution status, logs, parsed metric, keep/discard decision, rationale, runtime, and cumulative budget.

## 4. Standardize Keep/Discard Lifecycle

- Upgrade: make the state machine explicit: proposed, patched, executed, parsed, evaluated, kept, discarded, failed-invalid.
- Why meaningful: governance is the main contribution, so lifecycle transitions must be machine-readable and testable.
- Implementation: make the control plane own transitions and add tests for pending-run guards, crash handling, successful keep, discard rollback, and invalid worker edits.

## 5. Add Memory Modes for Ablation

- Upgrade: implement `none`, `append_only`, and `append_only_with_rationale` memory modes.
- Why meaningful: the required ablation tests whether memory and rationale reduce repeated bad proposals and improve acceptance efficiency.
- Implementation: add a top-level campaign config field, pass the selected memory view to the manager, and record memory mode in each trial.

## 6. Add Manager Baselines

- Upgrade: support at least a simple baseline manager and the current prompt-driven manager behind one manager interface.
- Why meaningful: this allows fair comparison under equal budget and avoids tying the framework to one planning style.
- Implementation: create `src/manager` with a common `propose_next_trial(status, memory)` contract and a minimal heuristic baseline.

## 7. Improve Metrics and Reporting

- Upgrade: compute governance and efficiency metrics, not only best AUC.
- Why meaningful: the KDD workshop framing needs measurable governance: accepted, discarded, invalid, repeated bad ideas, runtime, and gain per budget.
- Implementation: add summary export scripts that read append-only trials and write JSON, CSV, and Markdown tables for main campaigns and memory ablations.

## 8. Restore or Implement the Reported API Surface

- Upgrade: reconcile the director report REST endpoints (`/health`, `/status`, `/run`, `/keep`, `/discard`, `/memory`, `/loop`) with the current CLI-only autoresearch path.
- Why meaningful: the report describes a control-plane service boundary; the current code can run the loop, but the service API is not exposed by `src.main`.
- Implementation: either add the API server command or update the report/docs to specify the CLI as the current control-plane interface. Add parity tests so API and CLI produce equivalent trial records.
