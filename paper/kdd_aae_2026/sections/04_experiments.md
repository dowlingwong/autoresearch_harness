# 4. Evaluation Protocol

The evaluation asks whether an autonomous ML experimentation loop is bounded, auditable, failure-aware, reproducible, and behaviorally affected by governance memory. The task metric is still reported, but governance metrics are primary.

## Benchmark Node

The benchmark is the ResNet-trigger node (`resnet_trigger`), a real scientific ML task for near-threshold detector waveform binary classification. The node trains a ResNet-style classifier on signal and noise traces. The editable surface is deliberately narrow: workers may modify only `nodes/ResNet_trigger/train.py`, while data loading, split logic, node metadata, dependency files, and artifacts are frozen.

The scientific task metric is validation AUC, exposed to the control plane as `val_auc` with `metric_direction=maximize`. The node also prints a compatibility scalar `val_bpb = 1 - best_val_auc` for older worker paths, but the paper reports AUC directly.

## Fixed-Budget Campaign Protocol

Each campaign starts from a reset node state and a fixed budget. The canonical five-trial campaign uses:

| Field | Value |
|---|---|
| Node | `resnet_trigger` |
| Campaign | `kdd_main_5trial` |
| Budget | 5 trials |
| Manager | `baseline_manager` in current dry-run artifacts; prompt-manager support is part of the same control-plane interface |
| Memory mode | `append_only_summary_with_rationale` |
| Worker | `DryRunWorker` for governance-contract artifacts; Claw-style worker for real execution |
| Acceptance rule | keep iff candidate `val_auc` improves over current best |

For every trial, the control plane records the proposal, patch reference, raw log reference, parsed metrics, validity status, failure category if any, decision rationale, provenance IDs, and reproducibility hashes. A valid improving trial is kept; a valid non-improving trial is discarded; an invalid trial is failed invalid.

## Memory Ablation Design

The memory ablation runs equal budgets under three conditions: `none`, `append_only_summary`, and `append_only_summary_with_rationale`. Each arm uses the same node, budget, metric parser, acceptance rule, editable scope, and starting-state reset. The primary outcome is repeated-bad rate, computed as:

```text
RepeatedBadRate = repeated_bad_proposals / total_proposals
```

A repeated-bad proposal is one that repeats the same edit target and edit mechanism as a prior rejected, invalid, or degraded trial without adding a new corrective rationale, constraint, or justification.

The hypothesis is pre-stated before results:

```text
repeated_bad_rate(none) > repeated_bad_rate(append_only_summary)
                        > repeated_bad_rate(append_only_summary_with_rationale)
```

Rationale-augmented memory should reduce repeated-bad rate more than raw summaries because it gives the manager a targeted failure signal rather than history noise. This hypothesis is grounded in OpenAI's map-not-manual framing and Böckeler's combination of feedforward guides with feedback sensors. If the result is flat or reversed, it is a valid negative finding and is reported as such.

## Stress Trial

The stress trial exercises invalid edit-scope handling. A dedicated scope-violation worker generates a patch that touches a forbidden file. The expected control-plane response is:

```text
worker attempts forbidden edit
  -> scope validator rejects patch
  -> trial is marked failed_invalid / invalid_edit_scope
  -> patch and raw log references are retained
  -> git state remains unchanged
  -> pending guard is removed
```

This trial demonstrates that invalid actions are first-class audit objects rather than missing data.

## Governance Metrics

The evaluation reports the following governance metrics:

| Metric | Definition |
|---|---|
| Lifecycle counts | Counts of `kept`, `discarded`, and `failed_invalid` trials. |
| Acceptance rate | `kept / total_trials`. |
| Invalid rate | `failed_invalid / total_trials`. |
| Repeated-bad rate | Repeated-bad proposals divided by total proposals. |
| Complete provenance rate | Fraction of trial records with proposal, patch, run, metric, and decision IDs. |
| Artifact capture completeness | Fraction of trial records with captured patch and raw-log artifact references. |
| Scope-violation count | Number of trials rejected for editing frozen files. |
| Metric parsing failure rate | Fraction of trials where the expected metric is unavailable. |
| Gain per budget unit | Net task-metric gain divided by fixed budget. |

## Failure Taxonomy

**Table 2: Failure taxonomy for autonomous ML experimentation.**

| Category | Definition | Control-plane response |
|---|---|---|
| `invalid_edit_scope` | Patch touched a disallowed file or region. | Mark `failed_invalid`, retain patch reference, do not commit state. |
| `syntax_error` | Code cannot be parsed or imported. | Mark `failed_invalid`, retain logs, do not keep patch. |
| `runtime_error` | Training command exits nonzero. | Mark `failed_invalid`, retain raw log. |
| `metric_missing` | Run completes but expected metric cannot be parsed. | Mark `failed_invalid`, retain run artifact. |
| `degraded_metric` | Valid run is worse than current best. | Mark `discarded`, revert or avoid committing candidate state. |
| `no_op_patch` | Worker produced no effective code change. | Mark `failed_invalid`, skip training execution. |

The taxonomy is included in the main experiment section so the reader can interpret failures before seeing results.
