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
| Manager | `prompt_manager` with structured state-aware hyperparameter proposals |
| Memory mode | `append_only_summary_with_rationale` |
| Worker | `claw_style_worker` with deterministic patch path for constant edits |
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

**Two ablation configurations are reported.** The first uses `prompt_manager`, which delegates to `select_structured_hyperparameter_edit`, a deterministic round-robin indexed on `budget_index`. Memory context is injected correctly and differs in content across modes, but the selector's determinism means it cannot produce a different proposal sequence regardless of memory content. This arm is reported as a design-constrained negative finding.

The second ablation uses `LangGraphManager` (`langgraph_manager`), which calls `qwen2.5-coder:7b` via Ollama at `temperature=0.2`. The full formatted memory context is passed as part of the prompt and a free-form JSON proposal is parsed. This is a genuinely stochastic backend where memory context can in principle influence proposal selection. Campaign IDs: `lg_ablation_none`, `lg_ablation_append_only_summary`, `lg_ablation_append_only_summary_with_rationale`. Memory injection is verified per-trial by recording the SHA-256 of the formatted context in the ledger `extra` field; all 15 context hashes are distinct, confirming that memory accumulates correctly across trials and differs across modes.

A secondary failure mode emerges in the LangGraph ablation: `edit_failed`. The `claw_style_worker` requires a live AI coding agent (such as Claude Code or a claw-harness agent) to translate free-form LLM proposals into file patches. Without an agent present, training executes against the unmodified source and the worker reports "worker did not modify train.py." The control plane correctly classifies these trials as `failed_invalid / edit_failed` — it does not accept a baseline metric obtained on an unedited file as a keep. This demonstrates the governance layer's integrity under a new failure mode not present in the main campaign.

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

Governance metrics are the primary evaluation criterion in this work. They
measure the reliability of the experimentation process itself — how many
trials were valid, how many failures were classified and recorded, whether
provenance is complete, and whether the agent repeated known-bad proposals.
A campaign can show a high task metric while simultaneously hiding failures,
producing unauditable decisions, or running on corrupted state; governance
metrics make these behaviors visible. Task-metric AUC is reported as
secondary evidence that the loop runs correctly end-to-end.

**Definition 1 (Governance Metrics Suite).**
Given a fixed-budget campaign of $N$ trials, the following metrics
characterise process reliability:

| Metric | Formal definition | What it detects |
|---|---|---|
| **Acceptance rate** | $\lvert\text{kept}\rvert / N$ | Whether the agent produces valid improving edits at all. |
| **Invalid rate** | $\lvert\text{failed\_invalid}\rvert / N$ | Proportion of trials that could not be evaluated. |
| **Repeated-bad rate** | Repeated-bad proposals $/ N$ | Whether the agent avoids redundant failure modes across trials. |
| **Complete provenance rate** | Trials with all five IDs (proposal, patch, run, metric, decision) $/ N$ | Whether every trial is independently reproducible. |
| **Artifact capture completeness** | Trials with captured patch and log refs $/ N$ | Whether physical evidence is retained for every trial. |
| **Failure taxonomy** | Counts per `FailureCategory` | Whether failures are classified, not silently discarded. |
| **Scope-violation count** | Trials rejected for editing frozen files | Whether the agent respects editable-scope constraints. |
| **Gain per budget unit** | Net task-metric gain $/ N$ | Secondary: whether governance overhead impedes improvement. |

A well-governed campaign is not necessarily one with high acceptance rate. A
campaign with 60% invalid rate but 100% provenance completeness, correct
failure taxonomy, and no scope violations provides more interpretable evidence
than one with 100% acceptance rate and no audit trail.

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
| `edit_failed` | Worker ran but did not modify the target file (e.g., no AI coding agent present to apply a free-form LLM proposal). | Mark `failed_invalid`, retain run artifact; do not accept baseline metric as a keep. |

The taxonomy is included in the main experiment section so the reader can interpret failures before seeing results.
