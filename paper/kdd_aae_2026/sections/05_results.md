# 5. Results

Results are presented in governance-first order. The main campaign (`kdd_main_5trial`) uses a real `claw_style_worker` with `prompt_manager` on the ResNet-trigger node. All governance and task-metric numbers below are derived from the current ledgers in `experiments/ledgers/`. The full artifact suite is indexed in `artifact_manifest.json`.

## Main Campaign Governance

**Table 1: Main campaign governance and optimisation summary.**

| Field | Value |
|---|---|
| Table artifact | [`paper/tables/main_campaign_summary.csv`](../../tables/main_campaign_summary.csv) |
| Governance metrics | [`paper/tables/governance_metrics.csv`](../../tables/governance_metrics.csv) |
| Campaign | `kdd_main_5trial` |
| Node | `resnet_trigger` |
| Manager | `prompt_manager` |
| Worker | `claw_style_worker` |
| Memory mode | `append_only_summary_with_rationale` |
| Budget | 5 |
| Initial `val_auc` | 0.773778 |
| Best / final accepted `val_auc` | 0.774711 |
| Net gain | +0.000933 |
| Kept / discarded / failed-invalid | 2 / 0 / 3 |
| Acceptance rate | 0.40 |
| Invalid rate | 0.60 |
| Complete provenance rate | 1.00 |
| Artifact capture completeness | 0.40 (patch artifacts present for kept trials only; invalid trials have empty patch refs by design) |

**Per-trial summary:**

| Trial | Proposal | Decision | Failure | `val_auc` | Patch artifact |
|---:|---|---|---|---:|---|
| 1 | `lower-weight-decay` | `failed_invalid` | `runtime_error` | — | none |
| 2 | `small-dropout` | `kept` | — | 0.773778 | present (323 B) |
| 3 | `smaller-kernel` | `failed_invalid` | `runtime_error` | — | none |
| 4 | `lower-grad-clip` | `failed_invalid` | `runtime_error` | — | none |
| 5 | `larger-batch` | `kept` | — | 0.774711 | present (340 B) |

The campaign demonstrates lifecycle diversity: two valid improving edits are kept with full patch and provenance artifacts, and three runtime-invalid trials are classified and recorded as first-class audit objects. Provenance is complete for all five records. The campaign does not include a `discarded` valid-but-worse trial; full three-way lifecycle diversity would require a longer campaign or a valid non-improving trial.

## Failure Taxonomy Results

**Table 2 result artifact:** [`paper/tables/failure_taxonomy.csv`](../../tables/failure_taxonomy.csv).

| Campaign | Failure category | Count | Rate | Control-plane response |
|---|---|---:|---:|---|
| `kdd_main_5trial` | `runtime_error` | 3 | 0.60 | Mark `failed_invalid`, retain run artifact reference. |
| `lg_ablation_*` (all 3 arms) | `edit_failed` | 5 each | 1.00 | Mark `failed_invalid`; training ran on unmodified file; baseline metric not accepted as keep. |
| `kdd_stress_scope` | `invalid_edit_scope` | 1 | 1.00 | Reject forbidden-file patch; retain audit artifacts; git state unchanged. |
| `kdd_stress_noop` | `no_op_patch` | 1 | 1.00 | Mark `failed_invalid`, skip training. |

The main campaign's runtime errors represent proposals that changed source text but caused the training command to exit nonzero. The LangGraph ablation campaigns surface a distinct new failure mode: `edit_failed`, detailed below. The stress campaigns each exercise one forced failure path at 100% rate.

## Memory Ablation

Two ablation configurations are reported, differing in the manager backend.

### Arm A: `prompt_manager` (deterministic round-robin)

Three arms at equal budgets (5 trials each), differing only in the memory context injected into the proposal selector. Campaign IDs: `ablation_none`, `ablation_append_only_summary`, `ablation_append_only_summary_with_rationale`.

**Result: flat.** All three arms produced identical proposal sequences, outcomes, and final metrics (2 kept, 0 discarded, 3 failed-invalid, repeated_bad_rate = 0.40, best val_auc = 0.774711). The mechanistic explanation is in Section 4: the deterministic round-robin makes memory-based avoidance structurally redundant. This is a design-constrained negative finding — the memory subsystem and repeated-bad detector function correctly, but the proposal generator is insensitive to memory content.

### Arm A2: 10-trial `prompt_manager` extension

After the five-trial run, we ran an optional 10-trial extension under the same node, manager, worker, metric parser, and reset discipline. The longer budget exposes behaviour hidden by the shorter run.

| Memory mode | Budget | Kept | Discarded | Failed-invalid | Best `val_auc` | Repeated-bad rate |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 10 | 3 | 7 | 0 | 0.776756 | 0.60 |
| `append_only_summary` | 10 | 4 | 5 | 1 | 0.782733 | 0.40 |
| `append_only_summary_with_rationale` | 10 | 4 | 5 | 1 | 0.782733 | 0.40 |

The 10-trial extension shows a memory effect at the process level: both memory modes reduce repeated-bad rate from 0.60 to 0.40 and reach a higher best validation AUC than the no-memory arm. The two memory modes are tied, so the stronger pre-stated ordering is still not confirmed; specifically, rationale memory does not outperform summary-only memory in this run. The single failed-invalid trial in each memory arm is `proposal_precondition_failed`, meaning the state-aware structured selector could not find a remaining non-no-op effective edit at the current state. Evidence is recorded in `paper/tables/p8_memory10_summary.csv`.

### Arm B: `langgraph_manager` (stochastic LLM, `qwen2.5-coder:7b`, `temperature=0.7`, `budget=10`)

Three arms using the same node, budget (10), metric parser, acceptance rule, starting-state reset, and explicit avoidance prompt. Campaign IDs: `lg_ablation2_none`, `lg_ablation2_append_only_summary`, `lg_ablation2_append_only_summary_with_rationale`. All 30 trials used the deterministic patch bridge (`_extract_structured_edit`), routing through `claw_style_worker`'s constant-patch path without requiring a live AI coding agent. The prompt includes an AVOIDANCE RULE section (active only when memory is present) that instructs the LLM not to repeat parameter changes in the same direction that have already failed.

**Figure 2:** [`paper/figures/fig2_repeated_bad_rate.svg`](../../figures/fig2_repeated_bad_rate.svg) plots repeated-bad rate by memory mode (LangGraph arms).

**Table 3: LangGraph memory ablation v2 summary (`temperature=0.7`, `budget=10`, explicit avoidance prompt).**

| Memory mode | Manager | Budget | Kept | Discarded | Failed-invalid | Repeated-bad rate | Best `val_auc` | First switch |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `none` | `langgraph_manager` | 10 | 1 | 9 | 0 | 0.78 | 0.774711 | T2 (not sustained) |
| `append_only_summary` | `langgraph_manager` | 10 | 4 | 6 | 0 | 0.44 | 0.8212 | T3 (sustained) |
| `append_only_summary_with_rationale` | `langgraph_manager` | 10 | 1 | 9 | 0 | 0.78 | 0.774711 | T2 (not sustained) |

**Repeated-bad rate result.** The pre-stated hypothesis — repeated_bad_rate(none) > repeated_bad_rate(summary) > repeated_bad_rate(rationale) — is partially confirmed. The `append_only_summary` arm reduces repeated-bad rate from 0.78 to 0.44 relative to the no-memory baseline. However, the `append_only_summary_with_rationale` arm does not improve over the no-memory baseline: its repeated-bad rate equals that of the `none` arm (0.78). The full ordering is: none = rationale (0.78) > summary (0.44). This non-monotonic result — where the mid-complexity memory mode outperforms the richest mode — is itself an empirical finding.

**Parameter exploration.** The difference in governance outcomes is traceable to parameter-class exploration:

| Memory mode | Param sequence (T1–T10) | Unique params | First switch | Sustained? |
|---|---|---:|---|---|
| `none` | BS, TL, BS×7, TL, BS | 2 | T2 | No — reverts after one trial |
| `append_only_summary` | BS×2, LR, BS×2, TF×2, LR×2 | 3 | T3 | Yes — explores LR and TRAIN\_FRACTION |
| `append_only_summary_with_rationale` | TL, BS×8, DO, BS | 3 | T2 | No — reverts, then stays on BS |

The `append_only_summary` arm is the only one to explore TRAIN\_FRACTION (T6, T8), which produced the run's largest AUC gains (0.774711 → 0.803067 at T6; 0.803067 → 0.8212 at T8). The `none` and `with_rationale` arms both remain effectively stuck on BATCH\_SIZE changes, which produced no AUC change across all 9 BATCH\_SIZE trials in either arm (val\_auc = 0.774711 in every case). The AUC gap between the summary arm (best: 0.8212) and the other two arms (best: 0.774711) is therefore entirely attributable to parameter-class exploration driven by memory, not to batch-size tuning.

**Non-monotonic finding: why rationale does not help.** The `with_rationale` arm adds per-trial rationale text on top of the summary. Despite this richer context and the explicit avoidance rule in the prompt, the arm reverts to BATCH\_SIZE after T2 and continues proposing it in 8 of 10 trials. A plausible explanation is that verbose rationale text gives the LLM material to re-justify previously attempted parameter directions, partially overriding the avoidance instruction. This interaction between rationale verbosity and avoidance effectiveness is a testable hypothesis for future work.

**Full lifecycle diversity.** All three LangGraph arms produce both `kept` and `discarded` outcomes (zero `failed_invalid`), providing full evidence of the control plane's decision authority across valid improving and valid non-improving trial types.

A pre-bridge LangGraph run produced 15/15 `edit_failed` trials because `LangGraphManager` did not populate the `structured_edit` field required for the deterministic path. The control plane correctly refused to accept baseline metrics on unedited files as keeps in that run, so this remains failure-taxonomy evidence independent of the memory ablation.

## Decision Breakdown

**Figure 3:** [`paper/figures/fig3_decision_breakdown.svg`](../../figures/fig3_decision_breakdown.svg) shows kept, discarded, and failed-invalid counts across the main, ablation, and stress campaigns.

All eight campaigns in the current evidence set produce non-empty ledgers with no pending guard and complete provenance. The decision distribution is not collapsed into a single score; the governance claim is that the harness correctly records, classifies, and exposes all outcomes including novel failure modes.

## Governance in Action: What Breaks Without Controls

The stress campaigns (`kdd_stress_scope`, `kdd_stress_noop`) directly
demonstrate what the governance layer prevents. Table 5 contrasts the
observed governed outcome against the outcome a naive unguarded loop would
produce for each failure scenario.

**Table 5: Governance controls demonstrated by stress trials.**

| Scenario | Naive (unguarded) outcome | Governed outcome | Evidence campaign |
|---|---|---|---|
| **No-op patch** — worker produces a byte-identical change to `train.py` | Training runs on unmodified file; baseline metric accepted as a keep; spurious improvement recorded. | Training skipped; trial classified `failed_invalid / no_op_patch`; ledger records empty patch hash; git state unchanged. | `kdd_stress_noop` |
| **Out-of-scope edit** — worker touches `prepare.py` (frozen) alongside `train.py` | Forbidden file modified; node state corrupted; no record of the violation. | Patch rejected before training; trial classified `failed_invalid / invalid_edit_scope`; `changed_files: ["prepare.py"]` recorded in ledger; git state unchanged. | `kdd_stress_scope` |
| **Stale pending guard** — crash or kill between worker call and guard removal | Next campaign launch overwrites in-flight state; previous trial has no terminal ledger record; ledger integrity violated. | `PendingTrialError` raised on next launch; operator must resolve via `recover_pending.py`, which appends a `failed_invalid` record before clearing the guard. | (Recovery tooling; not a campaign ledger.) |

Each row is a silent failure in a naive loop that the harness makes visible and auditable. The no-op and scope-violation cases are backed by real ledger records from the stress campaigns. The pending-guard case is structural: the invariant is that every opened budget slot must have a terminal audit object, enforced in code before any new campaign can proceed.

This contrast is the answer to the question "why does governance matter?" An agent that maximises AUC by any means can produce any of the above scenarios. The governance layer ensures the paper-facing metrics reflect only trials that were validly executed, correctly scoped, and fully provenance-tracked.

**Table 4 result artifact:** [`paper/tables/provenance_chain.csv`](../../tables/provenance_chain.csv).

All five trials in `kdd_main_5trial` have complete provenance: proposal ID, patch ID, run ID, metric ID, and decision ID are present in every record. The artifact completeness checker [`paper/tables/artifact_completeness_report.txt`](../../tables/artifact_completeness_report.txt) reports 88/88 checks passed (100.0%) across all evidence campaigns. The manifest in [`artifact_manifest.json`](../../../artifact_manifest.json) indexes ledgers, tables, figures, and per-trial artifact references.

## Task Metric Trajectory

**Figure 4:** [`paper/figures/fig4_trajectory.svg`](../../figures/fig4_trajectory.svg) shows the main campaign `val_auc` trajectory.

AUC is secondary evidence that the loop runs correctly end-to-end, not the primary result. The campaign reports `val_auc` increasing from 0.773778 (T2) to 0.774711 (T5), a net gain of +0.000933 over five trials. This margin is small relative to baseline seed variation: five clean baseline runs over seeds 123-127 have mean `val_auc=0.784755`, population standard deviation 0.013837, and bootstrap 95% CI for the mean [0.774809, 0.798329]. Therefore we do not interpret the primary campaign's AUC change as evidence of meaningful optimisation. The central result is that lifecycle decisions, failures, provenance, and artifacts are jointly auditable for each trial — whether or not the task metric improves.
