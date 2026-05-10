# 5. Results

Results are presented in governance-first order. The fixed-budget KDD artifacts for the main campaign and memory ablation are dry-run governance-contract ledgers; the stress trial records a real scope-validation failure artifact. The earlier real ResNet full-loop accepted edit is reported last as secondary task evidence.

## Main Campaign Governance

**Table 1: Main campaign governance and optimisation summary.**

| Field | Value |
|---|---|
| Table artifact | [`paper/tables/main_campaign_summary.csv`](../../tables/main_campaign_summary.csv) |
| Governance metrics | [`paper/tables/governance_metrics.csv`](../../tables/governance_metrics.csv) |
| Campaign | `kdd_main_5trial` |
| Node | `resnet_trigger` |
| Memory mode | `append_only_summary_with_rationale` |
| Budget | 5 |
| Initial / best / final accepted `val_auc` | 0.781 / 0.783 / 0.783 |
| Net gain | +0.002 |
| Kept / discarded / failed-invalid | 3 / 1 / 1 |
| Acceptance rate | 0.60 |
| Invalid rate | 0.20 |
| Complete provenance rate | 1.00 |
| Artifact capture completeness | 0.80 |

The campaign demonstrates lifecycle diversity under a fixed budget: improving trials are kept, a non-improving valid trial is discarded, and a no-op trial is failed invalid. Provenance is complete for all five trial records. Artifact capture is lower than provenance completeness because dry-run patch/log paths are placeholder references by design.

## Memory Ablation

**Figure 2:** [`paper/figures/fig2_repeated_bad_rate.svg`](../../figures/fig2_repeated_bad_rate.svg) plots repeated-bad rate by memory mode.

**Table 3: Memory ablation summary.**

| Memory mode | Budget | Repeated-bad count | Repeated-bad rate | Kept | Discarded | Failed-invalid | Best `val_auc` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 5 | 3 | 0.60 | 1 | 2 | 2 | 0.781 |
| `append_only_summary` | 5 | 1 | 0.20 | 3 | 2 | 0 | 0.783 |
| `append_only_summary_with_rationale` | 5 | 0 | 0.00 | 4 | 1 | 0 | 0.784 |

The dry-run ablation follows the pre-stated governance-memory hypothesis: repeated-bad rate decreases from no memory, to summary memory, to summary-plus-rationale memory. This is governance-contract evidence, not yet a replacement for a full real-worker ablation. It shows that the reporting path and repeated-bad detector expose the intended behavioral metric under equal budgets.

## Decision Breakdown

**Figure 3:** [`paper/figures/fig3_decision_breakdown.svg`](../../figures/fig3_decision_breakdown.svg) shows kept, discarded, and failed-invalid counts across the main, ablation, and stress campaigns.

The decision distribution is intentionally not collapsed into a single score. For governed experimentation, a `failed_invalid` trial is evidence that the control plane observed and classified an invalid action. The stress campaign contributes one `failed_invalid / invalid_edit_scope` record, while the main dry-run campaign contributes one `failed_invalid / no_op_patch` record.

## Failure Taxonomy Results

**Table 2 result artifact:** [`paper/tables/failure_taxonomy.csv`](../../tables/failure_taxonomy.csv).

| Campaign | Failure category | Count | Control-plane response |
|---|---|---:|---|
| `kdd_main_5trial` | `degraded_metric` | 1 | Discard valid but non-improving trial. |
| `kdd_main_5trial` | `no_op_patch` | 1 | Mark failed invalid and skip execution. |
| `ablation_none` | `degraded_metric` | 2 | Discard valid but non-improving trials. |
| `ablation_none` | `no_op_patch` | 2 | Mark failed invalid and skip execution. |
| `ablation_append_only_summary` | `degraded_metric` | 2 | Discard valid but non-improving trials. |
| `ablation_append_only_summary_with_rationale` | `degraded_metric` | 1 | Discard valid but non-improving trial. |
| `kdd_stress_scope` | `invalid_edit_scope` | 1 | Reject forbidden-file patch; keep audit artifacts. |

The taxonomy makes failures inspectable. In particular, the stress result shows that scope violations are not hidden behind worker exceptions: the ledger contains the decision, failure category, patch reference, raw-log reference, and unchanged git state.

## Provenance Chain

**Table 4 result artifact:** [`paper/tables/provenance_chain.csv`](../../tables/provenance_chain.csv).

Across the KDD campaign ledgers, the provenance chain records proposal, patch, run, metric, and decision IDs for each trial. The artifact completeness checker reports [`84/84 checks passed (100.0%)`](../../tables/artifact_completeness_report.txt), covering the main campaign, three ablation arms, and stress campaign. The manifest in [`artifact_manifest.json`](../../../artifact_manifest.json) indexes ledgers, tables, figures, and per-trial patch/log references.

## Task Metric Trajectory

**Figure 4:** [`paper/figures/fig4_trajectory.svg`](../../figures/fig4_trajectory.svg) shows the main campaign metric trajectory.

The fixed-budget dry-run artifact reports `val_auc` moving from 0.781 to 0.783. The accepted edit also improved validation AUC by 0.002845; we report this as secondary evidence that the governed loop can execute meaningful real experiments. The central result is not the magnitude of AUC movement, but that task performance, lifecycle decisions, failures, provenance, and artifacts are reported together.
