# Up-to-date Status

Date: 2026-05-15

This file is the current planning source of truth for the KDD AAE paper and
`autoresearch_harness` cleanup/evidence work. It folds in useful material from
the active and archived files under `plan/`, while treating old run-state notes
as historical context.

Historical inputs folded in:

- `plan/archive/KDD_AAE_refinement_plan_v2.md`
- `plan/archive/KDD_AAE_competitive_analysis.md`
- `plan/archive/ablation_hypothesis.md`
- `plan/archive/current_readiness_next_steps.md`
- `plan/archive/ml_intern_analysis.md`
- `plan/archive/*.md`
- `plan/archive/cleanup_candidates.md`

## Short Version

**Current milestone: EVIDENCE EXPANSION BEFORE FINAL SUBMISSION.**

The paper is technically submission-ready as an anonymous 7-page ACM draft.
The evidence expansion has shifted from a new synthetic MLP node to two public
OpenML tabular nodes, which gives the paper a stronger public reproducibility
story than another synthetic-only substrate.

The evidence package is complete:
- **Canonical ResNet scientific case study** (`kdd_resnet_scientific_20`): 20
  real-worker trials, 4 kept, 5 discarded, 11 failed-invalid, best validation
  AUC 0.782733, complete provenance. The old `kdd_main_5trial` is historical
  smoke evidence, not the canonical paper row.
- **Memory ablation v2** (`lg_ablation2_*`, budget=10, temp=0.7): non-monotonic result — summary arm (RBR=0.44, best AUC=0.8212) beats both none and rationale (RBR=0.78). Summary arm consistently discovers TRAIN_FRACTION.
- **Memory ablation v3** (`lg_ablation3_*`, budget=20): ordering replicates — summary (RBR=0.75) < none=rationale (RBR=0.80). AUC advantage of summary arm persists (0.815 vs 0.803 vs 0.775).
- **Three-way lifecycle**: demonstrated across evidence set — `p8_memory10_none` has 7 real `discarded` trials with full provenance. Also demonstrated in a single campaign on `lr_synthetic`.
- **Seed replication (Rep2)**: COMPLETE. Ordering holds in **2/3 replicates** (rep1 budget=10 ✅, rep2 budget=10 ❌, rep3 budget=20 ✅). Rep2 reversal: rationale RBR=0.556 < summary RBR=0.667. Paper updated to "suggestive rather than robust."
- **Rationale verbosity ablation (P13-H)**: COMPLETE. NOT SUPPORTED — short rationale (50 tok) matches full rationale (RBR=0.778), not summary (RBR=0.444). Length is not the mechanism.
- **Second node (P13-I)**: COMPLETE. `lr_synthetic` node (pure-NumPy logistic regression, LocalWorker) ran 5-trial baseline/protocol campaign and a clean LangGraph 3-arm × 10-trial memory ablation. The ablation passed strict validation but did **not** reproduce the ResNet memory ordering: RBR none/summary/rationale = 0.70/0.70/0.80. This now supports the diagnostic-governance framing, not a memory-success claim.
- **Public OpenML nodes (P15)**: COMPLETE. `openml_credit_g_main_20` has 1 kept, 19 discarded, 0 failed-invalid, best AUC 0.761058. `openml_bank_marketing_main_20` has 7 kept, 1 discarded, 12 failed-invalid, best AUC 0.934117. Both have complete provenance.
- **Third synthetic model-class node (P14 support evidence)**: COMPLETE.
  `mlp_synthetic` is implemented as a pure-NumPy one-hidden-layer MLP node.
  The support campaign `mlp_synthetic_baseline_5` has 1 kept, 4 discarded,
  0 failed-invalid, best `val_score=0.968949`, complete provenance, and
  complete artifact capture. This is supporting portability evidence and is not
  integrated into the main paper unless the evidence set is expanded again.
- **Priority 16/17 refinement**: COMPLETE. Results now foreground multi-node
  governance transfer; the ResNet node is described as near-threshold detector
  waveform classification without unsupported detector-domain overclaiming;
  artifact evidence is split into provenance, decisions, failure evidence, and
  valid-patch diffs.
- **LaTeX**: COMPILED CLEAN after P16/P17 integration. 7 total pages, no undefined refs, no missing citations. Non-fatal overfull/underfull warnings remain.
- **Paper rewrite (submission prep)**: COMPLETE. Paper shortened from ~15 pages to 6 pages. `review` removed from documentclass. All stale verbosity claims fixed. Tables compacted, comparison tables removed, moved to prose.
- **Venue**: Jeju Island, Republic of Korea.

Key claims after P15:
1. ✅ Governance protocol transfers beyond the private ResNet node to synthetic and public OpenML nodes.
2. ✅ Repeated-bad rate diagnoses memory sensitivity and non-transfer; memory improvement is mixed, not robust.
3. ✅ Rationale verbosity is NOT the mechanism for the non-monotonic ResNet result.
4. ✅ Writing quality and framing revised (P13-B through P13-G complete).
5. ✅ Structural elements present: lifecycle diagram, KDD AAE mapping table, evidence strength table.

**Priority 13 status**: ALL COMPLETE (P13-A through P13-I, plus Rep2 and the `lr_synthetic` LangGraph transfer).

**Priority 14 status**: PARTIALLY COMPLETE AS SUPPORT EVIDENCE. The originally
planned `mlp_synthetic` node is now implemented and has a 5-trial baseline
campaign. The heavier P14 LangGraph memory ablation and stress test remain
optional because the higher-ROI submission path is still the public OpenML
evidence (`openml_credit_g` and `openml_bank_marketing`).

**Priority 15 status**: COMPLETE. Public OpenML evidence integration:

1. `openml_credit_g` implemented as a controlled config-edit node using OpenML
   dataset 31.
2. `openml_bank_marketing` implemented as a controlled config-edit node using
   OpenML dataset 1461.
3. Both nodes ran 20-trial LangGraph + summary-memory campaigns with complete
   provenance.
4. Results exported to
   `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/tables/openml_campaign_summary.*`.
5. Integrated into the canonical anonymous paper as public reproducibility and
   governance-transfer evidence, not as an AutoML benchmark claim.

Target claim after P15:

> Across nodes, the governance protocol remains stable; memory effects vary by
> manager and task, which repeated-bad rate exposes.

**Priority 16 status**: COMPLETE. The canonical paper now treats the full
multi-node governance package as the main evidence. Section 5.1 is a
real-worker scientific-node case study, and the OpenML results are framed as
main public governance-transfer evidence rather than secondary benchmark
results. Memory ablation is explicitly a diagnostic use of repeated-bad rate,
not a proposed memory method.

**Priority 17 status**: COMPLETE. A new 20-trial ResNet case-study campaign
(`kdd_resnet_scientific_20`) replaces the old 5-trial quantitative row in the
paper. Node documentation now clarifies the fixed scientific/data contract
without changing data loading, split logic, H5 assets, or editable scope.

## Current Readiness Level

Current status has passed Level 7. The anonymous paper compiles and can be
submitted after final visual/prose review.

| Level | Meaning | Status |
|---|---|---|
| 0 | Only dry-run ledgers or code-only claims | Passed |
| 1 | One real node demonstration plus governed infrastructure | Passed |
| 2 | Usable real main campaign, real ablation, stress trial, Tables 1-4 | Passed |
| 3 | 10 trials or clean memory effect, manager comparison, reproducibility package | Passed: 10-trial memory extension, manager comparison, seed replicates, LangGraph ablation v2+v3, trace/context utilities done |
| 4 | Second node, repeated seeds, public artifact package, reviewer quickstart | **Complete**: lr_synthetic second node ✅; rep2 seed replication ✅ (2/3 hold); verbosity ablation ✅ |
| 5 | Prose formalization, structural figures, submission-quality PDF | **Complete**: intro reframed, lifecycle diagram, KDD AAE table, evidence strength table, paper shortened to 6pp, compiled clean |
| 6 | Submission-ready anonymous PDF | **Passed**: `[sigconf,anonymous]`, no review markers, correct claims, ≤9 body pages |
| 7 | Evidence expansion for higher reviewer confidence | **Passed**: two public OpenML nodes integrated; 20-trial ResNet case study integrated; paper compiles at 7 pages |

Required work before final submission is now final visual/prose review only.
The anonymous draft has been recompiled with the public OpenML evidence package
and remains under the 9-page body limit.

## Core Paper Identity

One-sentence thesis:

> Autonomous ML experimentation becomes scientifically credible only when the
> agent loop is governed by an explicit control plane that owns lifecycle,
> enforces bounded execution, records decisions append-only, and reports
> auditable governance metrics independent of the specific manager or worker
> backend.

Contribution type:

- Evaluation methodology.
- Governed harness design.
- Auditable control-plane protocol for autonomous experimentation.

Non-claims:

- This is not a general autonomous scientist.
- This is not proof of scientific discovery.
- This is not a universal optimization algorithm.
- This is not a detector-physics result.
- This is not dependent on a specific coding-agent backend.
- The ResNet-trigger task is a controlled scientific ML case study, not a claim
  of broad ML benchmark coverage.

Core filter for every paragraph, table, figure, and experiment:

> Does this help a reviewer evaluate whether the autonomous agent loop is
> bounded, auditable, failure-aware, reproducible, and behaviorally affected by
> governance memory?

If not, it should be moved to secondary context or cut.

## Current Implementation Progress

### Implemented

- Six-layer framework structure: manager, control plane, worker, node spec,
  memory/audit, evaluation/reporting.
- `run_real_campaign()` path through the Stage 2 control plane.
- Manager/control-plane separation: manager proposes; control plane validates,
  budgets, records, and decides.
- Append-only JSONL trial ledgers.
- Pending-trial guard and recovery tooling.
- Editable-scope validation from node specs.
- Three memory modes:
  - `none`
  - `append_only_summary`
  - `append_only_summary_with_rationale`
- Repeated-bad detection.
- No-op patch guard and `no_op_patch` failure category.
- Seed/config/hash fields in trial records.
- Patch capture before legacy discard in `harness/claw-code`.
- Persisted campaign event streams under `experiments/events/`.
- Paper table and figure exporters.
- Artifact manifest and artifact-completeness tooling.
- Provider resolver and LangChain proposal backend.
- LangGraph manager scoped correctly as a proposal backend, not as lifecycle
  authority.
- KDD paper scaffold under `paper/kdd_aae_2026/`.
- Forced stress evidence for invalid scope and no-op patch.
- Clean `kdd_main_5trial` real campaign rerun after reset hygiene:
  2 kept valid edits, 3 runtime-invalid trials, complete provenance, event
  stream present, and no pending guard.
- State-aware hyperparameter proposal hygiene:
  - `train.py` constants are parsed before bounded hyperparameter proposals;
  - structured edits record `{symbol, old, new}` in proposal metadata;
  - impossible/no-op structured edits are rejected before calling the worker;
  - simple constant edits use a deterministic patch path instead of an LLM edit
    loop;
  - effective-config checks reject patches that do not change the actual
    training configuration under fast-search overrides;
  - edit/precondition/effective-config failures are split from true
    `runtime_error` training failures.
- Real manager comparison with reset-before-each-arm hygiene:
  - `baseline_manager` vs `prompt_manager`;
  - 5 trials each;
  - both arms produced 2 kept, 3 discarded, 0 failed-invalid;
  - both reached best `val_auc=0.776533`;
  - `paper/tables/manager_comparison_summary.csv` is current evidence.
- Future-facing reproducibility/governance utilities:
  - pre-campaign node research context with SHA-256 metadata hook;
  - similarity detector comparison against token-set and fuzzy baselines;
  - intra-proposal doom-loop guard;
  - isolated trial worktree creation with approval JSON;
  - promoted-master snapshot export;
  - breadth/depth research scouting plan generation.

### Partially Implemented or Needs Verification

- Reset hygiene:
  - implemented for editable files, campaign ledgers, pending guards, event
    streams, exact campaign artifact directories, `.autoresearch_state.json`,
    `results.tsv`, `experiment_memory.jsonl`, node-local `run.log`, and
    node-local `artifacts/`;
  - focused regression coverage exists in `tests/test_reset_node_state.py`;
  - the clean reset has been run before the current main campaign.
- LangGraph:
  - architecturally scoped correctly;
  - dependencies are in `pyproject.toml`;
  - still needs normal CI-style tests and at least one real control-plane smoke
    with a stubbed worker or fake LLM path if it will be mentioned strongly.
- Paper artifacts:
  - exporters exist;
  - primary tables, figures, artifact completeness report, and manifest have
    been regenerated from current evidence.
- Trace export:
  - event streams exist;
  - reviewer-facing trace export with redaction exists under
    `scripts/export_kdd_traces.py`.

### Repository Hygiene Progress

Low-risk cleanup has been done for generated/ignored smoke artifacts, old
untracked smoke ledgers, empty artifact directories, node-local caches, old
node checkpoints/backups, `.DS_Store`, pytest caches, and node-local virtual
environments.

Remaining hygiene caveats:

- Root `.venv/` remains.
- `harness/claw-code` is a submodule and currently appears modified in the
  parent worktree.
- Historical planning notes have been consolidated under `plan/archive/`.
`plan/Up-to-date.md` and `plan/TODO.md` are the active planning files.

### Priority 15 OpenML Evidence Snapshot

Canonical ledgers:

- `experiments/ledgers/openml_credit_g_main_20_trials.jsonl`
- `experiments/ledgers/openml_bank_marketing_main_20_trials.jsonl`

Summary:

| Campaign | Node | Trials | Kept | Discarded | Failed-invalid | Best metric | Provenance |
|---|---|---:|---:|---:|---:|---:|---:|
| `openml_credit_g_main_20` | `openml_credit_g` | 20 | 1 | 19 | 0 | 0.761058 AUC | 1.00 |
| `openml_bank_marketing_main_20` | `openml_bank_marketing` | 20 | 7 | 1 | 12 | 0.934117 AUC | 1.00 |

Interpretation:

- `credit-g` is a clean public transfer node: all 20 trials are valid and
  auditable; most non-improving trials are correctly discarded rather than
  failed.
- `bank-marketing` is the stronger paper result: validation AUC improves from
  0.922002 to 0.934117 while the ledger also records discarded and
  failed-invalid outcomes. Invalids are out-of-range config proposals
  (`max_depth > 30`, `n_estimators > 500`) caught by the control plane.
- Together, the OpenML nodes raise the evidence from synthetic-only transfer to
  public reproducible tabular transfer. They should be reported as governance
  evidence, not as competitive AutoML results.

## Current Evidence

### Canonical ResNet Scientific Case Study

Canonical ledger:

`experiments/ledgers/kdd_resnet_scientific_20_trials.jsonl`

| Field | Current value |
|---|---:|
| Records | 20 |
| Worker/manager | deterministic patch bridge under `prompt_manager` |
| Decisions | 4 `kept`, 5 `discarded`, 11 `failed_invalid` |
| Failures | 5 `degraded_metric`, 11 `proposal_precondition_failed` |
| Kept | 4 |
| Discarded | 5 |
| Initial / best parsed `val_auc` | 0.774711 / 0.782733 |
| Event stream | `experiments/events/kdd_resnet_scientific_20_events.jsonl` |
| Pending guard | absent |
| Provenance | complete for all 20 records |
| Patch artifacts | present for all 9 materialized valid runs |

Interpretation:

- Canonical real-worker scientific-node case study for the paper.
- Demonstrates kept, discarded, and failed-invalid lifecycle outcomes in one
  ResNet-trigger campaign with complete provenance.
- Supports governance correctness and auditability; the AUC gain remains inside
  baseline seed variation and should not be reported as scientific optimization
  evidence.
- After nine materialized valid edits, the structured manager exhausts
  available non-no-op bounded edits and records eleven
  `proposal_precondition_failed` outcomes. This is a useful bounded-proposal
  lifecycle result, not a training failure.

Historical earlier ledger:

`experiments/ledgers/kdd_main_5trial_trials.jsonl`

The 5-trial run remains useful as historical smoke evidence only: 2 kept,
0 discarded, 3 failed-invalid, complete provenance.

Trial summary:

| Trial | Proposal | Decision | Failure category | `val_auc` | Patch artifact |
|---:|---|---|---|---:|---|
| 1 | `lower-weight-decay` | `kept` | n/a | 0.774711 | present |
| 2 | `small-dropout` | `discarded` | `degraded_metric` | 0.773778 | present |
| 3 | `smaller-kernel` | `discarded` | `degraded_metric` | 0.773822 | present |
| 4 | `lower-grad-clip` | `kept` | n/a | 0.776533 | present |
| 5 | `lower-learning-rate` | `discarded` | `degraded_metric` | 0.773244 | present |
| 6 | `mild-weight-decay` | `kept` | n/a | 0.776756 | present |
| 7 | `tiny-dropout` | `discarded` | `degraded_metric` | 0.774622 | present |
| 8 | `larger-grad-clip` | `discarded` | `degraded_metric` | 0.773556 | present |
| 9 | `larger-kernel` | `kept` | n/a | 0.782733 | present |
| 10--20 | no valid non-no-op bounded edit available | `failed_invalid` | `proposal_precondition_failed` | n/a | n/a |

### Memory Ablation

Current ledgers:

- `experiments/ledgers/ablation_none_trials.jsonl`
- `experiments/ledgers/ablation_append_only_summary_trials.jsonl`
- `experiments/ledgers/ablation_append_only_summary_with_rationale_trials.jsonl`

Current state:

| Campaign | Worker | Records | Decisions | Final repeated-bad count | Final metric |
|---|---|---:|---|---:|---:|
| `ablation_none` | `claw_style_worker` | 5 | 2 kept, 3 failed-invalid | 2 | 0.774711 |
| `ablation_append_only_summary` | `claw_style_worker` | 5 | 2 kept, 3 failed-invalid | 2 | 0.774711 |
| `ablation_append_only_summary_with_rationale` | `claw_style_worker` | 5 | 2 kept, 3 failed-invalid | 2 | 0.774711 |

Interpretation:

- The old claim that memory ablation is incomplete is stale.
- All three arms now have 5 real records and event streams.
- The result is flat across modes.
- The current evidence does not support the pre-stated hypothesis that
  rationale memory reduces repeated poor proposals.
- This can be reported as a negative result if the setup is accepted as a fair
  test. Otherwise, rerun with a design that better exposes memory effects.

Pre-stated hypothesis:

```text
repeated_bad_rate(none) > repeated_bad_rate(append_only_summary)
                        > repeated_bad_rate(append_only_summary_with_rationale)
```

Grounding:

- OpenAI: give the agent a map, not a manual.
- Böckeler: feedforward guides plus feedback sensors should reduce recurrence.

Current result:

- flat, not confirmed.

### Manager Comparison

Current ledgers:

- `experiments/ledgers/manager_comparison_baseline_manager_trials.jsonl`
- `experiments/ledgers/manager_comparison_prompt_manager_trials.jsonl`

Summary table:

- `paper/tables/manager_comparison_summary.csv`

Current state:

| Manager | Records | Kept | Discarded | Failed-invalid | Best `val_auc` | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_manager` | 5 | 2 | 3 | 0 | 0.776533 | 0.001822 |
| `prompt_manager` | 5 | 2 | 3 | 0 | 0.776533 | 0.001822 |

Interpretation:

- Supports the manager-agnosticism claim at the lifecycle/provenance layer for
  bounded structured hyperparameter edits.
- Because both managers use the same deterministic structured selector, do not
  overclaim behavioral diversity; this evidence says the control plane can run
  both manager backends under equal budget and preserve complete governance
  records.

### Optional Priority 8 Extension

Current ledgers:

- `experiments/ledgers/p8_memory10_none_trials.jsonl`
- `experiments/ledgers/p8_memory10_append_only_summary_trials.jsonl`
- `experiments/ledgers/p8_memory10_append_only_summary_with_rationale_trials.jsonl`

Summary tables:

- `paper/tables/p8_memory10_summary.csv`
- `paper/tables/p8_memory10_run_report.txt`
- `paper/tables/p8_memory10_similarity_detector_comparison.csv`

Current state:

| Memory mode | Records | Kept | Discarded | Failed-invalid | Best `val_auc` | Repeated-bad rate |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 10 | 3 | 7 | 0 | 0.776756 | 0.60 |
| `append_only_summary` | 10 | 4 | 5 | 1 | 0.782733 | 0.40 |
| `append_only_summary_with_rationale` | 10 | 4 | 5 | 1 | 0.782733 | 0.40 |

Interpretation:

- The 10-trial extension shows a clearer memory effect than the 5-trial
  version: both memory modes reduce repeated-bad rate from 0.60 to 0.40 and
  reach a higher best AUC.
- The two memory modes are still tied, so do not claim rationale memory beats
  summary memory.
- The single failed-invalid record in each memory arm is
  `proposal_precondition_failed`, not a training runtime failure. It means the
  state-aware selector exhausted non-no-op structured edits at the current
  state.

### Baseline Seed Replicates

Current tables:

- `paper/tables/baseline_seed_replicates.csv`
- `paper/tables/baseline_seed_bootstrap_ci.csv`

Current state:

| Seed | Baseline `val_auc` |
|---:|---:|
| 123 | 0.774711 |
| 124 | 0.771511 |
| 125 | 0.810711 |
| 126 | 0.784800 |
| 127 | 0.782044 |

Bootstrap summary over seed means:

- mean baseline `val_auc`: 0.784755
- population std: 0.013837
- min/max: 0.771511 / 0.810711
- bootstrap 95% CI for mean: [0.774809, 0.798329]

Interpretation:

- Baseline seed noise is larger than the small within-campaign AUC gains in
  the 5-trial primary campaign. Keep AUC claims secondary and use these seed
  replicates as a guardrail against overclaiming optimization performance.
- Governance claims remain stronger because lifecycle/provenance/scope metrics
  are deterministic ledger properties.

### Priority 11 Utilities

Generated artifacts:

- `paper/notes/resnet_trigger_research_context.md`
- `paper/notes/resnet_trigger_scouting_plan.md`
- `paper/tables/similarity_detector_comparison.csv`
- `paper/tables/promoted_master_snapshot.json`

New scripts/modules:

- `scripts/build_node_research_context.py`
- `scripts/compare_similarity_methods.py`
- `scripts/create_trial_worktree.py`
- `scripts/generate_research_scouting_plan.py`
- `scripts/promote_master_snapshot.py`
- `src/autoresearch/memory/research_context.py`
- `src/autoresearch/manager/doom_loop.py`

### Stress Evidence

Current stress ledgers:

| Campaign | Decision | Failure category | Status |
|---|---|---|---|
| `kdd_stress_scope` | `failed_invalid` | `invalid_edit_scope` | useful |
| `kdd_stress_noop` | `failed_invalid` | `no_op_patch` | useful |

These support the claim that invalid and failed trials are first-class audit
objects. Rerun only if taxonomy logic, artifact paths, or no-op/scope behavior
changes.

### Patch and No-op Verification

Priority 2 verification has been run after the patch-capture fix:

- Focused tests passed:
  - `tests/test_legacy_noop_skip.py`
  - `tests/test_noop_guard.py`
  - `tests/stage2/test_stage2_control_plane.py`
- One-trial real smoke after reset completed as
  `failed_invalid/runtime_error` with complete provenance and no pending guard.
  This smoke did not produce a patch because the worker produced no effective
  changed file.
- Current main campaign proves real-edit patch capture:
  - `trial-002/patch.diff`: 323 bytes, `DROPOUT = 0.02` to `0.022`;
  - `trial-005/patch.diff`: 340 bytes, `BATCH_SIZE = 32` to `64`.
- Explicit no-op verification campaign `noop_verify_priority2` produced one
  `failed_invalid/no_op_patch` record with empty patch hash, unchanged git
  state, training skipped, complete provenance, and no pending guard.

### State-aware Proposal Fix

Implemented after diagnosing the invalid main-campaign trials:

- The prior invalid trials were mostly edit-loop failures caused by stale
  hard-coded objectives, not training crashes.
- `prompt_manager` / `baseline_manager` now produce structured state-aware
  hyperparameter proposals when current `train.py` constants are available.
- `ClawWorker` applies simple structured constant edits deterministically and
  only uses the LLM edit loop for unstructured/general edits.
- A deterministic real smoke `deterministic_patch_smoke` succeeded:
  `WEIGHT_DECAY 5e-5 -> 3e-5`, effective config changed, patch captured,
  metric parsed, one kept record, complete provenance, and no pending guard.
- The smoke ledger/artifacts are validation evidence only; the node state was
  reset afterward.

### Historical Evidence

Older runs such as `resnet_real_incremental` remain useful as historical proof
that the harness can execute a real node and keep an improving edit. They are
not current primary evidence for the KDD results section.

Best use:

- mention as secondary real-execution evidence if needed;
- avoid using it as the central empirical result.

## Supported and Unsupported Claims

### Currently Supported

- Bounded execution improves auditability.
- Explicit lifecycle control makes autonomous experiments inspectable.
- Invalid and failed trials can be recorded as first-class audit objects.
- The harness can execute a real scientific ML node.
- The manager cannot self-approve success; the control plane owns decisions.
- Event streams and append-only ledgers expose lifecycle and final record state.
- The architecture is manager/backend-agnostic by design.
- The current canonical real-worker campaign records kept, discarded, and
  failed-invalid trials with complete provenance.
- Real edit patches are captured as audit artifacts before the legacy worker
  discards or resets state.
- Governance protocol transfers across the private ResNet-trigger node,
  `lr_synthetic`, and two public OpenML tabular nodes.

### Not Yet Supported

- Rationale memory reduces repeated-bad rate (non-monotonic: none = rationale > summary; not consistently reversed by rep2 — verdict: suggestive, not robust).
- Non-monotonic ordering is robust across all seeds (2/3 replicates hold; rep2 fails).
- Rationale verbosity is the causal mechanism (verbosity ablation NOT SUPPORTED; length is not the driver).

### Now Supported (updated 2026-05-14)

- Full three-way lifecycle diversity: `kdd_resnet_scientific_20`,
  `p8_memory10_none`, and `lr_synth_baseline` demonstrate kept/discarded/failed
  outcomes under the same ledger schema.
- Summary memory (not rationale) reduces repeated-bad rate — non-monotonic finding held in 2/3 replicates (suggestive).
- Non-monotonic ordering replicates across budget levels (v2: budget=10, gap=0.34; v3: budget=20, gap=0.05); fails in rep2 (budget=10, second seed).
- Summary arm consistently discovers TRAIN_FRACTION in both v2 and v3; rationale arm gets stuck on dead-end params.
- Rationale length is NOT the mechanism: short rationale (50 tok) matches full rationale RBR, not summary RBR.
- Governance protocol operates across private scientific, synthetic, and public
  OpenML nodes without changing the control-plane contract.
- LaTeX compiles to a valid 7-page PDF with 0 errors after P16/P17.

## Current Problems

### 1. Writing Quality: Lab-Report Language in §6 [RESOLVED IN P13]

`06_discussion_limitations.tex` uses internal project language: "What Is Proven",
"stress artifacts regenerated from worker code", "legacy claw loop", "legacy loop
path", bare campaign IDs in prose sentences. These read as dev notes, not a
conference paper. See P13-B in TODO.md for the exact sentences to replace.

### 2. Framing: Paper Reads as Optimization System [RESOLVED IN P13]

The abstract and introduction lead with the control-plane architecture rather
than the evaluation methodology claim. The reviewer read it as "a system for
doing better ML" rather than "a framework for evaluating whether ML agents are
behaving reliably." The fix is a rewrite of the abstract first sentence and the
introduction's contribution bullets. See P13-C in TODO.md.

### 3. Missing Structural Elements [RESOLVED IN P13]

Three tables/figures would significantly improve reviewer confidence:
- KDD AAE mapping table (workshop concerns → harness mechanisms) — §3.
- Evidence strength classification table (claim → evidence level) — §6.
- Polished lifecycle diagram replacing the current Mermaid-style fig1 — §1 or §3.
See P13-D, P13-E, P13-F in TODO.md.

### 4. Seed Replication Rep2 [COMPLETE 2026-05-14]

Rep2 DONE: ordering fails in rep2 (rationale RBR=0.556 < summary RBR=0.667).
Paper updated: §5.4 (new seed-replication subsection) reports "2/3 replicates hold;
suggestive rather than robust." No further action needed.

### 5. Single-Node Evidence [RESOLVED THROUGH P15/P17]

`lr_synthetic`, `openml_credit_g`, and `openml_bank_marketing` are complete.
The 20-trial ResNet case study, synthetic transfer node, and public OpenML
nodes are now all integrated in the paper. Memory effects remain mixed and are
framed as diagnostic rather than robust.

### 6. Paper Artifacts Need Recompile [RESOLVED 2026-05-15]

Canonical paper recompiled after OpenML and P16/P17 integration:
`A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/main.pdf`.
It is 7 pages, anonymous, and has no undefined references or missing citations.

## Metrics and Tables to Preserve

### Repeated-Bad Rate

Definition:

```text
RepeatedBadRate = repeated_bad_proposals / total_proposals
```

Operational definition:

> A repeated-bad proposal repeats the same edit target and mechanism as a prior
> rejected, invalid, or degraded trial without adding a new corrective
> rationale, constraint, or justification.

### Failure Taxonomy

Use as a paper contribution, not just bookkeeping.

| Category | Meaning |
|---|---|
| `invalid_edit_scope` | Patch touched a disallowed file or region. |
| `syntax_error` | Code cannot be parsed or imported. |
| `runtime_error` | Training command exits nonzero. |
| `edit_failed` | Worker/coding loop failed to produce an edit. |
| `proposal_precondition_failed` | Structured proposal did not match current source state or was impossible. |
| `effective_config_unchanged` | Patch changed source text but not the actual training config being run. |
| `metric_missing` | Run completes but expected metric cannot be parsed. |
| `degraded_metric` | Valid run is worse than current best, so discarded. |
| `no_op_patch` | Worker produced no effective change. |

Key distinction:

- `failed_invalid`: validity failure or no parseable metric.
- `discarded`: valid metric, not accepted because it did not improve.
- `kept`: valid metric and accepted by deterministic decision rule.

### Provenance Completeness

```text
ProvenanceCompleteness = completed_required_artifact_links / total_required_artifact_links
```

Required chain:

- proposal id;
- worker packet reference;
- patch reference;
- run log reference;
- parsed metrics reference;
- decision id;
- ledger record id.

### Artifact Capture Completeness

```text
ArtifactCaptureCompleteness = captured_artifacts / expected_artifacts
```

Expected artifacts:

- proposal JSON;
- worker packet;
- patch diff;
- raw run log;
- parsed metrics;
- control-plane decision record;
- ledger entry.

### Paper Tables and Figures

Required:

- Table 1: main campaign optimization plus governance.
- Table 2: failure taxonomy.
- Table 3: memory ablation.
- Table 4: provenance chain.
- Table 5: AIDE comparison.
- Figure 1: architecture diagram.
- Figure 2: repeated-bad rate by memory mode.
- Figure 3: decision breakdown.
- Figure 4: campaign trajectory, with AUC as secondary evidence.

## Related Work and Reviewer Framing

### SHARP

SHARP is stronger on empirical validation today: multiple complete runs,
external validation, and a real reproducibility pipeline. This project is
deeper on governance: lifecycle state machine, scope enforcement, append-only
audit ledger, failure taxonomy, memory ablation, and separated decision
authority.

Positioning:

- SHARP: human-agent reproduction pipeline.
- `autoresearch_harness`: autonomous experimentation governance and evaluation
  protocol.

The main gap relative to SHARP is not concept design. It is clean real evidence.

### AIDE

AIDE is the closest academic system. Both use LLMs to propose code changes
evaluated by a hard metric. Difference:

- AIDE optimizes an ML metric through a search loop.
- `autoresearch_harness` governs the experimentation process.

Required comparison dimensions:

- LLM role;
- control-plane ownership;
- keep/discard authority;
- scope enforcement;
- audit ledger;
- failure taxonomy;
- memory/context;
- reproducibility.

### Harness Engineering Literature

Use these ideas explicitly:

- Agent = Model + Harness.
- Computational controls are the trust layer; inferential components add
  semantic richness but need deterministic boundaries.
- Self-evaluation is unreliable; evaluator/decision authority must be separate
  from the generator.
- Failures are signals to engineer against, not noise to hide.
- Memory should be structured failure context, not raw history.
- Holdout evaluation nodes are future work to avoid harness overfitting.

### Guides and Sensors Table

Add this to System Design:

| | Guide, feedforward | Sensor, feedback |
|---|---|---|
| Computational | Node spec, scope validator, budget cap, editable-path whitelist | Metric parser, state-machine validator, pending guard |
| Inferential | Manager prompt plus memory injection | Repeated-bad detector |

### Failure Mode Motivation Table

Add this to System Design:

| Failure mode | ML experimentation analog | Harness fix |
|---|---|---|
| Declares victory too early | Manager stops after one good trial | Budget enforcer and fixed trial count |
| Leaves broken state | Failed patch leaves repo dirty | Pending guard and reset |
| Marks done prematurely | Claims improvement without valid metric | Metric parser and `failed_invalid` state |
| Does not know how to run | Proposes out-of-scope edit | Node spec and editable-path whitelist |

## External Feature Lessons

Do not import broad external systems wholesale. Use their patterns only when
they strengthen governance, auditability, or paper clarity.

### Already Adopted or Mostly Adopted

- Persisted event stream from ml-intern-style typed lifecycle events.
- Provider-normalized model config and LangChain backend.
- LangGraph as optional proposal manager, not lifecycle authority.

### High-Value Future Features

1. Trace export with redaction.
   - Export `experiments/traces/{campaign_id}.jsonl`.
   - Include proposal, packet, worker result, patch, log, metric parse, scope
     validation, and decision.
   - Redact secret-like strings.
   - Status: implemented in `scripts/export_kdd_traces.py`.

2. Pre-campaign research context.
   - Immutable `paper/notes/{node_id}_research_context.md`.
   - Record hash in trial metadata.
   - Use as bounded context, not as authority.
   - Status: implemented in `scripts/build_node_research_context.py` and
     `src/autoresearch/memory/research_context.py`.

3. Similarity and repetition improvements.
   - Compare current Jaccard/parameter-direction detector with
     token-set/fuzzy matching.
   - Do not swap thresholds without false-positive tests.
   - Status: comparison utility implemented; current hybrid detector remains
     the default.

4. Intra-proposal doom-loop guard.
   - Useful only if manager becomes multi-turn/tool-using.
   - Detect repeated tool calls or repeated tool-call sequences.
   - Status: reusable guard implemented in `src/autoresearch/manager/doom_loop.py`.

5. Isolated trial execution plus approval/apply gate.
   - Required before an operator approval gate can truly prevent base-node
     changes.
   - Needs temporary worktree or patch staging.
   - Status: worktree creation plus approval JSON implemented in
     `scripts/create_trial_worktree.py`; not integrated into the campaign loop.

6. Promoted-master and isolated worktrees.
   - Future multi-worker scale feature.
   - Not required for current KDD paper.
   - Status: snapshot export implemented in `scripts/promote_master_snapshot.py`.

7. Breadth/depth research scouting.
   - Future hypothesis-generation mode.
   - Keep outside current core paper scope.
   - Status: deterministic scouting-plan generator implemented in
     `scripts/generate_research_scouting_plan.py`.

### Do Not Adopt Now

- Frontend/dashboard.
- Cloud deployment.
- Slack or notification gateways.
- Full Hermes-style assistant shell.
- Tight LangGraph or LangChain dependency in the core protocol.
- Direct `ml-intern` or `multiautoresearch` integration.
- SFT export pipeline.
- Broad multi-agent complexity.
- Many benchmark domains before the current evidence is clean.

## Writing Rules

- Lead with governance metrics, not AUC.
- Treat failed and invalid trials as first-class audit objects.
- Separate dry-run contract tests from real worker evidence.
- State non-claims explicitly.
- Pre-state the memory hypothesis before results.
- If memory is flat or reversed, report it as a valid negative finding.
- Put task metric last as secondary evidence.
- Add AIDE and SHARP comparison in Related Work.
- State limited-node and no-holdout limitations directly.
- Frame memory as a diagnostic governance probe unless stronger evidence
  supports a memory-method claim.
- Do not overclaim autonomous science.

## Safe to Write Now

- Related Work.
- System Design.
- Governance protocol.
- Experiment Setup, with final run ids left open.
- Failure taxonomy definitions.
- Artifact/provenance schema.
- Limitations framework.
- AIDE comparison table.
- SHARP comparison paragraph.
- Guides/Sensors table.
- Failure-mode motivation table.

## Do Not Finalize Yet

- Results.
- Abstract.
- Introduction.
- final Discussion claims.
- quantitative memory claims.
- acceptance-rate or improvement claims.
- final artifact-completeness claims.

These wait for final evidence and regenerated artifacts.
