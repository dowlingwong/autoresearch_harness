# Paper Revision & Edit Checklist

Merged from: `revision_checklist` (reviewer-driven P0/P1/P2) + `paper_edit_checklist` (overnight run edits).
Track progress here. Each item is either a reviewer-identified weakness, a data-driven edit, or a section-level writing task.

**Last data event:** overnight run 2026-05-17, 495 trials across 6 nodes.
**Last updated:** 2026-05-17.

---

## Recommended Execution Order

1. Generate all notebook figures (`paper_figures.ipynb` → Restart & Run All)
2. Re-run ResNet in a fresh terminal session (1 trial first, validate env)
3. Low-compute prose fixes: §1 framing, §6 limitations, P2 tone fixes
4. Section rewrites: §4 Experiments, §5 Results (use new data)
5. Abstract last — update numbers only after §4/§5 are stable
6. Validate all metric claims, run artifact checker, compile
7. Deferred large experiments (P0-1 A/B, P0-4 inter-rater) — after above is stable

---

## P0 — Blocking Issues (must fix before submission)

### ✅ P0-1. Head-to-head: governed vs. ungoverned
**Mitigated by reframing, not by claiming the unrun A/B.**
- [x] Removed/softened unsupported causal wording
- [x] Added explicit limitation: current experiments show what governed harness records/rejects, not a statistically significant governed-vs.-ungoverned gap
- [x] Reframed paper as lifecycle instrumentation/auditability evidence, not causal agent-performance improvement
- [ ] **DEFERRED:** Implement ungoverned arm (same manager, no control plane, manager self-evaluates)
- [ ] **DEFERRED:** Run Arm G vs Arm U on `resnet_trigger`, ≥5 seeds per arm
- [ ] **DEFERRED:** Run same A/B on at least one OpenML node
- [ ] **DEFERRED:** Add head-to-head table to §5

### ✅ P0-2. Larger budgets and bootstrap CIs
**Partially resolved by overnight run. Bootstrap CI tooling complete.**
- [x] Implement bootstrap CI computation: `scripts/bootstrap_governance_cis.py`
- [x] Removed headline claims resting on short-run point estimates without seed-level CIs
- [x] **NEW (2026-05-17):** LR memory ablation now has 5 seeds × 3 arms × 10 trials = 150 trials ✓
- [x] **NEW (2026-05-17):** OpenML-BM and OpenML-CG now have 4 seeds × 20 trials each ✓
- [x] **NEW (2026-05-17):** mlagentbench now has 2 seeds × 10 trials ✓
- [x] **NEW (2026-05-17):** Bootstrap CIs computed for LR (15 campaigns) — `paper/tables/governance_bootstrap_cis_lr.csv`
- [ ] **TODO:** Re-run ResNet in fresh session (all 135 overnight trials failed due to MPS device exhaustion — see §Data Status below)
- [ ] **TODO:** Update every result table cell to show "estimate [low, high]" after final ResNet data
- [ ] **DEFERRED:** Re-run memory ablation at budget 30, 5 seeds per arm per node (current: budget 10–15)
- [ ] **DEFERRED:** For ordering claims, compute bootstrap P(ordering holds) after replicated seeds available

### ☐ P0-3. Frontier-class manager (model confound)
- [x] Pick frontier model: **DeepSeek-V4-Flash** via `deepseek/deepseek-v4-flash`
- [x] LangGraphManager now accepts provider-prefixed backends (`deepseek/`, `anthropic/`, `openai/`)
- [x] DeepSeek API integrated across all campaign runners
- [x] **NEW (2026-05-17):** All 495 overnight trials used DeepSeek-V4-Flash as manager ✓
- [ ] **TODO:** Justify DeepSeek model choice in paper (§4 Experiments, ~1 paragraph)
- [ ] **TODO:** State explicitly whether each finding survives the model substitution (vs old qwen2.5-coder:7b)
- [ ] **TODO:** Remove any claim about "manager behavior" supported only by qwen2.5-coder:7b
- [ ] **DEFERRED:** Add random-valid-edit baseline as a true floor
- [ ] **DEFERRED:** Table of governance metrics × manager tier × memory mode

### ☐ P0-4. Validate governance metrics as measurement instruments
- [x] Implement metric-validation summary exporter: `scripts/validate_governance_metrics.py`
- [x] Test-retest reliability smoke probe: decision agreement = 1.00 (2/2 MLAgentBench kept trials)
- [x] Implement kept-trial re-execution tool: `scripts/reexecute_kept_trials.py`
- [x] Implement blind failure-taxonomy sample exporter: `scripts/export_failure_taxonomy_labels.py`
- [x] Implement Cohen's κ scorer: `scripts/score_failure_taxonomy_labels.py`
- [x] Add §4.5 "Metric Validation" subsection
- [ ] Run with different seeds, compare to same-seed runs (reliability ratio)
- [ ] Discrimination: plot each metric across manager tiers from P0-3
- [ ] Validity probe: re-execute 5 kept trials per campaign from audit records
- [ ] **Inter-rater reliability:** have one author + one external person classify 30 failed trials into the 6 categories (`plan/metric_validation/failure_taxonomy_sample.csv`)
- [ ] Compute Cohen's κ after two completed rater sheets
- [ ] Drop or caveat any metric that fails validation

### ✅ P0-5. Reference audit — COMPLETE (2026-05-16)
- [x] All 22 cited entries audited, all Low risk
- [x] All arXiv IDs resolve
- [x] SHARP [2] arXiv `2604.18752` verified
- [x] All 8 blog/online entries verified
- [x] 4 orphan uncited entries removed
- [x] `burtenshaw2026multiautoresearch` author field fixed: `{burtenshaw}` → `Burtenshaw, Ben`

**Remaining optional (non-blocking):** add DOI to `huang2024mlagentbench`, `elsken2019nas`, `deng2023mind2web`

---

## P1 — Substantial Strengthening

### ☐ P1-1. Sharpen the contribution and demote the memory ablation
- [ ] Move failure taxonomy from §4.4 paragraph into its own §3 subsection
- [ ] Add: how categories were derived, exhaustiveness argument, extension procedure for new nodes
- [ ] Include inter-rater reliability result (from P0-4) in taxonomy section
- [ ] **NEW:** Memory ablation now has a null result at n=5 seeds (LR) and a directional result at n=1 seed (MLP) — frame as "underpowered at current scale" not as a positive finding
- [ ] Move memory ablation to appendix or a single §5 paragraph, with LR null result as primary
- [ ] Rewrite §1 to state contribution as: lifecycle + taxonomy + external enforcement + cross-node evidence
- [ ] Confirm a reviewer can state the contribution in one sentence after reading abstract + §1

### ✅ P1-2. Wrap an external benchmark node — COMPLETE
- [x] MLAgentBench `vectorization` wrapped as `mlagentbench_vectorization` NodeSpec
- [x] Run governed campaign with full governance metrics: `mlagentbench_vectorization_main_30`
- [x] **NEW (2026-05-17):** 2 seeds × 10 trials (20 total), AR=0.60/0.70, IR=0.00 ✓
- [x] Document that harness required no control-plane code changes — only registry/adapter additions
- [x] Results added to §5

### ☐ P1-3. Formalize the lifecycle state machine
- [ ] Define states S, transition function δ, terminal states T, guard conditions
- [ ] List invariants the control plane enforces
- [ ] Argue exhaustiveness: under what conditions can the lifecycle hang?
- [ ] Add transition table and invariants list to §3.4
- [ ] Confirm a reader can reconstruct the state machine from the paper alone

### ☐ P1-4. Report governance overhead (cost)
- [ ] Measure wall-clock overhead per trial vs. ungoverned baseline
- [ ] Measure additional LLM tokens consumed by the manager interface
- [ ] Measure storage overhead (ledger + patch artifacts)
- [ ] Report engineering cost (LoC for the control plane)
- [ ] Add a cost table to §6 Discussion

### ☐ P1-5. Empirical related-work comparison
- [ ] Wrap at least one external system (AIDE, MLAgentBench agent) as a manager backend under the governed harness
- [ ] Report governance metrics for the external manager
- [ ] Replace AIDE positioning prose with empirical comparison

---

## P2 — Polish (final pass before submission)

### ☐ P2-1. Internal consistency
- [ ] Reconcile abstract AUC claim with Limitations caveat — qualify inline with seed CI
- [ ] Rewrite "100% provenance completeness" framing — say what it rules out, not just that it is 100%
- [ ] Confirm "governance metrics expose what task score hides" appears only where P0-1 supports it
- [ ] Check every quantitative claim in the abstract appears with CIs in §5
- [ ] **NEW:** Confirm ResNet numbers are not used as positive evidence in abstract (environment failure)

### ☐ P2-2. Tone and framing
- [ ] Pick a register — careful narrow contribution OR ambitious general protocol — and hold it throughout
- [ ] Remove "general governance evaluation protocol" language unless evidence supports the scope
- [ ] Ensure §6 Limitations does not directly contradict the abstract
- [ ] Pass the "would a hostile reviewer find a contradiction" test

---

## Section-Level Paper Edits (driven by overnight run data)

These are the specific tex edits needed to reflect the new multi-node results.
Priority ratings use the same P0/P1/P2 scale.

### Abstract (`main.tex`)
- [P0] Update total trial count: "5 trials" → "495 trials across 6 nodes"
- [P0] Remove "ResNet-trigger" as sole node; list the node suite
- [P0] Add cross-node governance claim: "100% provenance completeness across all trials"
- [P0] State memory-mode finding honestly: null result at n=5 seeds (LR), directional at n=1 seed (MLP)
- [P0] Do NOT claim ResNet ablation results — all 135 trials failed (env issue, not science)

### §1 Introduction (`01_introduction.tex`)
- [P0] Update scope: "single governing loop on binary classification" → "governed harness across 6 benchmark tasks"
- [P1] Add ResNet environment-failure as motivation: 135 failures detected and logged without human intervention
- [P1] Update contributions list: add cross-node evaluation, multi-seed ablation, null result on memory mode

### §3 System Design (`03_system_design.tex`)
- [P1] Add subsection on multi-node NodeSpec (6 distinct specs, generalises across task types)
- [P1] Add LocalWorker vs ClawWorker distinction and design rationale
- [P2] Add node spec inventory table (7 nodes, worker type, metric, budget) — draft in `plan/experiments_plan.md`

### §4 Experiments (`04_experiments.tex`) — near-complete rewrite
- [P0] Replace single-node description with 6-node table (node, task type, worker, metric, seeds, budget)
- [P0] Add memory-mode ablation subsection: LR 5 seeds × 3 arms, primary null result
- [P0] Add ResNet environment failure as an explicit experiment/stress test
- [P1] Add experimental setup paragraph: DeepSeek-V4-Flash, langgraph_manager, seeds, MPS hardware
- [P1] Add MLP ablation (1 seed, directional signal only, supplement to LR null result)

### §5 Results (`05_results.tex`) — major restructuring
- [P0] Replace Table 1 (kdd_main_5trial) with new Table 1 from `paper/tables/table1_governance_summary.csv`
- [P0] Delete old per-trial table (trials 1–5 of kdd_main_5trial); replace with Fig 4 timeline (OpenML-BM s1)
- [P0] Replace §5.3 Memory Ablation entirely: remove Arm A/A2/A3/B; add DeepSeek LR ablation (primary) + MLP (supplemental)
- [P0] Add "Cross-node Governance Summary" subsection: overall IR=5.8%, AR=28.3%, prov=100%
- [P1] Update failure taxonomy table (Table 2): add ResNet 135 runtime_error row
- [P1] Update OpenML results: BM AR=0.242/IR=17.5%, CG AR=0.219/IR=8.7%; add trajectory figure (Fig 7)
- [P1] Keep "Governance in Action" stress-campaign table (Table 5) — confirm campaigns still in ledger
- [P2] Update task metric trajectory (§5.5): use OpenML as representative (cleaner multi-seed story)

### §6 Discussion / Limitations (`06_discussion_limitations.tex`)
- [P0] Add ResNet environment failure as a limitation AND finding (framework handled it correctly)
- [P0] Add statistical power as a limitation: n=5 seeds × 10 trials insufficient to detect small memory effects
- [P1] Add autoresearch-macos as future work (node registered, data pending `uv run prepare.py`)
- [P1] Update cross-node claim with concrete evidence: "same harness operated across 6 tasks spanning 5 types"

### Figures (`paper/figures/`)
- [P0] Run `paper_figures.ipynb` → Kernel → Restart & Run All → generates all 8 figures + Table 1
- [P0] Update figure references in `main.tex` to point to new files
- [P0] Archive old figures (fig2_repeated_bad_rate.svg, fig3_decision_breakdown.svg, fig4_trajectory.svg)

### Numbers to find-and-replace in the tex
| Old text | Location | Replace with |
|---|---|---|
| "5 trials" (main campaign) | Abstract, §1, §5.1 | "495 trials across 6 nodes" |
| "val_auc = 0.774711" | §5.1, §5.3 | Context-dependent — see new Table 1 |
| "net gain +0.000933" | §5.1, §5.5 | Relegate to appendix or remove |
| "Kept 2 / discarded 0 / failed 3" | §5.1 | Replace with Table 1 |
| "Acceptance rate = 0.40" | §5.1 | "AR = 0.283 (valid nodes excl. ResNet env failure)" |
| "Invalid rate = 0.60" | §5.1 | "IR = 5.8% (valid nodes excl. ResNet env failure)" |
| "88/88 checks passed" | §5.4 | Re-run artifact checker and update |
| "repeated_bad_rate = 0.60" | §5.3 Arm A | Remove — superseded |
| "repeated_bad_rate = 0.80" | §5.3 Arm B | Remove — superseded |

---

## Minor Issues (from original review)

- [ ] Move the 2/3 replicate result out of the abstract
- [ ] Remove or rework old Table 1 (reads as marketing) — **NEW:** fully replaced by per-node governance table
- [ ] Clarify "auditable" vs. "audited" — paper does not perform an actual audit
- [ ] State that failure taxonomy was derived from observed failures on these nodes, not from first principles
- [ ] Add sentence on future metrics: proposal diversity, time-to-first-improvement, cost-per-kept-trial
- [ ] Check: does "artifact evidence completeness" overlap with "provenance completeness"? If yes, merge

---

## Data Status (as of 2026-05-17)

| Node | Status | Trials | AR | IR | Notes |
|---|---|---|---|---|---|
| lr_synthetic | ✅ Complete | 150 (5s×3a×10) | 0.26–0.28 | 0.0% | Primary ablation node |
| mlp_synthetic | ✅ Complete | 30 (1s×3a×10) | 0.20–0.60 | 0.0% | Single seed — directional |
| openml_bm | ✅ Complete | 80 (4s×20) | 0.242 | 17.5% | Real data, higher IR |
| openml_cg | ✅ Complete | 80 (4s×20) | 0.219 | 8.7% | Real data |
| mlagentbench | ✅ Complete | 20 (2s×10) | 0.60–0.70 | 0.0% | External benchmark |
| resnet_trigger | ❌ Re-run needed | 135 (3s×3a×15) | N/A | 100% | MPS env exhaustion overnight |
| autoresearch_macos | ⏳ Pending | 0 | — | — | Needs `uv run prepare.py` first |

**ResNet re-run plan:** fresh terminal session, `RESNET_TRIGGER_FAST_SEARCH=1`, budget 1 first to validate env, then full 9-campaign ablation. Command in chat from 2026-05-17.

---

## Submission Readiness Gate

Before submitting, confirm all of the following:

- [ ] All P0 items checked (or explicitly deferred with note in Limitations)
- [ ] All P1 items checked (or explicitly deferred)
- [ ] All P2 items checked
- [ ] No quantitative claim in the abstract lacks a CI in §5
- [ ] No reference is unverified
- [ ] The contribution claim in §1 matches what §5 demonstrates
- [ ] ResNet data status resolved (either re-run data or framed as environment stress test)
- [ ] All figures generated from `paper_figures.ipynb` and referenced in main.tex
- [ ] At least one empirically surprising result is present

---

## Quick Status Dashboard

| Section | Items | Done | Remaining |
|---------|-------|------|-----------|
| P0 | 5 | 2.5 | 2.5 |
| P1 | 5 | 1 | 4 |
| P2 | 2 | 0 | 2 |
| Section edits | ~30 | 0 | ~30 |
| Minor | 6 | 0 | 6 |
| **Total** | **~48** | **~3.5** | **~44.5** |

P0-1 = ½ done (reframe done, A/B deferred). P0-2 = ½ done (LR/OpenML/MLA complete, ResNet pending). P0-3 = ½ done (infra done, paper text pending). P0-5 = fully done.

_Update this table as items are checked off._
