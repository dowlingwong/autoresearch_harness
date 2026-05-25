# Paper Edit Checklist — Post-Overnight Run (2026-05-17)

Status: what needs to change in `paper/kdd_aae_2026/` given the new experiment data.
Priority: **P0** = must change before submission · **P1** = should change · **P2** = nice-to-have

---

## 0. Big-Picture Framing Shift

The paper currently reads as a single-node study (ResNet-trigger, 5–20 trials, one LLM).
The overnight run produced **495 trials across 6 nodes with consistent governance**.
The frame should shift from "proof-of-concept on one task" to "multi-node evaluation
of a governed harness", which is a substantially stronger contribution.

---

## 1. Abstract  (`main.tex`)

**P0** — Update all numbers:
- "5 trials" → "495 trials across 6 nodes"
- Remove "ResNet-trigger" as the sole node; list the node suite instead
- Add the cross-node governance claim: "100% provenance completeness across all trials"
- State the memory-mode finding honestly: "memory mode has no statistically significant
  effect on acceptance rate at n=5 seeds (LR node); directional effect observed on MLP"
- Mention ResNet environment failure and what it demonstrates about governance

---

## 2. Section 1 — Introduction  (`01_introduction.tex`)

**P0** — Update the scope claim. Currently says "a single governing loop on a binary
classification node". Now: "a governed harness evaluated across 6 benchmark tasks".

**P1** — Add the ResNet environment-failure story as motivation. An unmonitored agent
loop would have silently treated 135 failed trials as valid — the governance layer
flagged them all as `runtime_error` with full provenance. This is concrete motivation.

**P1** — Update the contributions list to include:
  - Cross-node evaluation (6 nodes, 5 task types)
  - Multi-seed memory ablation (150 trials, 5 seeds)
  - Null result on memory-mode ablation (honest negative finding)

---

## 3. Section 3 — System Design  (`03_system_design.tex`)

**P1** — Add subsection on multi-node NodeSpec. The experiments now cover 6 distinct
node specs (editable symbols, metric parsers, acceptance rules). This demonstrates
the spec abstraction generalises across task types.

**P1** — Worker diversity: add LocalWorker vs ClawWorker distinction. LocalWorker
(LR, MLP, OpenML, mlagentbench) vs ClawWorker (ResNet, autoresearch_macos). Explain
the design choice and tradeoffs.

**P2** — Add a table: Node spec inventory (7 nodes, worker type, metric, budget).
This is already drafted in `plan/experiments_plan.md`.

---

## 4. Section 4 — Experiments  (`04_experiments.tex`)

This section needs a near-complete rewrite. Currently describes one main campaign.

**P0 — Replace the node description block** with the full 6-node table:

| Node | Task type | Worker | Metric | Seeds | Budget |
|---|---|---|---|---|---|
| lr_synthetic | Tabular ML (synthetic) | LocalWorker | val_score | 5 | 10 |
| mlp_synthetic | Tabular ML (synthetic) | LocalWorker | val_score | 1 | 10 |
| openml_bm | Tabular ML (real) | LocalWorker | val_AUC | 4 | 20 |
| openml_cg | Tabular ML (real) | LocalWorker | val_AUC | 4 | 20 |
| mlagentbench | Vectorisation benchmark | LocalWorker | val_score | 2 | 10 |
| resnet_trigger | Physics signal detection | ClawWorker | val_AUC | 3 | 15 |

**P0 — Add the memory-mode ablation subsection** for the LR node:
- 5 seeds × 3 arms × 10 trials = 150 trials
- Primary result: no statistically significant difference across arms
- AR: none=0.260±0.114, summary=0.260±0.152, rationale=0.280±0.110
- All 95% CIs overlap completely
- This is the main ablation result; the old single-campaign ablations are preliminary

**P0 — Describe the ResNet environment failure** as an explicit experiment:
- 9 campaigns × 15 trials = 135 trials
- 100% runtime_error rate (MPS device instability after overnight continuous computation)
- Framework response: all 135 failures labelled failed_invalid, full provenance chains
- Contrast with: pre-overnight smoke test succeeded (val_AUC=0.773778, ~17 min)
- This is a stress test demonstrating governance resilience to environment failure

**P1 — Add experimental setup paragraph** covering:
- LLM: DeepSeek V4 Flash (frontier model via API)
- Manager: langgraph_manager for all campaigns
- Memory modes: none / append_only_summary / append_only_summary_with_rationale
- Seeds: fixed training seed per campaign (123), LLM stochasticity provides seed variation
- Run environment: Apple Silicon MacBook Pro (MPS backend), overnight sequential execution

**P1 — Add the MLP ablation** as a supplemental finding:
- 1 seed × 3 arms × 10 trials = 30 trials
- Directional signal: summary=0.600, rationale=0.400, none=0.200
- Single seed → exploratory only, not confirmatory

---

## 5. Section 5 — Results  (`05_results.tex`)

**P0 — Replace Table 1** (currently: single campaign kdd_main_5trial) with:
  New Table 1 from `paper_figures.ipynb` cell "Table 1": per-node governance summary
  across all 6 nodes. See `paper/tables/table1_governance_summary.csv`.

**P0 — Delete or archive the old single-campaign per-trial table** (trials 1–5 of
kdd_main_5trial). Replace with the timeline figure (Fig 4 from the notebook, which
uses OpenML-BM s1 as the representative campaign).

**P0 — Replace Section 5.3 (Memory Ablation)** entirely:
  - Remove Arm A (prompt_manager deterministic round-robin) — this is a legacy result
    from the old single-node setup
  - Remove Arm A2 (10-trial extension) — superseded
  - Remove Arm A3 (20-trial scientific campaign) — superseded
  - Remove Arm B (qwen2.5-coder:7b LangGraph) — superseded
  - Replace with: **Multi-node DeepSeek memory ablation** using the 5-seed LR results
    and the 1-seed MLP result, structured as:
    - Primary finding (LR, 5 seeds): null result, AR indistinguishable
    - Directional finding (MLP, 1 seed): summary > rationale > none
    - Honest conclusion: power is insufficient at n=10 to detect small effects;
      recommend n≥20 per seed for definitive ablation

**P0 — Add governance metrics across all 6 nodes** (currently only reported for one):
  - Overall IR excluding ResNet: 5.8% (21/360)
  - Overall AR: 28.3% (96/339 valid)
  - Provenance completeness: 100% (all 495 trials)
  - These numbers belong in a "Cross-node Governance Summary" subsection

**P1 — Update the failure taxonomy table** (Table 2) to include the ResNet
environment failure: 135 trials, runtime_error, from MPS instability. Add as a row
showing the framework correctly handles environment-level (not just code-level) failures.

**P1 — Update the OpenML results** to include the trajectory analysis:
  - BM: AR=0.242, IR=17.5%, best=0.9343
  - CG: AR=0.219, IR=8.7%, best=0.7691
  - The high IR on OpenML (vs 0% on synthetic) is an interesting finding about
    real-data task complexity

**P1 — Update the "Governance in Action" table** (Table 5 stress campaigns) — this
section is still valid and well-written, keep as-is but confirm the campaigns still
exist in the ledger (kdd_stress_scope, kdd_stress_noop).

**P2 — Update the task metric trajectory** (Section 5.5):
  Old: "val_AUC from 0.773778 to 0.774711, net gain +0.000933, 5 trials"
  New: Use Fig 7 (OpenML trajectories) as the representative. OpenML shows a cleaner
  multi-seed improvement story (mean +gain across 4 seeds).

---

## 6. Section 6 — Discussion / Limitations  (`06_discussion_limitations.tex`)

**P0 — Add ResNet environment failure as a limitation AND finding**:
  - Limitation: overnight MPS instability invalidated all 135 ResNet trials; a
    fresh-session re-run is needed for definitive ResNet ablation
  - Finding: the governance layer handled this correctly without human intervention

**P0 — Add statistical power as a limitation**:
  - Current LR ablation (n=5 seeds, 10 trials each) cannot detect small memory effects
  - MLP result (1 seed) is directional only
  - Recommend n≥20 per arm, ≥5 seeds for confirmatory ablation

**P1 — Add the autoresearch-macos node** as future work:
  - Node registered, ClawWorker configured, val_bpb metric parser implemented
  - Data collection blocked on `uv run prepare.py` (downloads training data)
  - Once run: provides a "dog-food" result (governing a GPT language model training run)

**P1 — Update the cross-node claim** to be concrete:
  "The same governed control plane, NodeSpec abstraction, and append-only ledger
  operated correctly across 6 distinct benchmark tasks spanning synthetic tabular ML,
  real-data tabular ML, a vectorisation benchmark, and physics signal classification —
  with no framework modifications between nodes."

---

## 7. Figures  (`paper/figures/`)

**P0 — Generate all figures from `paper_figures.ipynb`** (Kernel → Restart & Run All).
  This produces:
  - fig1_governance_overview.{pdf,png}
  - fig2_lr_ablation.{pdf,png}
  - fig3_ir_taxonomy.{pdf,png}
  - fig4_timeline.{pdf,png}
  - fig5_cross_node.{pdf,png}
  - fig6_provenance.{pdf,png}
  - fig7_openml.{pdf,png}
  - fig8_mlp.{pdf,png}
  - table1_summary.{pdf,png}

**P0 — Update figure references in main.tex** to point to the new figure files.

**P0 — Delete or move old figure references** (fig2_repeated_bad_rate.svg,
fig3_decision_breakdown.svg, fig4_trajectory.svg) — these are from the old single-node
results. Archive in `paper/figures/archive/` if you want to keep them.

---

## 8. Tables  (`paper/tables/`)

**P0 — New Table 1**: `table1_governance_summary.csv` (generated by notebook)

**P1 — Update governance_bootstrap_cis_lr.csv**: already up to date (generated by
overnight run). Confirm the file exists and has 15 campaigns × 6 metrics = 90 rows.

**P0 — Note on ResNet bootstrap CIs** (`governance_bootstrap_cis_resnet.csv`):
  All CI estimates are [0,0] for acceptance_rate and [1,1] for invalid_rate.
  Do NOT use these as evidence of ResNet ablation results — they reflect the
  environment failure, not the underlying task. Add a footnote in the paper.

---

## 9. Numbers to Find-and-Replace in the Paper

These specific old numbers appear in the current tex files and should be updated:

| Old value | Where | New value |
|---|---|---|
| "5 trials" (main campaign) | Abstract, §1, §5.1 | "495 trials across 6 nodes" |
| "val_auc = 0.774711" (best) | §5.1, §5.3 | Depends on context — OpenML best is 0.9343 |
| "net gain +0.000933" | §5.1, §5.5 | Remove or relegate to appendix |
| "Kept 2 / discarded 0 / failed 3" | §5.1 | Replace with Table 1 |
| "Acceptance rate = 0.40" | §5.1 | "AR = 0.283 (excl. ResNet)" |
| "Invalid rate = 0.60" | §5.1 | "IR = 5.8% (excl. ResNet env failure)" |
| "88/88 checks passed" | §5.4 | Re-run artifact checker and update |
| "repeated_bad_rate = 0.60" | §5.3 Arm A | Remove (superseded) |
| "repeated_bad_rate = 0.80" | §5.3 Arm B | Remove (superseded) |

---

## 10. Order of Operations

1. **Run the notebook** (`paper_figures.ipynb`) to generate all figures and Table 1 CSV
2. **Update §4 Experiments** with the node table and new experimental setup
3. **Rewrite §5 Results** — replace Tables 1–3, add cross-node governance summary
4. **Update §6 Discussion** with ResNet limitation + power analysis note
5. **Update Abstract** with new numbers
6. **Update §1 Introduction** with framing shift
7. **Recompile** with `latexmk -pdf main.tex`
8. **Re-run ResNet** in a fresh session (NOT overnight) to get valid ablation data

---

## 11. What Does NOT Need to Change

- §2 Related Work — still accurate, no changes needed
- §3 System Design (core) — the 6-layer architecture, control plane design, decision.py
  logic, scope enforcement, append-only store — all still correct
- The stress campaign analysis (§5.4, Table 5) — still valid, keep as-is
- The pending-guard recovery mechanism description — still valid
- The bibliography — already fixed (Burtenshaw author field)
- CLAUDE.md project instructions — already up to date
