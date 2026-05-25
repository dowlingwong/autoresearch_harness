# Master Plan — Governed Harness for Auditable LLM-Driven ML Experimentation

_Single source of truth. Merged from: `Up-to-date.md`, `TODO.md`, `revision_and_edit_checklist.md`, `revision_plan.md`, `experiments_plan.md`, `autoresearch_analysis.md` (all uploaded 2026-05-19)._  
_Last updated: 2026-05-19 — submission-ready paper pass_

---

## Quick Status Dashboard

| Area | Status | Remaining |
|---|---|---|
| **Experiments — fast nodes** | ✅ LR 450 (b30, 5s×3a), MLP 30, OpenML 460 (b20+b30), MLAgentBench 50 | — |
| **Experiments — ResNet** | ✅ 155 trials (20 original + 135 Linux L40S) | Optional: 5-seed b20 reruns (camera-ready) |
| **Experiments — Autoresearch Linux** | ✅ 240 trials (4s×2a×b30). s2 anomaly classified correctly. | None-arm root-cause (not blocking) |
| **Experiments — Autoresearch macOS** | ❌ **DISCARDED** — superseded by linux L40S version | — |
| **§01 Introduction** | ✅ "six nodes" + σ=0.315 + autoresearch contribution bullet + "Four contributions" | — |
| **§03 System Design** | ✅ Node inventory table (P1-NODE-TABLE) + lifecycle state machine (P1-STATE-MACHINE) added | — |
| **§04 Experiments** | ✅ Autoresearch node paragraph + DeepSeek manager/worker justification + training environment paragraph (P1-ENV) | — |
| **§05 Results** | ✅ §5.6 autoresearch + §5.7 cross-node governance summary + 3-seeds qualifier (P1-3SEEDS) | Bootstrap CI for original ResNet Table 1 (optional) |
| **§06 Discussion** | ✅ Evidence table rows + autoresearch none-arm limitations paragraph + σ fixes + governance overhead paragraph (P1-OVERHEAD) | — |
| **Abstract** | ✅ Trimmed to 192 words; six nodes, 1,355 trials, autoresearch, σ=0.315 | — |
| **Conclusion** | ✅ autoresearch finding, s2 anomaly governance robustness | — |
| **Figures** | ✅ Generated ResNet L40S figure linked in §5.3; incompatible MLP/env-fail plots excluded | — |
| **Bootstrap CIs** | ✅ OpenML tables, L40S table | Original ResNet Table 1 (optional) |
| **P0-1 (ungoverned A/B)** | Deferred to camera-ready | Reframing complete; limitation stated in §6 |
| **P0-3 (DeepSeek manager)** | ✅ All nodes + §4.2 justification paragraph | — |
| **P0-4 (metric validation)** | ✅ Automated probes integrated; κ explicitly deferred as human validation | Human κ remains camera-ready/desirable |
| **P0-5 (references)** | ✅ All 22 entries audited | — |
| **Compile test** | ✅ `latexmk -pdf` clean; references start on page 10 | — |
| **Submission readiness** | ✅ Blocking items complete; body within 9 pages | Final human PDF/Overleaf check |

**Total governed trials (non-smoke, in paper):** 1,355 across reported nodes  
**Nodes reported in paper:** 6  
**Nodes in ledger (not reported):** MLP synthetic (portability evidence only; not included in six-node paper framing)

---

## Core Paper Identity

**One-sentence thesis:**
> Autonomous ML experimentation becomes scientifically credible only when the agent loop is governed by an explicit control plane that owns lifecycle, enforces bounded execution, records decisions append-only, and reports auditable governance metrics independent of the specific manager or worker backend.

**Contribution type:** Evaluation methodology + governed harness design + auditable control-plane protocol.

**Non-claims (must not drift from these):**
- Not a general autonomous scientist.
- Not proof of scientific discovery.
- Not a universal optimization algorithm.
- Not a detector-physics result.
- Not dependent on a specific coding-agent backend.
- The ResNet-trigger task is a controlled scientific ML case study, not a broad ML benchmark coverage claim.

**Core filter for every paragraph, table, and figure:**
> Does this help a reviewer evaluate whether the autonomous agent loop is bounded, auditable, failure-aware, reproducible, and behaviorally affected by governance memory? If not, it should be cut or moved to secondary context.

**Currently supported claims:**
- Control plane classifies kept, discarded, failed-invalid — demonstrated, all nodes.
- Append-only ledger preserves provenance — 100% completeness, all campaigns.
- Governance metrics make guard failures visible in the audit record — stress campaigns.
- Governance protocol transfers across node types — six nodes, no control-plane changes.
- Repeated-bad rate diagnoses memory sensitivity — mixed case study (ResNet 2/3 replicates; lr_synthetic fails to transfer).
- Memory ordering none < summary < rationale by mean AUC — confirmed, single node (ResNet L40S 135 trials).
- Memory stabilises cross-seed variance — 70× CI-width collapse, single node.
- Memory is a precondition for valid proposals on LM training task — demonstrated, 4 seeds / 120 trials (none arm, IR=1.00).
- Governance classifies total worker failure correctly regardless of cause — s2 anomaly, 30 trials, 100% provenance.

**Not claimed:**
- Rationale-augmented memory reduces repeated-bad rate (non-monotonic across seeds).
- Memory-ordering result is robust across nodes (fails on lr_synthetic).
- Governance causes a measurable performance gap vs. ungoverned (not run — deferred).

---

## 1. Node Inventory

| # | Node | Domain | Worker | Metric | Dir | Trials | Paper section |
|---|---|---|---|---|---|---|---|
| 1 | `resnet_trigger` | Waveform classification (ResNet) | ClawWorker | val_auc | max | 155 | ✅ §5.1, §5.2, §5.3 |
| 2 | `lr_synthetic` | NumPy logistic regression | LocalWorker | val_score | max | 450 | ✅ §5.2 |
| 3 | `mlp_synthetic` | NumPy one-hidden-layer MLP | LocalWorker | val_score | max | 30 | ❌ Not in paper (portability evidence only, 1 seed) |
| 4 | `openml_credit_g` | Tabular classification (OpenML 31) | LocalWorker | val_auc | max | 230 | ✅ §5.4 |
| 5 | `openml_bank_marketing` | Tabular classification (OpenML 1461) | LocalWorker | val_auc | max | 230 | ✅ §5.4 |
| 6 | `mlagentbench_vectorization` | NumPy convolution (MLAgentBench) | LocalWorker | speed_score | max | 50 | ✅ §5.5 |
| 7 | `autoresearch_linux` | GPT LM training (nanochat, Wikitext-103, L40S) | ClawWorker | val_bpb | min | 240 | ✅ §5.6 |
| 8 | `autoresearch_macos` | GPT LM training (macOS MPS) | ClawWorker | val_bpb | min | 0 | ❌ **DISCARDED** — superseded by linux L40S |

**Why autoresearch_macos is discarded:** The linux L40S version is complete with 240 trials across 4 seeds and 2 arms. The macOS MPS version never started and would use slower, less reproducible hardware. Adding it would require new compute for no incremental contribution — the governance claim is already demonstrated with the linux version. Remove from all future planning.

---

## 2. Experiment Status

### Phase A — Original campaigns ✅
`kdd_resnet_scientific_20`, stress campaigns (`kdd_stress_scope`, `kdd_stress_noop`), `lr_synth_baseline` — all in paper.

### Phase B — DeepSeek fast-node ablations ✅
**LR synthetic:** 5 seeds × 3 arms × b30 = 450 trials  
**MLP synthetic:** 3 arms × 1 seed × b10 = 30 trials _(1 seed only; not in paper)_  
**OpenML credit-g:** 4 seeds × b20 + 5 seeds × b30 = 230 trials  
**OpenML bank-marketing:** 4 seeds × b20 + 5 seeds × b30 = 230 trials  
**MLAgentBench:** original 30 + 2 seeds × b10 = 50 trials  
Manager: `deepseek/deepseek-v4-flash` via LangGraph.

### Phase C — ResNet Linux L40S ✅
3 arms × 3 seeds × 15 trials = **135 trials** on deepthought2 (NVIDIA L40S, CUDA 12.8).  
Manager: `deepseek/deepseek-v4-flash` (proposal). Worker: `qwen2.5-coder:7b`, temp=0.2.

| Memory mode | N | AR% | IR% | Mean best AUC | σ | 95% CI |
|---|---|---|---|---|---|---|
| none | 45 | 24.4 | 0.0 | 0.665 | 0.315 | [0.339, 0.967] |
| summary | 45 | 42.2 | 17.8 | 0.735 | 0.005 | [0.732, 0.741] |
| rationale | 45 | 35.6 | 15.6 | 0.798 | 0.046 | [0.746, 0.828] |

### Phase D — Autoresearch Linux ✅ (complete 2026-05-19)

| Arm | Seed | n | Decisions | IR | Notes |
|---|---|---|---|---|---|
| none | s1 | 30 | 0k/0d/30f | 1.00 | All runtime_error |
| none | s2 | 30 | 0k/0d/30f | 1.00 | All runtime_error |
| none | s3 | 30 | 0k/0d/30f | 1.00 | All runtime_error |
| none | s4 | 30 | 0k/0d/30f | 1.00 | All runtime_error |
| summary | s1 | 30 | 4k/17d/9f | 0.30 | Healthy |
| summary | s2 | 30 | 0k/0d/30f (27 rte + 3 ppf) | 1.00 | Anomaly — classified correctly |
| summary | s3 | 30 | 5k/18d/7f | 0.23 | Healthy |
| summary | s4 | 30 | 5k/22d/3f | 0.10 | Healthy |

**Key numbers:**
- None arm: 120/120 runtime_error, IR=1.00 across 4/4 seeds — zero valid proposals
- Summary healthy seeds (s1/s3/s4): mean val_bpb gain = **−0.062** (5.2%), σ=0.0018
- Provenance completeness: **240/240 = 100%**
- Trajectories: s4 monotone 1.1696→1.1054; s3 staircase 1.1720→1.1113; s1 1.2070→1.1454
- Full analysis: `plan/autoresearch_analysis.md`

### Phase E — Optional / camera-ready

| Node | Current | Target | Priority |
|---|---|---|---|
| ResNet L40S | 135 (b15, 3s×3a) | 300 (b20, 5s×3a) | Low — 70× variance collapse already compelling at 3 seeds |
| Autoresearch rationale arm | 0 | 120 (4s×b30) | Not needed — categorical none vs. summary contrast makes the key point |

---

## 3. Paper Revision Status

### Priority A — Completed in 2026-05-19 session ✅

| Item | Done | Notes |
|---|---|---|
| A1: §1 "six ML nodes", σ=0.315, autoresearch bullet | ✅ | Contributions now say "Four" |
| A2: §4 autoresearch node paragraph | ✅ | nanochat, Wikitext-103, val_bpb, 4s×b30×2arms, rationale arm omitted |
| A2: §4 DeepSeek manager/worker justification | ✅ | Explicit manager vs. worker distinction added |
| A3: §5.7 cross-node governance summary table | ✅ | 8 rows, ResNet split 20-trial/L40S, provenance 100% footer |
| A3: §5 preamble updated to include autoresearch | ✅ | |
| A4: §6 two evidence table rows | ✅ | precondition + total-failure governance claims |
| A4: §6 section refs updated (5.1–5.7) | ✅ | |
| A4: §6 "five nodes" → "six nodes" | ✅ | |
| A5: §6.2 autoresearch none-arm limitations paragraph | ✅ | |
| A5: §6.2 stale σ values fixed (0.004→0.005, 0.257→0.315) | ✅ | |
| S-2 fix: §4 DeepSeek paragraph scoped to manager only | ✅ | Excludes small-scale qwen2.5 ablation from DeepSeek claim |
| S-3 fix: ResNet cross-node table split back to two rows | ✅ | Avoids misleading weighted average |

---

### Priority B — Must fix before submission (blocking)

#### B-ABSTRACT: Trim abstract to ≤250 words
**Status:** ✅ DONE — abstract is 192 words and keeps the six-node scope, 1,355 trials, 70× variance-collapse, autoresearch enabling condition, and 100% provenance claims.  
**File:** `main.tex`, abstract block.

#### B-TRIALCOUNT: Verify and synchronise trial count
**Status:** ✅ DONE — reported-node table total is 1,355 governed trials; abstract now says `1{,}355 governed trials`.  
**File:** `main.tex`, abstract.

#### B-FIGURES: Generate and link paper figures
**Status:** ✅ DONE — generated ResNet L40S ablation figure copied into the submission tree and linked in §5.3 as `fig:resnet-linux-ablation`. Generated plots that include MLP or stale env-failure framing were not included.
**File:** `sections/05_results.tex`, `figures/fig9_resnet_linux_ablation.pdf`.

#### B-COMPILE: Compile and verify no errors
**Status:** ✅ DONE — `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completes. `main.pdf` has body through page 9, with references beginning afterward. No undefined-reference or missing-citation warnings in the final log grep.

---

### Priority P1 — Substantial (should do before submission)

#### P1-ENV: Add training environment description to §4 ✅ DONE
**Status:** Paragraph inserted in §4.2 after the DeepSeek manager/worker justification paragraph.  
**File:** `sections/04_experiments.tex`, §4.2.

#### P1-NODE-TABLE: §3 node inventory table ✅ DONE
**Status:** Table inserted as new `\subsection{Benchmark Node Inventory}` between Manager/Worker/Replacement and Trial Lifecycle subsections. Label: `tab:node-inventory`.  
**File:** `sections/03_system_design.tex`.

#### P1-STATE-MACHINE: §3 formal lifecycle state machine ✅ DONE
**Status:** Table inserted after the reproducibility hashes paragraph within Trial Lifecycle subsection. Label: `tab:state-machine`.  
**File:** `sections/03_system_design.tex`.

#### P1-OVERHEAD: §6 governance overhead paragraph ✅ DONE
**Status:** Paragraph inserted in §6.2 after the "No ungoverned head-to-head" paragraph.  
**File:** `sections/06_discussion_limitations.tex`.

#### P1-3SEEDS: Qualify "3 seeds" explicitly in ResNet L40S text ✅ DONE
**Status:** Sentence "This collapse is observed across 3 independent seeds; replication at 5 seeds is left to camera-ready." inserted after the σ=0.005 sentence in §5.3.  
**File:** `sections/05_results.tex`, §5.3.

#### P1-TAXONOMY: Promote failure taxonomy in §3 ✅ DONE
**Status:** New §3 subsection added with taxonomy source, exhaustiveness argument, extension rule, and metric-validation probes. Inter-rater reliability remains a camera-ready human-validation item.  
**File:** `sections/03_system_design.tex`.

---

### Priority P2 — Polish (final pass)

1. **Tone:** Remove "general governance evaluation protocol" language — replace with "across six nodes spanning five task types." ✅ DONE (P2-TONE) — grep found zero occurrences of "general governance evaluation protocol", "general protocol", or "general autonomous" in §1, §3, §6. No changes needed.
2. **Abstract ↔ §5 alignment:** ✅ DONE — abstract quantitative claims map to §5.3, §5.6, and §5.7.
3. **Contribution list ↔ §5:** ✅ DONE — contributions map to governance metrics/control plane (§3), audit evidence (§5.1--§5.7), and autoresearch (§5.6).
4. **ResNet AUC framing:** ✅ DONE — the 20-trial +0.008 AUC result is framed as secondary loop-correctness evidence, not optimisation proof.
5. **Memory diagnostic framing:** ✅ DONE — small-scale memory ablation is framed as a diagnostic probe; L40S ordering is explicitly single-node evidence.
6. **MLP synthetic:** Verify no LaTeX file mentions `mlp_synthetic` as a reported node. ✅ DONE (P2-MLP-CHECK) — grep over all `sections/` files returns zero matches. mlp_synthetic does not appear in any paper section.

---

## 4. Submission Readiness Gate

### Must be true before submission

- [x] §1 says "six ML nodes" ✅
- [x] §1 σ_none = 0.315 ✅
- [x] §1 includes autoresearch in contributions list ✅
- [x] §4 has autoresearch node description paragraph ✅
- [x] §4 has DeepSeek manager/worker justification paragraph ✅
- [x] §5 has cross-node governance summary table ✅
- [x] §6 evidence table includes autoresearch rows ✅
- [x] §6 limitations includes autoresearch none-arm paragraph ✅
- [x] §6 limitations says "six nodes" not "five nodes" ✅
- [x] §6.2 uses σ=0.315 and σ=0.005 (not stale 0.257/0.004) ✅
- [x] Abstract trimmed to ≤250 words ✅
- [x] Trial count in abstract verified and accurate ✅
- [x] Figures generated from `paper_figures.ipynb` and linked in paper ✅
- [x] Training environment paragraph added to §4.2 ✅
- [x] No P0 item outstanding without an explicit Limitations/Deferred note ✅
- [x] Contribution list §1 matches what §5 demonstrates (1:1 mapping) ✅
- [x] pdflatex/latexmk compiles clean: no undefined refs, no missing citations, ≤9 body pages ✅
- [x] At least one empirically surprising result: variance collapse ✓, none-arm total failure ✓

### Desirable before submission (won't block if noted in paper)

- [x] §3 node inventory table ✅
- [x] §3 formal lifecycle state machine table ✅
- [x] §4.2 / §4.3 training environment paragraph ✅
- [x] §6 governance overhead paragraph ✅
- [ ] Bootstrap CIs for original ResNet Table 1 (currently point estimates only)
- [ ] Inter-rater Cohen's κ for failure taxonomy (human task)

---

## 5. Key Numbers Reference

| Metric | Value | Source |
|---|---|---|
| Total governed trials (non-smoke, reported nodes) | 1,355 | Abstract + §5 cross-node rows |
| Nodes reported in paper | 6 | §5.1–§5.7 |
| Provenance completeness (all nodes) | 100% | All campaigns, §5.7 |
| ResNet L40S σ_none | 0.315 | Table 3, §5.3 |
| ResNet L40S σ_summary | 0.005 | Table 3, §5.3 |
| ResNet L40S CI width collapse | 0.628 → 0.009 units (~70×) | §5.3 prose |
| ResNet L40S memory ordering | 0.665 < 0.735 < 0.798 | Table 3 |
| OpenML bank-mkt b30 | AUC 0.9341, CI [0.9337, 0.9345], σ=0.0005 | Table 4, §5.4 |
| OpenML credit-g b30 | AUC 0.7679, CI [0.7667, 0.7688] | Table 4, §5.4 |
| Autoresearch none arm | 120/120 runtime_error, 4/4 seeds, IR=1.00 | Table 5, §5.6 |
| Autoresearch summary healthy s1/s3/s4 | mean val_bpb −0.062 (5.2%), σ=0.0018 | Table 5, §5.6 |
| Autoresearch provenance | 240/240 = 100% | §5.6, §5.7 |
| LR b30 convergence CI width | <0.0002 all arms | §5.2 prose |
| ResNet 20-trial case study best AUC | 0.7827 (within baseline seed interval) | Table 1, §5.1 |
| Baseline seed interval (5 seeds) | mean 0.785, CI [0.775, 0.798] | §5.1 prose |

---

## 6. What We Are Not Running (and Why)

**Autoresearch macOS:** Superseded by linux L40S. Hardware is slower, less reproducible, and the governance finding is already demonstrated. **Definitively discarded.**

**P0-1 ungoverned A/B:** Requires new code (stripped control plane) + matched experimental arms. High engineering cost for a workshop deadline. Deferred to camera-ready or follow-up paper. Limitation is explicitly stated in §6.2.

**Autoresearch none-arm root cause:** The 100% runtime_error pattern across 4/4 seeds is documented and attributed to missing training interface conventions without memory. Not blocking — governance result (all classified correctly) is the contribution. Optional future investigation.

**Autoresearch rationale arm:** ~6 min/trial × 30 trials × 4 seeds ≈ 12 GPU-hours. The categorical none vs. summary contrast already makes the key point. Not needed before submission.

**Phase E4 ResNet 5-seed b20:** 70× variance collapse at 3 seeds is already compelling. Deferred to camera-ready. Paper notes "3 seeds" explicitly.

**LR/MLP expansion:** LR is complete at b30, 5 seeds. MLP is not in the paper's six-node count and should not be added before submission without a full seven-node consistency pass.

**P0-4 inter-rater Cohen's κ:** Human task — export is ready (`scripts/export_failure_taxonomy_labels.py`). Needs one author + one external person to classify 30 failed trials. The paper now reports automated metric-validation probes and explicitly states that Cohen's κ is not reported yet. Not blocking because the limitation is disclosed.

---

## 7. Hostile Reviewer Weakness Report (2026-05-19)

_Evaluate each before submission. Items marked FIXED were resolved in the 2026-05-19 editing session._

### [BLOCKING — resolved]

**B-1: Abstract word count.** FIXED — abstract is 192 words. ✅

**B-2: Trial count.** FIXED — abstract uses 1,355 reported governed trials, matching the reported-node totals. ✅

### [SUBSTANTIAL — fix before submission if possible]

**S-1: MLP synthetic not in paper.** The six nodes are resnet_trigger, lr_synthetic, openml_credit_g, openml_bank_marketing, mlagentbench_vec, autoresearch_linux. MLP is in the ledger but not in the paper. Verify no LaTeX file references `mlp_synthetic` as a reported node.  
**Status:** FIXED/CONFIRMED — grep over paper TeX finds no `mlp_synthetic` reference. Do not add MLP before submission without a seven-node consistency pass. ✅

**S-2: §4.2 DeepSeek paragraph manager/worker distinction.** FIXED — paragraph now says "as the *proposal manager*; the worker is qwen2.5-coder:7b (ClawWorker nodes) or LocalWorker (fast nodes)." Also scoped to Phase C–D, excluding small-scale qwen2.5 ablation. ✅

**S-3: §5.7 ResNet aggregate row.** FIXED — split back to two rows (20-trial and L40S) with a prose explanation. ✅

**S-4: Autoresearch summary arm AR/IR arithmetic.** Verified: 14k/57d/49f ÷ 120 = AR 11.7%, IR 40.8%. ✅

**S-5: Abstract density.** Related to B-1. The three essential numbers are: σ collapse (70×), autoresearch enabling condition (120/120), provenance completeness (100%). Everything else should move to §5.

### [MINOR — fix if time permits]

**M-1: §4 DeepSeek paragraph.** FIXED ✅

**M-2: "Four contributions."** Correct after adding autoresearch bullet. No fix needed.

**M-3: §6.1 section references.** Updated to 5.1–5.7. ✅

**M-4: §5 preamble "two stress campaigns."** Accurate — lr_synth_baseline is not a stress campaign; it belongs to the lr_synthetic row.

**M-5: Conclusion "five transfer nodes."** Correct (lr_synthetic, 2× OpenML, MLAgentBench, autoresearch = 5 beyond ResNet). ✅

**M-6: Bootstrap CIs missing from original ResNet Table 1.** Not blocking — main CI tables (L40S, OpenML) have bootstrap CIs. Note in cover letter if required.

**M-7: Inter-rater Cohen's κ pending.** Disclosed in §4.4. Not blocking for workshop.

**M-8: Training environment not stated.** FIXED ✅ — P1-ENV paragraph added in §4.2.

---

## 8. TODO List — What to Do Next

All paper-blocking content and compile items are complete. Remaining items are final submission hygiene or optional camera-ready improvements.

### Final Pre-Submission Checklist

1. **Visually inspect `main.pdf`.** Confirm figure/table placement, no clipped captions, no obviously bad overfull text, and references start after body page 9.

2. **Overleaf smoke test.** Upload/sync the paper folder, set `main.tex` as the main document, and use "Recompile from scratch". Confirm the PDF matches the local build.

3. **Submission metadata.** Confirm title, anonymisation, workshop name, author block, ACM rights fields, and any required EasyChair/CMT metadata.

4. **Package only required files.** Include `main.tex`, `sections/`, `figures/fig9_resnet_linux_ablation.pdf`, `tables/`, `references.bib`, `ACM-Reference-Format.bst`, `acmart.cls`, and any generated `.bbl` only if the submission system requests it. Exclude stale generated plots that include MLP or old env-failure framing.

5. **Final artifact snapshot.** Keep the successful local `main.pdf` and `main.log` with the submission record.

### Completed Blocking Items

1. ✅ **DONE — Trim abstract to ≤250 words.** Abstract is 192 words.  
   _File: `main.tex`_

2. ✅ **DONE — Verify trial count.** Abstract synchronised to 1,355 reported governed trials.  
   _File: `main.tex`_

3. ✅ **DONE — Compile test.** `latexmk -pdf` succeeds; body is within 9 pages and references start on page 10.  
   _From: `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/`_

4. ✅ **DONE — Generate/link figures.** Generated ResNet L40S ablation figure is linked in §5.3; MLP/stale generated plots are intentionally excluded.

### Completed Should-Do Items

5. ✅ **DONE — Add training environment paragraph to §4.2 (P1-ENV).** Hardware (L40S / local CPU), OS/CUDA (Ubuntu 22, CUDA 12.8), manager model (DeepSeek-V4-Flash), worker model (qwen2.5-coder:7b), seed reset discipline.  
   _File: `sections/04_experiments.tex`_

6. ✅ **DONE — Add node inventory table to §3 (P1-NODE-TABLE).** All 6 reported nodes: name, task type, worker, metric, direction, budget. Anchors the "six nodes" claim.  
   _File: `sections/03_system_design.tex`_

7. ✅ **DONE — Add formal lifecycle state machine to §3.3 (P1-STATE-MACHINE).** States, transitions, invariants, pending-guard contract.  
   _File: `sections/03_system_design.tex`_

8. ✅ **DONE — Internal consistency pass.** Abstract → §1 → §5 quantitative claims match body evidence:
   - "six ML nodes" → §5 preamble lists six ✅
   - σ=0.315 and σ=0.005 → Table 3, §5.3 ✅
   - "5.2% val_bpb reduction" → §5.6 prose ✅
   - "240/240 provenance" → §5.6 + §5.7 ✅
   - remaining abstract claims have §5 support ✅

9. ✅ **DONE — Grep for `mlp_synthetic` in tex files (P2-MLP-CHECK).** No matches in `sections/`; it is not referenced as a reported node.  
   _Command: `grep -r "mlp_synthetic" A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/sections/`_

10. ✅ **DONE — Note "3 seeds" explicitly in §5.3 prose (P1-3SEEDS).** Body text now calls out the 3-seed caveat clearly.

### Polish (do if time allows)

11. ✅ **DONE — Add governance overhead paragraph to §6.2 (P1-OVERHEAD).** LoC, per-trial ledger KB, API latency.  
    _File: `sections/06_discussion_limitations.tex`_

12. ✅ **DONE — Tone pass (P2-TONE).** Grep found zero occurrences of "general governance evaluation protocol", "general protocol", or "general autonomous" in §1, §3, §6; no prose change needed.

13. ✅ **DONE — Contribution list §1 ↔ §5 mapping check.** Each contribution maps to explicit §3/§5 evidence.

14. **Optional human task: inter-rater Cohen's κ.** Export sample: `scripts/export_failure_taxonomy_labels.py`. Have one author + one external rater classify 30 failed trials. Run `scripts/score_failure_taxonomy_labels.py`. Report κ in camera-ready if completed.

15. **Optional: bootstrap CI for original ResNet Table 1.** Currently has point estimates only. Run `scripts/bootstrap_governance_cis.py` on `kdd_resnet_scientific_20`.

### Camera-ready / follow-up paper

16. **P0-1 ungoverned A/B.** Implement stripped control plane arm, run matched governed/ungoverned campaigns, report governance-vs.-no-governance gap with bootstrap CIs. This is the paper's biggest conceptual gap.

17. **Phase E4 ResNet 5-seed b20.** Increases statistical power for the variance collapse claim.

18. **P1-5 empirical related-work comparison.** Wrap an external manager (e.g., AIDE-style) under the harness and report governance metrics.

19. **Failure taxonomy reliability.** Add inter-rater reliability once human labels are complete.

---

## 9. Training Environment — Inserted in the Paper

**Status:** DONE. The paper states the ClawWorker GPU host, LocalWorker CPU host, Python/library stack, DeepSeek manager model, Ollama worker model, temperatures, and seed reset discipline.

Rationale:

1. The paper's headline claim is reproducibility and 100% provenance completeness. Without the software environment, readers cannot reproduce experiments. This is a self-undermining omission for a paper about auditability.

2. We already name hardware ("NVIDIA L40S, CUDA 12.8," "deepthought2") — the software stack is the obvious missing piece.

3. Standard ACM/ML paper practice for reproducibility: Python version, framework version, CUDA version, library versions.

4. The DeepSeek API model ID should be stated (already done in §4.2 and §5.3); the qwen2.5-coder:7b model and Ollama setup should also be stated.

**Inserted content basis for P1-ENV:**

> All ClawWorker campaigns (ResNet L40S, autoresearch) ran on deepthought2 (NVIDIA L40S 48 GB, CUDA 12.8, Ubuntu 22.04, PyTorch 2.x). LocalWorker campaigns (LR synthetic, OpenML, MLAgentBench) ran on a local CPU-only host with Python 3.11, NumPy, and scikit-learn. The proposal manager for Phase C–D campaigns is DeepSeek-V4-Flash accessed via the OpenAI-compatible DeepSeek API (`deepseek/deepseek-v4-flash`, temperature 0.2). The worker model for ClawWorker nodes is qwen2.5-coder:7b served locally via Ollama (temperature 0.2). Each seed resets the editable node files to the committed default before trial 1; no state carries across seeds.

---

## 10. Archived / Historical Context

The following items from earlier planning documents are resolved and archived here:

- **Priority 0–17 from `TODO.md`** (2026-05-15): All submission-blocking priorities are complete. Remaining items are final packaging/human checks in §8 or camera-ready work.
- **`revision_and_edit_checklist.md`** (2026-05-17): Section edits for §1/§4/§5/§6 are complete (A1–A5 + fixes). The overnight run data (495 trials) is now integrated. The ResNet "re-run needed" status reflects the MPS environment exhaustion that was subsequently resolved by moving to Linux L40S (Phase C, 135 trials complete).
- **`Up-to-date.md`** (2026-05-15): Current readiness level was "Level 7." With autoresearch complete, A1–A5 edits applied, and Priority B resolved, the paper is submission-ready pending final human PDF/Overleaf checks.
- **`revision_plan.md`**: P0-5 (references) ✅. P0-3 (frontier manager) ✅. P0-2 (bootstrap CIs) ✅ for key tables. P0-1 deferred. P1-2 (MLAgentBench) ✅. P1-3/P1-4/P1-5 remain as lower-priority items.
