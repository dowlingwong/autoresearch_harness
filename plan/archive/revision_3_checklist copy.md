# Revision 3 Checklist — Updated Against Real Paper

Audited against: `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/sections/`
Audit date: 2026-05-21
Status of s1–s5 clean rerun: **COMPLETE** (all 225 trials, shared `node_state_hash = 9c0118f33fcb`).

Legend:
- `[x]` complete / confirmed in real paper
- `[ ]` still pending
- `[H]` human-only task (cannot be done by Claude)

---

## 0. Non-Negotiable Constraints

- [ ] Main paper fits the 9-page workshop limit excluding references. **Unverified — PDF not compiled yet.**
- [x] Paper is anonymous.
- [x] ACM conference proceedings format (`acmart`, `sigconf` option).
- [x] Main paper is self-contained; extended details in `supplement.tex`.
- [x] Central claim is governance/auditability, not memory or task performance.
- [x] 70× result absent from abstract.
- [x] No detailed seed-specific memory claims in introduction.
- [x] Memory language in conclusion is softened to "diagnostic probe, node-specific."

---

## 1. Critical Credibility Fixes

### 1.1 s1–s5 Clean Rerun — COMPLETE

- [x] All five seeds rerun with verified frozen baseline (`node_state_hash = 9c0118f33fcb`).
- [x] §5.3 (Large-Scale Replication) updated: ordering confirmed 5 seeds, 0.899 < 0.938 < 0.941.
- [x] §5.3 variance updated: σ_none = 0.009 vs σ_summary = σ_rationale = 0.002 (4–5× reduction).
- [x] Reset-semantics disclosure replaced with clean-rerun description.
- [x] Evidence-strength table (§6) updated: "Confirmed, 5 clean seeds."
- [x] Limitations memory paragraph updated: 5-seed σ values, no s4/s5 anomaly warning.
- [x] Conclusion updated: "across all five clean ResNet seeds (0.899 < 0.938 < 0.941)" — **fixed this session**.
- [x] Cross-node summary table (Table 2): L40S row shows AR=46.7%, IR=0.4%, "(5 clean seeds verified)".

### 1.2 Trial-Count Consistency

- [x] Abstract uses 1,445.
- [x] Ungoverned counterfactual section clarifies 1,135 as primary-campaign subset.
- [x] Tables consistent with this split.
- [ ] README still says 1,355 — **out of scope per prior decision; leave for post-submission**.

### 1.3 Internal Consistency

- [x] No unqualified "six failure categories."
- [x] `degraded_metric` framed as discarded, not failed-invalid.
- [x] No-ungoverned-head-to-head limitation preserved.
- [x] "Sharpest memory-effect result" absent from conclusion.
- [x] No Markov paragraph (method section uses lifecycle table + figure only).
- [x] MDP/policy-optimization language absent.

---

## 2. Paper Structure — All Sections

### Introduction (§1)
- [x] Opens on the evaluation gap (not memory).
- [x] Governance/control-plane perspective introduced in first two paragraphs.
- [x] Manager/worker/control-plane authority separation introduced.
- [x] Evidence paragraph is governance-first (lifecycle, provenance, failure taxonomy).
- [x] Exactly **four contributions** listed.
- [x] Memory mentioned only as diagnostic probe.
- [x] 1,445 trials stated.

### Related Work (§2)
- [x] Compressed to three bold-paragraph blocks (no subsections): ML-agent benchmarks / experiment tracking / harness engineering.
- [x] Required contrast sentence present ("not another task benchmark — a governance layer").
- [x] Zaharia et al. compound-AI citation replaced with correct entry.

### System Design / Method (§3)
- [x] Manager/worker/control-plane separation clearly present.
- [x] Replacement principle, NodeSpec contract, lifecycle states, pending guard, scope validator, no-op guard, append-only ledger, failure taxonomy all present.
- [x] No Markov paragraph.
- [x] Lifecycle figure present.
- [x] Concise lifecycle state table present.

### Governance Metrics (§4)
- [x] Governance metric table exists.
- [x] Metrics framed alongside task metrics.
- [x] Failure labels are control-plane-assigned.
- [x] Metric definitions consistent with tables and prose.

### Evaluation Setup (§4 / node inventory)
- [x] Six-node inventory table present.
- [x] "Purpose in evaluation" column present.
- [x] Campaign IDs/hostnames moved to supplement.
- [x] /ceph artifact archive location disclosed in supplement.
- [x] "Committed default" reset language absent; clean-rerun description in place.

### Results (§5)
- [x] Governance-first order: (1) ResNet case study → (2) cross-node governance transfer → (3) stress & ungoverned counterfactual → (4) memory diagnostic → (5) autoresearch Linux governance stress.
- [x] All five subsections present.
- [x] Cross-references verified.
- [x] Supplement contains full L40S seed tables, bootstrap details, campaign IDs, env details, OpenML per-seed data.

### Discussion & Limitations (§6)
- [x] Evidence-strength table present and updated for clean 5-seed run.
- [x] Limitations section covers: memory mixed generalisation, node scope, bounded-proposal exhaustion, autoresearch none-arm failure mode, worker dependencies, task-metric noise, metric-validation scope, no-ungoverned-head-to-head, governance overhead, generalisation path.
- [x] All limitation statements match revised claims in the body.
- [ ] **Two-rater Cohen's κ not reported** — human validation still pending. Taxonomy labels noted as control-plane-assigned, not independently validated. *(Acknowledged limitation — acceptable to leave for post-submission.)*

### Conclusion (§7)
- [x] Short (~3 paragraphs).
- [x] Ends on methodological contribution.
- [x] No "sharpest memory-effect result."
- [x] Autoresearch framed as governance-under-failure.
- [x] Memory ordering now says "all five clean ResNet seeds (0.899 < 0.938 < 0.941)" — **fixed this session**.

---

## 3. Abstract

- [x] Governance-first; includes: bounded, auditable, append-only, provenance completeness, failure taxonomy, six ML nodes.
- [x] 1,445 total governed trials.
- [x] No 70× statistic.
- [x] Within 150–250 words.
- [x] Memory framed as node-specific diagnostic, not enabling condition.

---

## 4. Tables and Figures

- [x] Lifecycle figure present.
- [x] Governance metrics table present.
- [x] Node inventory with purpose column present.
- [x] Cross-node governance summary table present.
- [x] Evidence-strength table (§6) present and updated.
- [ ] **Figure 9** (`figures/fig9_resnet_linux_ablation.pdf`): verify it was regenerated from clean 5-seed data. *(Check that figure files match the updated ledger statistics.)*
- [ ] Keep at most five major visual/table elements in main paper — **verify after PDF compile**.
- [ ] Move optional memory/ungoverned tables to supplement if still present in main body — **verify after PDF compile**.

---

## 5. Supplement

- [x] `supplement.tex` exists; contains artifact availability, ledger schema, lifecycle pseudocode, OpenML table, campaign IDs, hardware/model environment, bootstrap details.
- [ ] **`[anonymous artifact URL]` placeholder** — fill in before final submission.
- [ ] **Commit hashes `d506631`/`2754828`** — update to final submission commit.

---

## 6. References

- [x] Zaharia et al. BAIR 2024 compound-AI citation replaced and entry added.
- [H] Verify every URL in bibliography.
- [H] Resolve every arXiv ID (confirm correct paper).
- [H] Verify all author names, titles, years, venues.
- [H] Verify all 2026 blog/GitHub citations.
- [H] Verify SHARP citation and arXiv identifier.
- [H] Remove or replace any citation that cannot be verified.

---

## 7. Page Budget

- [ ] Compile final PDF (run pdflatex / latexmk on `main.tex`).
- [ ] Visually confirm main body is ≤ 9 pages (excluding references).
- [ ] If over budget: trim seed-level memory discussion, shorten related work, move remaining env/campaign details to supplement.

---

## Summary: What Still Needs Doing Before Submission

### Claude can do now
1. [x] **Conclusion stale language** — fixed this session.
2. [ ] **Figure 9 verification** — confirm `fig9_resnet_linux_ablation.pdf` was regenerated from clean 5-seed ledgers (check file mtime vs ledger mtime).
3. [ ] **Supplement placeholders** — fill `[anonymous artifact URL]` and update commit hashes once final.

### Requires PDF compile (quick)
4. [ ] **Page budget** — compile and count pages; currently estimated ~11 pages, target ≤ 9.
5. [ ] **Visual table/figure count** — confirm ≤ 5 major elements in main body.

### Human-only
6. [H] **Full reference audit** — URLs, arXiv IDs, author names, venues, 2026 citations.
7. [H] **Failure-taxonomy inter-rater validation** (Cohen's κ) — optional/post-submission acceptable.
8. [H] **Artifact manifest** — confirm `artifact_manifest.json` indexes final 225 L40S campaigns.
9. [H] **Anonymous submission** — fill placeholder URL before uploading.
