# Revision 3 Checklist

Derived from `plan/revision_3.md`.

Legend:
- `[x]` complete or already addressed in the current working tree.
- `[ ]` still pending.
- `[~]` in progress / blocked on overnight run.

---

## 0. Non-Negotiable Constraints

- [ ] Main paper fits the 9-page workshop limit excluding references.
  - Current compiled PDFs inspected locally are 11 pages. Full structural rewrite required.
- [x] Keep the paper anonymous.
- [x] Use ACM conference proceedings format.
- [ ] Ensure main paper is self-contained, with extended details moved to supplement.
- [ ] Ensure the central claim is governance/auditability, not memory or task-performance improvement.
- [x] Remove the 70x result from the abstract.
- [ ] Remove or demote detailed seed-specific memory claims from the introduction.
- [ ] Remove or soften memory-forward conclusion language.

---

## 0A. Critical Credibility Fixes

### 0A.1 s4/s5 Reset Mechanism — RESOLVED via clean rerun

- [x] Inspect reset code.
- [x] Inspect L40S reset/log evidence.
- [x] Classify reset method (mutable HEAD — confirmed).
- [x] Add reset-semantics disclosure in results and limitations.
- [x] Remove clean-default reset wording that contradicted the disclosure.
- [x] Decision: optional clean rerun — chosen and executed.
- [x] s4/s5 clean rerun completed (90 trials, all PASS).
  - Result: 0% invalid rate, mean AUC none=0.900 / summary=0.940 / rationale=0.941.
  - Expected none < summary ≈ rationale ordering confirmed in 2 clean seeds.
  - 100% provenance completeness across all arms.
- [~] s1/s2/s3 clean rerun running on server (135 trials, ~18h, started overnight).
  - Cross-arm contamination also confirmed in s1/s2/s3 (different hash per arm within each seed).
  - Script: scripts/run_resnet_s123_clean_rerun.sh --gpu 1 --no-smoke
- [ ] AFTER s1/s2/s3 rerun: sync ledgers, update §5.3 table and prose with clean 5-seed stats.
- [ ] AFTER s1/s2/s3 rerun: remove reset-semantics disclosure; replace with clean-rerun description.
- [ ] AFTER s1/s2/s3 rerun: update evidence-strength table (memory ordering confirmed 5 clean seeds, if holds).

### 0A.2 L40S Accounting Rule

- [x] Keep all 225 L40S trials for governance-integrity totals.
- [x] Label 3-seed performance statistics as 3-seed diagnostic.
- [x] Explain s4/s5 anomalous seeds where performance statistics are interpreted.
- [ ] AFTER rerun: update L40S table denominators to 5/5/5 across all arms.
- [ ] Ensure every L40S paragraph/table note states whether s4/s5 included or excluded.

### 0A.3 Trial-Count Discrepancy

- [x] Confirm headline total: 1,445 governed trials.
- [x] Clarify 1,135 as primary-campaign subset in ungoverned counterfactual section.
- [x] Abstract uses 1,445.
- [ ] Reconcile README stale 1,355 count. (Out of scope per user.)
- [ ] Confirm all tables/prose use same inclusion rule after larger rewrite.

### 0A.4 Internal Consistency Pass

- [x] Remove unqualified "six failure categories" wording.
- [x] Clarify degraded_metric as discarded, not failed_invalid.
- [x] Preserve no-ungoverned-head-to-head limitation.
- [ ] Reframe conclusion so autoresearch is not the "sharpest memory-effect result".
- [ ] If Markov-process paragraph is cut, also remove later MDP/policy-optimization language.
- [ ] Align Table 4, method, metrics, results, and conclusion on taxonomy names/counts.

### 0A.5 Reference Audit

- [x] Replace incorrect compound AI system citation with zaharia2024compound.
- [x] Add Zaharia et al. BAIR 2024 bibliography entry.
- [ ] Human: verify every URL.
- [ ] Human: resolve every arXiv ID.
- [ ] Human: verify author names, titles, years, venues.
- [ ] Human: verify all 2026 blog/GitHub citations.
- [ ] Human: verify SHARP citation and arXiv identifier.
- [ ] Remove or replace any citation that cannot be verified.

---

## 0B. Experimental Footprint

- [x] Do not expand experiments for Revision 3.
- [x] Keep evidence set limited to current nodes/campaigns.
- [x] Verify primary ledgers exist.
- [x] Disclose /ceph execution-host artifact storage.
- [ ] Confirm artifact manifest/index matches final submitted campaign set.
- [ ] Failure-taxonomy human validation remains optional/pending.
- [x] Do not expand ungoverned counterfactual beyond N=10.
- [x] Frame N=10 ungoverned counterfactual as illustrative.

---

## 0C. Reset-Mechanism Inspection Protocol — COMPLETE

- [x] Locate reset implementation.
- [x] Classify reset method as mutable HEAD.
- [x] Compare seed-start hashes from ledgers.
- [x] Apply decision table: chose clean rerun path.
- [x] s4/s5 clean rerun: DONE.
- [~] s1/s2/s3 clean rerun: IN PROGRESS.

---

## 1–3. Narrative Spine and Claim Discipline

- [ ] Paper clearly targetable as governance/evaluation protocol by page 1.
- [ ] Results reordered: governance-first, memory-as-diagnostic-probe second.
- [x] Allowed strong claims confirmed (lifecycle, provenance, scope enforcement).
- [ ] Autoresearch memory language softened in abstract, intro, results, conclusion.
- [ ] "Memory is the enabling condition" replaced with node/interface/prompt-specific qualifier.
- [ ] "Sharpest memory-effect result" removed from conclusion.
- [x] 70x result absent from abstract.
- [x] 70x result caveated as 3-seed diagnostic in results section.

---

## 4. Abstract

- [x] Include bounded, auditable, append-only, provenance completeness, failure taxonomy, six ML nodes.
- [x] Use 1,445 total governed trials.
- [x] Remove 70x statistic.
- [x] Rewrite abstract LAST after body is stable.
- [x] Keep abstract within 150-250 words after final rewrite.
- [x] Replace memory-enabling language with node-specific diagnostic framing.

---

## 5. Introduction

- [ ] Opening paragraph centers evaluation gap.
- [x] Governance/control-plane perspective introduced.
- [x] Manager/worker/control-plane authority separation introduced.
- [ ] Evidence paragraph is governance-first, not memory-forward.
- [ ] Contributions list has exactly four contributions.
- [ ] Remove detailed seed-specific L40S memory claims from intro.
- [ ] Mention memory only as diagnostic probe.

---

## 6. Related Work

- [ ] Compress to three contrast-focused subsections.
- [ ] Include required contrast sentence (not another task benchmark; a governance layer).
- [x] Replace incorrect compound-AI citation.
- [ ] Human: citation audit pending.

---

## 7. Method Section

- [x] Manager/worker/control-plane separation present.
- [x] Replacement principle, NodeSpec contract, lifecycle states, pending guard, scope validator,
      no-op guard, append-only ledger, failure taxonomy all present.
- [ ] Remove or keep Markov paragraph intentionally; if removed, remove MDP follow-on.
- [ ] Keep one clean lifecycle figure.
- [ ] Keep one concise lifecycle table.

---

## 8. Governance Metrics Section

- [x] Governance metric table exists.
- [x] Metrics framed alongside task metrics.
- [x] Failure labels are control-plane-assigned.
- [ ] Ensure metric definitions exactly match final table/prose after rewrite.

---

## 9. Evaluation Setup

- [x] Six-node inventory exists.
- [ ] Add "Purpose in evaluation" column in node table.
- [ ] Move hostnames/excess environment/campaign IDs to supplement.
- [x] Disclose /ceph artifact archive location.
- [ ] Remove or soften "committed default" reset language everywhere.

---

## 10–12. Results and Supplement

- [ ] Reorder results to governance-first sequence:
  1. ResNet scientific-node case study.
  2. Cross-node governance transfer (100% provenance completeness).
  3. Stress and ungoverned counterfactual.
  4. Memory as diagnostic probe.
  5. Autoresearch Linux as real-task stress case.
- [x] All five sections present in paper (ordering and framing need work).
- [x] Verify broken cross-references after results reorder.
- [x] Create supplement with OpenML table, campaign IDs, hardware/model environment, and bootstrap details.
- [ ] Reframe autoresearch primarily as governance integrity under failure.
- [ ] Move to supplement: full L40S seed tables, bootstrap detail, campaign IDs, env details,
      OpenML b20/b30 per-seed details, artifact manifest.

---

## 13. Memory Ablation

- [x] Memory ablation described as diagnostic in results.
- [x] L40S 70x result absent from abstract; caveated as 3-seed diagnostic in results.
- [x] s4/s5 reset-state divergence disclosed.
- [~] AFTER clean 5-seed rerun: update or remove s4/s5 anomaly disclosure.
- [ ] Remove memory-forward claims from intro/conclusion.
- [ ] Ensure lr_synthetic negative transfer is clear.
- [ ] Ensure rationale memory is not claimed consistently better.

---

## 14. Autoresearch Linux

- [x] None arm total failure, summary arm mixed, 240/240 provenance completeness reported.
- [ ] Reframe as real-task governance stress case.
- [ ] Soften "memory is the enabling condition" to specific interface/prompt configuration.
- [ ] Confirm none-arm failure caveat present (not general property of memory-free managers).

---

## 15. Discussion and Limitations

- [x] Evidence-strength table, no-head-to-head, taxonomy pending, reset-semantics limitation all present.
- [ ] AFTER clean rerun: update evidence-strength table accordingly.
- [ ] Ensure all limitations match revised claims after body rewrite.

---

## 16. Conclusion

- [ ] Shorten conclusion.
- [ ] End on methodological contribution.
- [ ] Remove "sharpest memory-effect result".
- [ ] Reframe autoresearch as governance-under-failure.

---

## 17. Page Budget

- [ ] Compile final PDF.
- [ ] Visually confirm main body is <= 9 pages (currently ~11).
- [ ] Cut seed-level memory discussion, long related work, env details/campaign IDs if over budget.
- [ ] If cutting Markov paragraph, remove MDP/policy-optimization follow-on.

---

## 18. Tables and Figures

- [x] Lifecycle figure, governance metrics table, node inventory, cross-node summary all exist.
- [ ] Keep at most five major visual/table elements in main paper.
- [ ] Add purpose column to node table.
- [ ] AFTER clean rerun: update L40S table labels and denominators (5/5/5).
- [ ] Move optional memory/ungoverned tables to supplement.

---

## 19. Language Tightening

- [ ] Replace "memory is the enabling condition" with node-specific wording.
- [ ] Replace "sharpest memory-effect result" with governance-under-failure wording.
- [x] Replace "six failure categories" wording.
- [x] Replace compound-AI citation source.
- [ ] Remove stale clean-seed language for s4/s5 or update to "verified clean after rerun".
- [ ] Avoid "proves"; prefer "shows"/"demonstrates"/"provides evidence".

---

## 20. Final Revision Checklist Summary

### Still open (grouped by type)

**Structural rewrite (biggest remaining work):**
- [ ] Reorder results governance-first.
- [ ] Rewrite introduction (4 contributions, governance-first, no seed details).
- [ ] Rewrite conclusion (short, methodological, no memory headline).
- [ ] Compress related work (3 subsections).
- [ ] Move seed tables/campaign IDs/env details to supplement.
- [ ] Add purpose column to node table.
- [ ] Markov paragraph: keep or cut consistently.

**After overnight run completes:**
- [~] Update §5.3 table and prose with clean 5-seed stats.
- [~] Remove reset-semantics disclosure; replace with clean-rerun description.
- [~] Update evidence-strength table.
- [~] Update L40S table denominators (5/5/5).

**Last steps (do after body is stable):**
- [x] Rewrite abstract (do this LAST).
- [ ] Compile PDF and verify <= 9 pages.

**Human-only tasks:**
- [ ] Reference audit: all URLs, arXiv IDs, author names, venues, 2026 citations.
- [ ] Optional: failure-taxonomy inter-rater validation.

---

## Editing Priority Order (from revision_3.md §22)

1. [x] Resolve credibility blockers (P0 fixes done; clean reruns in progress).
2. [ ] Reorganize method around control-plane authority.
3. [ ] Make governance metrics central.
4. [ ] Simplify evaluation setup; add purpose column.
5. [ ] Reorder results governance-first.
6. [ ] Move detailed memory/seed/env/campaign material to supplement.
7. [ ] Strengthen limitations and claim discipline.
8. [ ] Compress related work.
9. [ ] Rewrite introduction and contributions.
10. [x] Rewrite abstract last, after body is stable.
11. [ ] Check page budget and remove redundant details.
12. [ ] Compile PDF and visually verify 9-page body limit.
