# Revision 2 — Submission Review Plan
_Based on external venue review. Deadline: June 1, 2026. ~2 weeks._  
_Last updated: 2026-05-20_

---

## Venue Correction

**Correct workshop name:** KDD Workshop on Evaluation and Trustworthiness of Agentic AI  
**OpenReview identifier:** `KDD.org/2026/Workshop/Agentic_AI_Evaluation_and_Trustworthiness`  
**Previous paper said:** old workshop abbreviation and old workshop title; corrected in source and generated PDFs.  
**External check:** workshop website confirms the long venue name and 9-page body limit; OpenReview portal title confirms the same corrected venue scope. No official short abbreviation is displayed publicly, so `KDD ETAAI'26` remains a submission-time human check.  
**Page limit:** 9 pages body, excluding references (unlimited supplement)  
**Deadline:** June 1, 2026 | Notification: July 1 | Camera-ready: July 10

---

## Priority 1 — Venue Strings (30 min, Claude can do)

- [x] Fix `\acmConference` block in `main.tex`:  
  Change to `\acmConference[KDD ETAAI'26]{KDD Workshop on Evaluation and Trustworthiness of Agentic AI}{August 2026}{Jeju Island, Republic of Korea}`  
  _(Verify exact short abbreviation from OpenReview portal before submitting)_
- [x] Fix `\acmConference` year/DOI/ISBN fields — year remains 2026; DOI/ISBN remain empty; outdated total-page wording removed from documentation
- [x] Search and replace all occurrences of the old venue abbreviation in all `.tex` files
- [x] Fix `sections/01_introduction.tex` — old workshop-title phrasing replaced
- [x] Fix `sections/03_system_design.tex` — §3.1 table caption / framing referencing old name replaced
- [x] Fix `sections/04_experiments.tex` — checked; no old workshop-name occurrence was present
- [x] Grep entire paper dir for the old abbreviation to catch any remaining occurrences: none after regenerating PDFs

---

## Priority 2 — 9-Page Body Verification (1–2 hrs, human + LaTeX)

- [x] Run `latexmk -pdf main.tex` and check which page references begin on: references begin on page 10, so body ends on page 9
- [x] If body exceeds page 9: identify what can move to supplement without breaking self-containment — not needed
- [x] **Supplement candidates (move if needed — unlimited pages):** no moves needed because body is within the 9-page limit
  - [x] Full ledger JSONL schema — stayed in body
  - [x] `recover_pending.py` tool description detail — stayed in body
  - [x] Blind-label taxonomy export details — stayed in body
  - [x] Full per-seed raw tables (autoresearch, OpenML) — stayed in body
  - [x] Reference-verification audit table — not present as a body move candidate
- [x] Ensure paper remains self-contained after any moves — no moves made
- [x] Recheck page count after any trims: `main.pdf` is 11 pages total with references starting on page 10

---

## Priority 2b — Ungoverned Baseline (Level 1 + Level 2) (~6 hrs total)

**Issue:** Every "the harness makes X visible" claim is really "the harness records X." Without a counterfactual, a reviewer writes: "You've shown the instrument produces readings, not that the readings matter."

**Strategy:** Two-layer evidence that triangulates the critique:
- **Level 1** (retroactive, no new experiments): across all 1,355 paper trials, show what each governance guard caught and what would have been invisible without it.
- **Level 2** (minimal new run, ~30 min GPU): governed vs ungoverned side-by-side on autoresearch_linux none arm — the substrate that already produces 100% runtime_error failures. Expected result: governed ledger = 10 records; ungoverned ledger = 0 records; 10 crashes invisible.

**Level 1 — already done:**
- [x] `scripts/analyze_governance_counterfactual.py` written and validated
- [x] Run: `python3 scripts/analyze_governance_counterfactual.py --csv paper/tables/governance_counterfactual.csv`
- [x] Incorporate into paper: new §5.x "Ungoverned Counterfactual" subsection written with Level 1 + Level 2 prose

**Level 2 — COMPLETE:**
- [x] `--ungoverned` flag implemented in `src/autoresearch/control_plane/campaign.py`
  - Pending-trial guard: NOT written/deleted (crash leaves no sentinel)
  - Ledger append on failure: SKIPPED (failed_invalid trials silently dropped)
  - `*_ungoverned_obs.jsonl` observation log written alongside ledger
- [x] `scripts/run_ungoverned_level2.sh` written (governed + ungoverned back-to-back)
- [x] `scripts/compare_governed_ungoverned.py` written (produces comparison CSV + table)
- [x] Synced updated src/ and scripts to server
- [x] Run on server: 10 governed records, 0 ungoverned records, 10 obs entries
- [x] `paper/tables/level2_governed_vs_ungoverned.csv` generated and synced back
- [x] Paper prose written in new §5.x "Ungoverned Counterfactual" subsection

**Key claim this enables:**
"Across 1,355 governed trials, governance surfaced N failures that would otherwise be silent [Level 1 table]. To confirm this is not merely retroactive accounting, we ran a matched 10-trial ungoverned campaign on the autoresearch node. The ungoverned ledger is empty; all 10 crashes left no record [Level 2 table]."

**Timeline:** Level 2 run May 22 (after s4/s5). Prose May 22–23.

---

## Priority 3 — Reference Audit (2–3 hrs, human task)

**Risk:** Amazon/Microsoft PC will not forgive a hallucinated citation. All 22 entries must be human-verified.

- [ ] Open every entry in `references.bib` and confirm: title, authors, venue, year are correct
- [ ] **High-priority entries to check first:**
  - [ ] `burtenshaw2026multiautoresearch` — author field was previously fixed; verify full entry is real
  - [ ] `trivedy2026anatomy` — recent; verify title + venue exist
  - [ ] `trivedy2026betterharness` — recent; verify title + venue exist
  - [ ] `anthropic2026longrunningb` — cited in §3.5; confirm real document + correct URL
  - [ ] Any citation added in the autoresearch / variance-collapse / Markov sessions (May 19–20)
- [ ] Mark each entry Low / Medium / High risk
- [ ] Replace any High-risk entry before submitting

---

## Priority 4 — Variance-Collapse Claim (1 hr, or 2 days GPU)

**Issue:** Abstract leads with "70× CI-width reduction" — based on 3 seeds. Reviewer will flag.

### Option A — Soften (recommended if L40S not available)
- [x] Add "3-seed estimate" qualifier to abstract variance-collapse sentence
- [x] Reframed abstract: governance leads (100% provenance on 225 trials incl. 60 anomalous), variance observation scoped to 3 converging seeds
- [x] §5.3 bold finding includes "(3 converging seeds)" qualifier after AUC ordering numbers
- [x] §6.2 σ comparison has "(3-seed estimate)" qualifier
- [x] Evidence table in §6 already says "3 converging seeds" and "3-seed estimate" ✓

### Option B — Strengthen (only if L40S is free before ~May 27)
- [ ] Run 5-seed × b20 × 3-arm ResNet ablation on deepthought2 (~300 trials, ~1–2 days GPU)
- [ ] Recompute bootstrap CIs with 5 seeds
- [ ] Update Table 3, §5.3 prose, abstract, evidence table
- [ ] Change qualifier to "5 seeds per arm" throughout

_Decision gate: check L40S availability by May 21. If not free by May 23, take Option A._

---

## Priority 5 — Autoresearch Confound Preemption (30 min, Claude can do) ✅ DONE

**Issue:** §5.6 claims "memory is enabling condition" strongly; §6.2 walks it back quietly. Reviewer will feel misled.

- [x] Added sentence to §5.6 "Summary memory is the enabling condition" paragraph: "Whether a different base prompt or explicit interface description could recover the \texttt{none} arm is not tested; the governance finding is the classification fidelity across all 120 trials, not the failure cause."
- [x] Verified §6.2 autoresearch limitations paragraph is present and consistent
- [x] Confirmed the two paragraphs do not contradict each other

---

## Priority 6 — CFP Framing Alignment (30 min, Claude can do) ✅ DONE

The CFP explicitly names your core contributions. §1 now mirrors the language.

- [x] "auditability" added to §1 line 3: "...observability of process determines whether an autonomous experiment has \emph{auditability}: a traceable, reproducible record of every decision and its evidence."
- [x] "standardized metrics and a logging protocol" added to §1 line 7 (protocol definition sentence)
- [x] "compound AI system" added to §1 line 5: "manager and harness together form a compound AI system~\cite{burtenshaw2026multiautoresearch}"
- [x] CFP keyword mapping verified:
  - "Lifecycle and Governance Frameworks" ✓ (control plane, lifecycle states throughout)
  - "auditability, liability attribution" ✓ (now explicit in §1)
  - "Benchmarking, Metrics, and Standardization" ✓ (six governance metrics; now framed as "standardized metrics and a logging protocol")
  - "compound AI systems" ✓ (now in §1)

---

## Priority 7 — Final PDF and Layout Check (1 hr, human)

- [ ] Generate final PDF: `latexmk -pdf main.tex`
- [ ] Read through PDF (not source) — check for broken refs, missing figures, layout issues
- [ ] Verify all figures render correctly (especially ResNet L40S Fig linked in §5.3)
- [ ] Verify running headers show correct workshop name after venue string fixes
- [ ] Verify abstract word count ≤ 150–200 words (currently 192 ✓ — recheck after any edits)
- [ ] Verify author block is anonymous (`\author{Anonymous Author(s)}`)
- [ ] Check no `\todo`, `\fixme`, or draft comments remain in source
- [ ] Verify references section starts no earlier than page 10 (i.e., body ≤ 9 pages)

---

## Do Not Do Before Submission

- ❌ Add new experiments
- ❌ Restructure sections
- ❌ Rewrite abstract from scratch (it's 192 words and correctly scoped)
- ❌ Add new related work entries (risk of hallucination; only add if you can physically verify)
- ❌ Submit to any other KDD 2026 workshop simultaneously (shared reviewer pool → desk-reject at both)

---

## Timeline

| Task | Effort | Owner | Target date |
|---|---|---|---|
| Fix venue strings | 30 min | Claude | May 20 ✅ |
| Verify 9-page limit, plan supplement | 1 hr | Human + LaTeX | May 21 ✅ |
| Decide Option A vs B on variance claim | 15 min | Human | May 21 ✅ → Option B (s4+s5 at b15) |
| Run s4+s5 ResNet on server | ~12 hr GPU | Server | May 20–21 (running) |
| Level 1 governance counterfactual table | 2 hr | Claude ✅ | May 20 ✅ |
| Level 2 ungoverned run on server | ~30 min GPU | Server | May 20 ✅ |
| Sync new src/ + scripts to server | 5 min | Human | May 20 ✅ |
| Write ungoverned baseline prose (§5 or §6) | 30 min | Claude | May 20 ✅ |
| Human reference audit (22 entries) | 2–3 hr | Human | May 22–24 |
| Preempt autoresearch confound in §5.6 | 30 min | Claude | May 20 ✅ |
| CFP framing alignment check | 30 min | Claude | May 20 ✅ |
| Update paper with 5-seed CIs (after s4s5 sync) | 1 hr | Claude | May 20 ✅ |
| Final PDF check (layout, figures, headers) | 1 hr | Human | May 29–30 |
| Submit via OpenReview | — | Human | June 1 |

---

## Acceptance Estimate (per external review)

| Condition | Estimate |
|---|---|
| All 7 priorities completed | ~65–75% |
| Priorities 1–5 only (no CFP framing / layout) | ~60–70% |
| Priorities 1 and 3 skipped | ~40–50% (venue mismatch + citation risk) |

**Venue fit: exceptional.** The CFP explicitly names lifecycle governance, auditability, standardized logging protocols, and compound AI systems — this paper is a near-ideal match.
