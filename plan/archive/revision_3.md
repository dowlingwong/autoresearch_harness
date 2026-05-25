# Revision 3 — KDD ETAAI 2026 Workshop Paper

**Target paper:** *Evaluating Governed LLM-Driven ML Experimentation: Lifecycle, Provenance, and Failure Metrics*
**Target venue:** KDD 2026 Workshop on Evaluation and Trustworthiness of Agentic AI (ETAAI)
**Goal:** Convert the draft from a dense multi-result experimental report into a clean, reviewer-friendly workshop paper centered on governance/evaluation of autonomous ML experimentation agents.

---

## Legend

- `[x]` Complete or addressed in current working tree.
- `[ ]` Still pending.
- `[~]` In progress / blocked on overnight run.
- `[H]` Human-only task — cannot be delegated to an agent.

---

## Current Status Summary (as of 2026-05-21)

**Done:** All Priority 0 credibility fixes (s4/s5 reset disclosure, trial-count reconciliation, taxonomy qualifier, citation fix, abstract restructure, Fixes 1–4 and A–C).

**Done:** s4/s5 clean rerun (90 trials, all PASS, 0% invalid rate, none < summary ≈ rationale confirmed, 100% provenance completeness).

**In progress:** s1/s2/s3 clean rerun (135 trials, running on server overnight, `scripts/run_resnet_s123_clean_rerun.sh --gpu 1 --no-smoke`).

**Remaining:** Full structural rewrite (Phase B) — reorder results, rewrite intro/conclusion, compress related work, move content to supplement. Then update §5.3 with clean 5-seed data after overnight run. Then rewrite abstract last. Then compile and verify ≤9 pages.

---

## 0. Non-Negotiable Constraints

- [ ] Main paper fits the 9-page workshop limit excluding references. Currently ~11 pages.
- [x] Paper is anonymous (`sigconf,anonymous`).
- [x] ACM conference proceedings format (`acmart`).
- [ ] Main paper is self-contained; extended details moved to supplement.
- [ ] Central claim is governance/auditability, not memory or task-performance improvement.
- [x] 70× result removed from abstract.
- [ ] Detailed seed-specific memory claims removed from introduction.
- [ ] Memory-forward language removed from conclusion.

**Central claim (use as internal guide for all edits):**
> A governed control-plane protocol makes autonomous ML experimentation auditable by enforcing bounded edits, terminal lifecycle states, provenance completeness, artifact evidence, and explicit failure taxonomy.

**One-sentence thesis (use while rewriting):**
> Autonomous ML experimentation agents should be evaluated as governed processes, where every trial has bounded authority, terminal lifecycle state, artifact evidence, provenance, and explicit failure classification — not merely as optimizers judged by best final task score.

---

## 0A. Critical Credibility Fixes

### 0A.1 s4/s5 Reset Mechanism — RESOLVED via clean rerun

- [x] Inspect reset code (`scripts/reset_node_state.py` uses `baseline_ref or "HEAD"`).
- [x] Classify reset method as mutable current HEAD.
- [x] Add reset-semantics disclosure in results and limitations.
- [x] Remove clean-default reset wording that contradicted the disclosure.
- [x] Decision: optional clean rerun — chosen and executed.
- [x] **s4/s5 clean rerun completed** (90 trials, all PASS).
  - 0% invalid rate across all arms. Mean AUC: none=0.900, summary=0.940, rationale=0.941.
  - Expected ordering none < summary ≈ rationale confirmed in 2 clean seeds.
  - 100% provenance completeness.
- [~] **s1/s2/s3 clean rerun in progress** (135 trials, server overnight).
  - Cross-arm contamination confirmed in s1/s2/s3 (different `node_state_hash` per arm within each seed).
- [ ] AFTER overnight run: sync ledgers, update §5.3 table and prose with clean 5-seed stats.
- [ ] AFTER overnight run: remove reset-semantics disclosure; replace with clean-rerun description.
- [ ] AFTER overnight run: update evidence-strength table.

### 0A.2 L40S Accounting Rule

- [x] All 225 L40S trials kept for governance-integrity totals.
- [x] 3-seed performance statistics labeled as 3-seed diagnostic.
- [x] s4/s5 anomalous seeds explained where performance statistics are interpreted.
- [ ] AFTER overnight run: update L40S table denominators to 5/5/5 across all arms.
- [ ] Every L40S paragraph/table note must state whether s4/s5 included or excluded.

### 0A.3 Trial-Count Discrepancy

- [x] Headline total confirmed: 1,445 governed trials.
- [x] 1,135 clarified as primary-campaign subset in ungoverned counterfactual section.
- [x] Abstract uses 1,445.
- [ ] Confirm all tables and prose use the same inclusion rule after rewrite.

### 0A.4 Internal Consistency Pass

- [x] Unqualified "six failure categories" wording removed.
- [x] `degraded_metric` clarified as discarded, not `failed_invalid`.
- [x] No-ungoverned-head-to-head limitation preserved.
- [x] Conclusion must not call autoresearch the "sharpest memory-effect result". Reframe as governance-under-failure.
- [x] Markov-process paragraph kept; MDP/policy-optimization language in discussion kept consistently.
- [ ] Table 4, method, metrics, results, and conclusion must use the same taxonomy names and counts.

### 0A.5 Reference Audit

- [x] Incorrect `compound AI system` citation replaced with `zaharia2024compound`.
- [x] Zaharia et al. BAIR 2024 bibliography entry added.
- [H] Verify every URL.
- [H] Resolve every arXiv ID.
- [H] Verify author names, titles, years, venues.
- [H] Verify all 2026 blog/GitHub citations.
- [H] Verify SHARP citation and arXiv identifier.
- [H] Remove or replace any citation that cannot be verified.

---

## 0B. Experimental Footprint

- [x] No new experiments for Revision 3.
- [x] Evidence set limited to current nodes/campaigns.
- [x] Primary ledgers verified to exist.
- [x] `/ceph` execution-host artifact storage disclosed.
- [ ] Confirm artifact manifest matches the final submitted campaign set.
- [ ] Failure-taxonomy human validation: optional/pending.
- [x] Ungoverned counterfactual: not expanded beyond N=10.
- [x] N=10 counterfactual labeled illustrative.

---

## 1. Target Reviewer Interpretation

After revision, a reviewer should summarize the paper as:

> This paper proposes an evaluation/governance protocol for agentic ML experimentation. Instead of evaluating only final task metrics, it evaluates whether autonomous experimentation is bounded, auditable, reproducible, and failure-aware. The framework separates proposal generation, worker execution, and control-plane authority; records every trial in an append-only ledger; and reports governance metrics such as invalid-action rate, provenance completeness, artifact evidence, repeated-bad rate, and failure taxonomy. The protocol is validated across six heterogeneous ML nodes, including public OpenML tasks and a real LM-training node.

- [ ] This summary is achievable from the current draft.
- [ ] First-page story centers bounded execution, auditability, reproducibility, failure taxonomy.
- [ ] Memory/performance salience is reduced in abstract, intro, and conclusion.

---

## 2. Main Narrative Spine

The paper must follow this sequence:

1. **Problem:** Task metrics alone do not reveal whether the experimentation process was valid, bounded, reproducible, or auditable.
2. **Gap:** Existing benchmarks and experiment trackers evaluate final task success or record outcomes; they do not govern the agent loop itself.
3. **Method:** A governed control plane owns scope enforcement, lifecycle transitions, metric parsing, keep/discard decisions, memory updates, and append-only audit records.
4. **Metrics:** Governance metrics are first-class evaluation outputs: acceptance rate, invalid-action rate, repeated-bad rate, provenance completeness, artifact evidence completeness, and failure taxonomy.
5. **Evidence:** Six ML nodes and stress/failure regimes. Strongest empirical claim: **100% provenance completeness across all reported nodes and failure regimes**.
6. **Diagnostic probe:** Memory ablation tests whether governance metrics expose manager behavior differences. Results are node-dependent diagnostics, not a general memory-method comparison.
7. **Conclusion:** Agentic ML systems should be evaluated not only by what score they achieve, but by whether their experimentation process is bounded, reproducible, and scientifically inspectable.

---

## 3. Claim Discipline

### Allowed Strong Claims

- [x] Control-plane protocol separates manager proposal, worker execution, and governance authority.
- [x] Every trial is forced into one of three terminal states: `kept`, `discarded`, `failed_invalid`.
- [x] Append-only ledger preserves trial-level provenance.
- [x] Provenance completeness is 100% across governed ledgers.
- [x] Stress campaigns show that scope violations and no-op patches are classified before corrupting node state.
- [x] Ungoverned counterfactual shows that runtime failures become invisible without the ledger/pending-trial guard.
- [x] NodeSpec/control-plane contract transfers across heterogeneous nodes.

### Claims To Qualify (must have explicit caveats)

- [x] Memory effects described as diagnostic in L40S results.
- [ ] Autoresearch memory language still needs softening in abstract, intro, results, conclusion.
- [ ] Repeated-bad rate: diagnostic, not proof of memory superiority.
- [ ] 70× variance result: only in L40S subsection, with same-sentence 3-seed caveat.

### Forbidden or Dangerous Claims (remove if present)

- [x] Remove "memory is the enabling condition" or qualify as this node/interface/prompt only.
- [x] Remove "sharpest memory-effect result" from conclusion.
- [x] 70× result removed from abstract.
- [x] No governance-improves-task-performance claims.
- [x] MLAgentBench adapter language scoped as adapter/compatibility.
- [x] Taxonomy labels: control-plane-assigned; human validation pending.

---

## 4. Abstract

### Required structure (rewrite LAST, after body is stable)

1. Problem: task metrics alone do not audit autonomous ML experimentation.
2. Method: governed control plane with bounded scope, lifecycle states, append-only ledger, separated authority.
3. Metrics: acceptance, invalid-action, repeated-bad, provenance completeness, artifact evidence, failure taxonomy.
4. Evidence: six nodes, 1,445 trials, 100% provenance completeness, visible failure modes.
5. Caveat: memory ablation is diagnostic and node-dependent.

### Word budget: 150–250 words.

### Must include: `bounded`, `auditable`, `append-only`, `provenance completeness`, `failure taxonomy`, `six ML nodes`.

### Must not include: 70× CI-width reduction, detailed seed accounting, campaign identifiers, unsupported memory-superiority claims.

**Checklist:**
- [x] Includes required keywords.
- [x] Uses 1,445 total governed trials.
- [x] 70× statistic removed.
- [ ] Rewrite abstract LAST after body is stable.
- [ ] Keep within 150–250 words after final rewrite.
- [ ] Replace memory-enabling language with node-specific diagnostic framing.

---

## 5. Introduction

### Required structure

1. Opening paragraph: evaluation gap (task metrics do not show whether the agent edited allowed files, preserved failed trials, etc.).
2. Second paragraph: governance perspective — autonomous ML experimentation as compound AI system.
3. Third paragraph: the proposed protocol — manager proposes, worker materializes, control plane owns authority.
4. Fourth paragraph: evidence summary — six nodes, 100% provenance completeness across heterogeneous settings.
5. Contributions: exactly four.

### Four contributions

1. **Governance metric suite:** acceptance, invalid-action, repeated-bad, provenance completeness, artifact evidence, failure taxonomy.
2. **Control-plane protocol:** separation of proposal, execution, validation, and keep/discard authority.
3. **Append-only audit schema:** proposal, patch, run log, parsed metric, decision, failure category, provenance IDs, reproducibility hashes.
4. **Multi-node validation:** six nodes plus stress/ungoverned counterfactuals showing process observability and protocol transfer.

### Intro cuts

- Remove detailed seed-specific memory claims.
- Mention memory only once as a diagnostic probe.

**Checklist:**
- [ ] Opening paragraph centers evaluation gap.
- [x] Governance/control-plane perspective introduced.
- [x] Manager/worker/control-plane authority separation introduced.
- [ ] Evidence paragraph is governance-first, not memory-forward.
- [ ] Contributions list has exactly four contributions.
- [ ] Detailed seed-specific L40S memory claims removed from intro.
- [ ] Memory mentioned only as diagnostic probe.

---

## 6. Related Work

### Structure: three subsections only

1. **ML-agent and autonomous experimentation benchmarks** (AIDE, MLAgentBench, MLE-bench, MLGym).
   - Contrast: they evaluate task success; this paper evaluates governance of the experimentation process.
2. **Experiment tracking, AutoML, and NAS** (MLflow, W&B, AutoML/NAS).
   - Contrast: trackers record outcomes after runs; AutoML optimizes within predefined search spaces; neither governs self-directed agent edits with terminal lifecycle records.
3. **Harness engineering and production-agent governance**.
   - Contrast: this paper turns those design principles into measurable governance metrics and an experimental protocol.

### Required contrast sentence

> We do not propose another task benchmark; we propose a governance layer and metric suite that can wrap heterogeneous ML nodes and make the experimentation process itself evaluable.

**Checklist:**
- [ ] Compressed to three contrast-focused subsections.
- [ ] Required contrast sentence included.
- [x] Incorrect compound-AI citation replaced.
- [H] Citation audit pending.

---

## 7. Method Section

### Title: "Governed Control Plane for Agentic ML Experimentation"

### Must keep

- Manager/worker/control-plane separation.
- Replacement principle: managers are swappable; control-plane authority is fixed.
- NodeSpec: editable paths, frozen paths, run command, parser, budget/range constraints.
- Trial lifecycle state machine.
- Pending guard.
- Scope validator.
- No-op guard.
- Append-only ledger.
- Failure taxonomy.

### Must clarify

The control plane, not the LLM manager, owns: validity, state transitions, metric parsing, keep/discard decisions, memory update, ledger append, provenance IDs.

### Figure instruction

Keep one clean lifecycle figure: `manager proposal → control plane validation → worker patch/run → node training → metric parser → control plane decision → append-only ledger/memory`. Terminal states (`kept`, `discarded`, `failed_invalid`) must be visually explicit.

**Checklist:**
- [x] All required components present.
- [ ] Markov paragraph: keep or cut consistently with MDP follow-on.
- [ ] One clean lifecycle figure.
- [ ] One concise lifecycle table.

---

## 8. Governance Metrics Section

### Central table

| Metric | Definition | What it detects |
|---|---|---|
| Acceptance rate | kept / N | valid improving edits |
| Invalid-action rate | failed_invalid / N | unevaluable or invalid trials |
| Repeated-bad rate | repeated-bad proposals / N | redundant failure modes |
| Provenance completeness | records with required provenance IDs / N | independent reproducibility |
| Artifact evidence completeness | records with patch/log or classified pre-run evidence / N | evidence retention |
| Failure taxonomy | counts by category | classified failure modes |

Metrics are reported **alongside** task metrics, not as replacements.
Failure labels are "control-plane-assigned" unless human validation is completed.

**Checklist:**
- [x] Governance metric table exists.
- [x] Metrics framed alongside task metrics.
- [x] Failure labels are control-plane-assigned.
- [ ] Metric definitions match final table/prose after rewrite.

---

## 9. Evaluation Setup

### Node inventory table columns

Node | Task type | Worker | Metric | Direction | Editable scope | **Purpose in evaluation**

### Purpose column values

- `resnet_trigger`: real scientific case study
- `lr_synthetic`: dependency-free protocol validation
- `openml_credit_g`: public tabular transfer
- `openml_bank_marketing`: public tabular transfer (stronger task metric)
- `mlagentbench_vectorization`: external benchmark adapter compatibility
- `autoresearch_linux`: real LM-training stress case

### Move to supplement

Exact hostnames, excessive environment detail, campaign IDs, most exact seed-level configurations.

**Checklist:**
- [x] Six-node inventory exists.
- [x] "Purpose in evaluation" column added.
- [ ] Hostnames/excess environment/campaign IDs moved to supplement.
- [x] `/ceph` artifact archive location disclosed.
- [ ] "Committed default" reset language removed or softened everywhere.

---

## 10. Results Section

### Required order

1. **Scientific-node case study:** ResNet-trigger demonstrates all lifecycle outcomes.
2. **Cross-node governance transfer:** six-node table, 100% provenance completeness.
3. **Stress and ungoverned counterfactual:** scope/no-op/runtime failures become visible.
4. **Memory as diagnostic probe:** repeated-bad and variance effects are node-dependent.
5. **Real LM-training stress case:** autoresearch Linux shows governance integrity under total failure and partial success regimes.

### Ungoverned counterfactual

Disclose its scale: N=10 trials, single node, single arm, single replicate. Use as illustrative evidence only. Keep the broader limitation that a full governed-vs-ungoverned replicated study remains future work.

**Checklist:**
- [ ] Results reordered to governance-first sequence.
- [x] ResNet scientific-node case study present.
- [x] Cross-node governance summary present.
- [x] Stress and ungoverned counterfactual present.
- [x] Memory as diagnostic probe present (still too prominent — needs demotion).
- [x] Autoresearch Linux present (framing needs softening).
- [ ] Autoresearch reframed as governance integrity under failure, not memory proof.

---

## 11. Results: What to Move to Supplement

Move these unless page budget allows:

- Full seed-level L40S memory tables.
- All campaign IDs.
- Detailed bootstrap procedure.
- Full OpenML b20/b30 per-seed tables.
- Exact runtime microbenchmark details.
- Full hardware/software environment.
- Full artifact manifest description.
- Exact hashes, context hashes, JSONL schema fields.

Replace with one sentence: *Extended seed-level tables, artifact manifests, and reproducibility details are provided in the supplement.*

**Checklist:**
- [ ] Full L40S seed tables moved to supplement.
- [ ] Campaign IDs moved to supplement.
- [ ] Full OpenML per-seed tables moved to supplement.
- [ ] Artifact manifest/schema details moved to supplement.

---

## 12. Memory Ablation

### Required framing

> Memory ablation is used as a diagnostic probe: it tests whether governance metrics such as repeated-bad rate and invalid-action rate can expose behavioral differences between manager conditions. The results are interpreted as node-dependent diagnostics rather than as a general memory-method comparison.

### Must say explicitly

- ResNet shows partial memory sensitivity.
- `lr_synthetic` does not reproduce the repeated-bad reduction.
- L40S variance reduction is based on converging seeds only; not confirmed across all five seeds originally (s4/s5 anomaly).
- After clean rerun: update with whether 5-seed ordering is confirmed.
- Rationale-augmented memory is not consistently better.
- Memory format remains a hypothesis, not a general conclusion.

### 70× rule

Only in the L40S memory subsection, with same-sentence caveat:
> Among the three converging ResNet L40S seeds, summary memory reduces CI width by 70×; this estimate should be treated as a node-specific diagnostic rather than a general memory effect.

**Checklist:**
- [x] Memory ablation described as diagnostic in results.
- [x] L40S 70× absent from abstract.
- [x] L40S 70× caveated as 3-seed diagnostic in results.
- [x] s4/s5 reset-state divergence disclosed.
- [~] AFTER clean 5-seed rerun: update or remove s4/s5 anomaly disclosure.
- [ ] Memory-forward claims removed from intro/conclusion.
- [ ] `lr_synthetic` negative transfer clearly stated.
- [ ] Rationale memory not claimed consistently better.
- [ ] All L40S governance totals use all 225 trials.

---

## 13. Autoresearch Linux

### Keep

- None arm: 120/120 runtime_error failures.
- Summary arm: valid improving trials in 3 of 4 seeds.
- Summary seed s2: total worker failure despite memory.
- Both arms: 240/240 provenance completeness.

### Required interpretation

> The autoresearch Linux node shows that the governance contract remains faithful under expensive real-task execution, total worker failure, and mixed success/failure regimes.

### Required caveat

> The none-arm failure is not claimed to be a general property of memory-free managers; it may reflect prompt/interface mismatch on this node.

**Checklist:**
- [x] None arm total failure, summary arm mixed, 240/240 provenance completeness reported.
- [ ] Reframed as real-task governance stress case.
- [x] "Memory is the enabling condition" softened to specific interface/prompt configuration.
- [ ] None-arm failure caveat present.

---

## 14. Discussion and Limitations

### Three evidence levels

1. **Demonstrated:** lifecycle classification, scope enforcement, append-only provenance, complete records, stress-failure visibility.
2. **Case-study finding:** memory affects behavior on some nodes; autoresearch summary memory enables valid proposals under tested setup.
3. **Not claimed:** general memory superiority, task-performance improvement from governance, full benchmark dominance, human-validated taxonomy.

### Required limitations

- Six nodes are still limited; edit surfaces are narrow.
- Memory effects are mixed and node-dependent.
- L40S 70× variance reduction is a 3-seed estimate; update after clean rerun.
- Taxonomy labels are deterministic control-plane labels until human inter-rater validation.
- No full statistically replicated governed-vs-ungoverned performance comparison.
- OpenML and MLAgentBench results are governance-transfer evidence, not competitive benchmark claims.
- Autoresearch none-arm total failure is interface/prompt-specific.

**Checklist:**
- [x] Evidence-strength table present.
- [x] No-head-to-head limitation preserved.
- [x] Taxonomy validation pending preserved.
- [x] OpenML/MLAgentBench adapter limitations present.
- [x] Reset-semantics limitation present.
- [~] AFTER clean rerun: update evidence-strength table.
- [ ] All limitations match revised claims after body rewrite.

---

## 15. Conclusion

### Required

End on methodological contribution:

> The stronger conclusion is methodological: autonomous experimentation agents should be evaluated not only by final task scores, but by governance metrics that determine whether the experimentation process was bounded, reproducible, failure-aware, and scientifically inspectable. The proposed control-plane protocol and audit schema provide one concrete way to make those properties measurable across heterogeneous ML nodes.

**Checklist:**
- [x] Conclusion shortened.
- [ ] Ends on methodological contribution.
- [ ] "Sharpest memory-effect result" removed.
- [ ] Autoresearch reframed as governance-under-failure.

---

## 16. Page Budget

### Target allocation

| Section | Target pages |
|---|---:|
| Abstract | 0.25 |
| Introduction | 1.0 |
| Related Work | 1.0 |
| Method / Control Plane | 2.0 |
| Governance Metrics | 0.75 |
| Evaluation Setup | 1.0 |
| Results | 2.0 |
| Discussion / Limitations / Conclusion | 1.0 |
| **Total** | **9.0** |

### Cut order if over budget

1. Detailed seed-level memory discussion.
2. Long related-work paragraphs.
3. Environment details.
4. Campaign IDs.
5. Redundant OpenML b20/b30 explanation.
6. Detailed autoresearch per-seed prose.
7. Markov-process paragraph **and** any later MDP/policy-optimization reference.

**Checklist:**
- [ ] Compile final PDF.
- [ ] Visually confirm main body ≤ 9 pages (currently ~11 pages).

---

## 17. Tables and Figures

### Main paper: at most five major visual/table elements

- [ ] Figure 1: Governed trial lifecycle.
- [ ] Table 1: Workshop concerns mapped to harness mechanisms (if space allows).
- [ ] Table 2: Six governance metrics.
- [ ] Table 3: Six benchmark nodes with purpose column.
- [ ] Table 4: Cross-node governance summary.

### Move to supplement

- [ ] Full memory ablation seed tables.
- [ ] Full L40S replication details.
- [ ] Full OpenML b20/b30 tables.
- [ ] Full MLAgentBench vectorization details.
- [ ] Artifact manifest / schema details.

---

## 18. Language Tightening Rules

### Replace

| Risky phrase | Safe replacement |
|---|---|
| "proves" | "demonstrates," "shows," "provides evidence" |
| "memory improves" | "memory changes manager behavior under this node/prompt configuration" |
| "benchmark result" (OpenML/MLAgentBench) | "governance-transfer result" |
| "failure labels are validated" | "failure labels are control-plane-assigned; human validation is pending" |
| "memory is the enabling condition" | "summary memory enabled valid proposals in this specific autoresearch interface and prompt configuration" |
| "sharpest memory-effect result" | "sharpest governance-under-failure result" |
| "anomalous seeds" (alone) | "anomalous seeds, attributed to reset-state contamination" (or remove after clean rerun) |

### Preferred keywords

governed control plane, compound AI system, bounded execution, lifecycle state, terminal trial record, append-only ledger, provenance completeness, artifact evidence, failure taxonomy, failure observability, scientific inspectability, manager replaceability.

**Checklist:**
- [ ] "Memory is the enabling condition" replaced.
- [ ] "Sharpest memory-effect result" replaced.
- [x] "Six failure categories" wording replaced.
- [x] Compound-AI citation replaced.
- [ ] Stale clean-seed language for s4/s5 updated or removed after rerun.
- [ ] "proves" replaced throughout.

---

## 19. Final Revision Checklist

### Alignment
- [ ] Paper clearly targets agentic AI evaluation/trustworthiness, not generic AutoML.
- [ ] Abstract and introduction center governance/evaluation, not memory.
- [ ] Workshop fit obvious by end of page 1.

### Claims
- [ ] No uncaveated claim that memory generally improves performance.
- [x] No claim that governance improves final task metrics.
- [x] No full MLAgentBench evaluation claim.
- [x] No human-validated taxonomy claim.
- [x] 70× variance result caveated as 3-seed diagnostic outside abstract.

### Method
- [x] Manager, worker, control plane separated.
- [x] Control-plane authority explicit.
- [x] Lifecycle states defined.
- [x] NodeSpec contract defined.
- [x] Append-only ledger fields summarized.

### Metrics
- [x] Six governance metrics defined.
- [x] Metrics reported alongside task metrics.
- [x] Repeated-bad rate framed diagnostically.
- [x] Failure taxonomy framed as control-plane-assigned.

### Evaluation
- [x] Six nodes described.
- [x] Cross-node governance summary included.
- [x] 100% provenance completeness highlighted.
- [x] Stress campaigns included.
- [x] Ungoverned counterfactual included and labeled illustrative.
- [ ] Autoresearch Linux needs governance-stress framing.

### Limitations
- [x] Mixed memory generalization explicit.
- [x] Narrow edit surfaces acknowledged.
- [x] Limited node count acknowledged.
- [x] No full causal governed-vs-ungoverned comparison acknowledged.
- [x] Human taxonomy validation status clear.

### Page limit
- [ ] Main paper ≤ 9 pages excluding references.
- [ ] Supplement contains extended reproducibility details.
- [ ] Main paper remains self-contained.

### Credibility
- [x] s4/s5 reset mechanism verified; s4/s5 clean rerun done.
- [~] s1/s2/s3 clean rerun in progress.
- [x] 1,445 vs 1,135 discrepancy explained.
- [x] L40S 3-seed vs 5-seed caveats added.
- [x] Failure-taxonomy count/names consistent.
- [ ] Autoresearch still needs memory-result reframing.
- [H] Human reference audit pending.
- [x] "Compound AI system" citation corrected.
- [ ] Final compiled PDF body visually checked ≤ 9 pages.

---

## 20. Acceptance-Oriented Final Check

Before submission, a skeptical ETAAI reviewer should answer "yes" to all:

- [ ] Is this clearly an agentic AI evaluation paper?
- [ ] Does it provide metrics or protocol, not just a system demo?
- [ ] Does it evaluate behavior/process, not only task success?
- [ ] Are failure modes visible and classified?
- [ ] Are governance claims stronger than task-performance claims?
- [ ] Are limitations honest enough to prevent overclaim rejection?
- [ ] Is the paper readable within 9 pages?
- [ ] Does every reference resolve to a real, correctly attributed source?
- [ ] Are all trial counts, denominator choices, and seed exclusions explained?
- [ ] Is the 70× result absent from the abstract and caveated in the results?
- [ ] Is the s4/s5 reset issue either verified-clean (via rerun) or disclosed?
- [ ] Have all reported ledgers/artifacts been verified to exist?
- [ ] Is the ungoverned counterfactual presented as illustrative?
- [ ] If Cohen's kappa is mentioned, was it actually run and reported?

---

## 21. Agent Editing Priority Order

Apply changes in this order:

1. [x] Resolve credibility blockers (P0 fixes done; clean reruns in progress).
2. [ ] Reorganize method around control-plane authority.
3. [ ] Make governance metrics central.
4. [ ] Simplify evaluation setup; add purpose column.
5. [ ] Reorder results governance-first.
6. [ ] Move detailed memory/seed/env/campaign material to supplement.
7. [ ] Strengthen limitations and claim discipline.
8. [ ] Compress related work (3 subsections).
9. [ ] Rewrite introduction and contributions (exactly 4).
10. [ ] Rewrite conclusion (short, methodological).
11. [ ] Rewrite abstract LAST, after body is stable.
12. [ ] Compile PDF and visually verify ≤ 9-page body limit.

**Do not rewrite the abstract first.**

---

## 22. Phase B — Agent Execution Prompt

Copy this prompt verbatim to a new agent to execute the structural rewrite.

```
You are editing a LaTeX workshop paper for KDD 2026 ETAAI (Evaluation and Trustworthiness of Agentic AI).

PAPER DIRECTORY:
  /Users/wongdowling/Documents/autoresearch_harness/A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/

SECTION FILES TO EDIT:
  main.tex                          (abstract, overall structure)
  sections/01_introduction.tex
  sections/02_related_work.tex
  sections/03_system_design.tex     (method section)
  sections/04_experiments.tex       (evaluation setup)
  sections/05_results.tex           (results — reorder required)
  sections/06_discussion_limitations.tex
  (create if needed) sections/supplement.tex

MASTER INSTRUCTIONS FILE (read this first):
  /Users/wongdowling/Documents/autoresearch_harness/plan/revision_3.md

CENTRAL CLAIM (every edit must serve this):
  A governed control-plane protocol makes autonomous ML experimentation auditable
  by enforcing bounded edits, terminal lifecycle states, provenance completeness,
  artifact evidence, and explicit failure taxonomy.

YOUR TASK — apply in this order:

STEP 1 — Conclusion (sections/06_discussion_limitations.tex or wherever conclusion lives):
  - Shorten to ≤2 paragraphs.
  - Remove "sharpest memory-effect result" — replace with "sharpest governance-under-failure result".
  - Final paragraph must end on the methodological contribution (see §15 of revision_3.md for exact wording).
  - Reframe autoresearch Linux as real-task governance stress case, not memory proof.

STEP 2 — Related work (sections/02_related_work.tex):
  - Compress to exactly three subsections: (1) ML-agent benchmarks, (2) experiment tracking/AutoML/NAS, (3) harness engineering/production-agent governance.
  - Each subsection must end with a contrast sentence explaining how this paper differs.
  - Add the required contrast sentence: "We do not propose another task benchmark; we propose a governance layer and metric suite that can wrap heterogeneous ML nodes and make the experimentation process itself evaluable."
  - Cut any prose beyond what is needed to establish the contrast. Target: 1 page.

STEP 3 — Introduction (sections/01_introduction.tex):
  - Required structure (4 paragraphs + contributions):
    1. Opening: evaluation gap — task metrics do not reveal whether the agent edited allowed files, preserved failed trials, made auditable keep/discard decisions.
    2. Governance perspective — autonomous ML experimentation as compound AI system (manager + worker + control plane).
    3. Protocol — manager proposes, worker materializes, control plane owns authority.
    4. Evidence — six nodes, 1,445 trials, 100% provenance completeness.
    5. Contributions: exactly four (see §5 of revision_3.md for the list).
  - Remove any detailed seed-specific L40S memory claims.
  - Mention memory only once as "a diagnostic probe."
  - Do not mention 70×, s4/s5, bootstrap CIs, or seed counts in the introduction.

STEP 4 — Evaluation setup (sections/04_experiments.tex):
  - Add a "Purpose in evaluation" column to the node inventory table with values:
      resnet_trigger: real scientific case study
      lr_synthetic: dependency-free protocol validation
      openml_credit_g: public tabular transfer
      openml_bank_marketing: public tabular transfer (stronger task metric)
      mlagentbench_vectorization: external benchmark adapter compatibility
      autoresearch_linux: real LM-training stress case
  - Move exact hostnames, full software versions, all campaign IDs to supplement.
  - Remove or soften any "committed default" reset language.

STEP 5 — Results (sections/05_results.tex):
  - Reorder the results subsections to this exact sequence:
    §5.1 ResNet scientific-node case study
    §5.2 Cross-node governance transfer (100% provenance completeness)
    §5.3 Stress campaigns and ungoverned counterfactual
    §5.4 Memory as diagnostic probe (L40S replication + lr_synthetic)
    §5.5 OpenML public nodes (keep brief; move per-seed tables to supplement)
    §5.6 Autoresearch Linux: real LM-training stress case
  - DO NOT change the numerical values in §5.4 (memory/L40S section) — those numbers are pending an overnight clean rerun and will be updated separately.
  - In §5.6 (autoresearch Linux): reframe opening sentence as governance stress case, not memory finding. Add caveat that none-arm failure is not a general property of memory-free managers.
  - In the ungoverned counterfactual paragraph: confirm it is labeled as "small illustrative counterfactual (N=10, single node, single arm, single replicate)."
  - Move full seed-level L40S tables and full OpenML per-seed tables to supplement (or mark with \supplementonly{} if the supplement file does not yet exist).

STEP 6 — Method section (sections/03_system_design.tex):
  - Verify the Markov-process paragraph. If it is present, either:
    (a) keep it AND ensure no MDP/policy-optimization follow-on language exists anywhere that depends on it, OR
    (b) remove it AND also remove all later references to "MDP-like treatment" and "policy-optimization methods."
  - Do not restructure the rest of the method section — it is already in good shape.

STEP 7 — Language pass (all files):
  - Replace "memory is the enabling condition" → "summary memory enabled valid proposals in this specific autoresearch interface and prompt configuration."
  - Replace "sharpest memory-effect result" → "sharpest governance-under-failure result." (Already done in Step 1 but check everywhere.)
  - Replace "proves" → "demonstrates" or "shows" or "provides evidence."
  - Replace "anomalous seeds" alone → "anomalous seeds, attributed to reset-state contamination" (or remove if clean-rerun disclosure has been added).
  - Do NOT rewrite the abstract yet.

CONSTRAINTS:
  - Do not change numbers in §5.4 (L40S memory table and prose). Those are pending clean data.
  - Do not rewrite the abstract. It will be rewritten last after this structural pass is reviewed.
  - Do not add new experiments or new content — only restructure, cut, and reframe.
  - Preserve all LaTeX \label{}, \ref{}, \cite{} tags correctly.
  - The paper must remain compilable after your edits.
  - When moving content to supplement, create sections/supplement.tex if it does not exist.

ACCEPTANCE CRITERIA:
  - Introduction has exactly 4 contributions, no seed details, memory mentioned once as diagnostic probe.
  - Related work has exactly 3 subsections with contrast sentences.
  - Results section follows the 6-subsection governance-first order above.
  - Conclusion is ≤2 paragraphs and ends on methodological contribution.
  - "Sharpest memory-effect result" does not appear anywhere.
  - "Memory is the enabling condition" does not appear unqualified.
  - Autoresearch Linux opening sentence frames it as governance stress case.
  - Ungoverned counterfactual is labeled as small/illustrative/N=10.
```

---

## 23. After Phase B — Remaining Steps

1. **Wait for s1/s2/s3 clean rerun** to finish (running overnight).
2. **Sync and update §5.4** with clean 5-seed statistics (update table denominators to 5/5/5, update AUC numbers, update or remove reset-semantics disclosure, update evidence-strength table).
3. **Rewrite abstract** — last, after the body is stable.
4. **Compile PDF** and visually verify main body ≤ 9 pages.
5. **[H] Human reference audit** — click every URL, resolve every arXiv ID, verify author names/venues/years.
