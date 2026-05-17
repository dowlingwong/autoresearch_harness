# Paper Revision Checklist

Track progress on fixing the reviewer-identified weaknesses. Check off items as they are completed. Each item links back to a specific weakness in the review.

---

## P0 — Blocking Issues (must fix before submission)

### Recommended execution order

Do **not** start with the large deferred reruns. First make the code, paper
claims, reference list, and experiment environment clean; then rerun campaigns
once the measurement pipeline is stable. Correct order:

1. **Low-compute blockers first:** finish P0-5 reference audit, paper claim
   guardrails, and P1/P2 prose fixes that do not depend on new data.
2. **Instrumentation before reruns:** finish provider support, CI/export tools,
   metric-validation scripts, reproduction-check scripts, and cleanup/reset
   commands.
3. **Smoke tests:** run one-trial smoke tests for DeepSeek manager + local Qwen
   worker and for each LocalWorker node. Fix parser/runtime problems here.
4. **Clean environment:** reset nodes, remove stale ledgers/events/artifacts,
   clear pending guards, rotate any exposed API keys, and record exact command
   versions.
5. **Targeted reruns:** run P0-3 DeepSeek manager substitution first because it
   addresses the model-confound directly and also feeds P0-4 discrimination.
6. **Metric validation:** run P0-4 reliability/reproduction/inter-rater checks
   using the cleaned/new ledgers.
7. **Large-scale reruns last:** run P0-2 budget-50 / multi-seed campaigns only
   after the above is stable; otherwise compute may be wasted on stale code or
   unclear claims.
8. **Final paper integration:** update tables, CIs, limitations, and compile
   only after the accepted evidence set is frozen.

### ☑ P0-1. Head-to-head: governed vs. ungoverned
Current-submission fix: **mitigated by reframing, not by claiming the unrun A/B.**
- [x] Removed/softened unsupported causal wording such as "governance metrics expose what task score hides" where it implied an ungoverned comparison.
- [x] Added an explicit limitation: the current experiments show what the governed control plane records/rejects, but do not prove a statistically significant governed-vs.-ungoverned gap.
- [x] Kept the full A/B design below as the next experiment, rather than presenting it as completed evidence.
- [ ] DEFERRED: Implement an "ungoverned" arm: same manager, same model, no control plane (manager self-evaluates, no scope guard, no append-only ledger)
- [ ] DEFERRED: Run Arm G (governed) vs. Arm U (ungoverned) on `resnet_trigger`, ≥5 seeds per arm
- [ ] DEFERRED: Run the same A/B on at least one OpenML node
- [ ] DEFERRED: For each arm, record claimed kept count vs. verifier-checked kept count
- [ ] DEFERRED: Count silent scope violations in Arm U vs. pre-run rejections in Arm G
- [ ] DEFERRED: Re-execute one "kept" trial from each arm's records and check whether the metric reproduces
- [ ] DEFERRED: Report bootstrap 95% CIs on every metric
- [ ] DEFERRED: Add the head-to-head table to §5
- [x] If no significant gap exists in current evidence, reframe the paper's central claim. Current draft is now framed as lifecycle instrumentation/auditability evidence, not causal agent-performance improvement.

### ☑ P0-2. Larger budgets and bootstrap CIs
Current-submission fix: **CI tooling added and headline claims narrowed; larger reruns remain deferred.**
- [ ] DEFERRED: Re-run ResNet-trigger case study at budget 50, 5 seeds
- [ ] DEFERRED: Re-run memory ablation at budget 30, 5 seeds per arm per node
- [ ] DEFERRED: Re-run OpenML campaigns at budget 50, 3 seeds per node
- [x] Implement bootstrap CI computation for governance metrics: `scripts/bootstrap_governance_cis.py`
- [x] Exported current trial-level CI table: `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/tables/governance_bootstrap_cis.csv`
- [ ] DEFERRED: For ordering claims, compute bootstrap P(ordering holds) after replicated seeds are available
- [ ] DEFERRED: Update every result table cell to show "estimate [low, high]" after seed-level reruns
- [x] Removed or rephrased headline claims that rested on short-run point estimates without seed-level CIs.

### ☐ P0-3. Frontier-class manager (model confound)
- [x] Pick a frontier/large API model for the next run: **DeepSeek-V4-Flash** via model id `deepseek/deepseek-v4-flash`.
- [x] Code path prepared: `LangGraphManager` now accepts provider-prefixed LangChain backends such as `openai/<model>`, `anthropic/<model>`, or `deepseek/<model>` when API keys are set.
- [x] DeepSeek API interface added: set `DEEPSEEK_API_KEY`, optionally `DEEPSEEK_BASE_URL=https://api.deepseek.com`, and run with `--model deepseek/deepseek-v4-flash`.
- [x] DeepSeek manager can now be used across the active campaign runners: `run_kdd_memory_ablation.py`, `run_kdd_main_campaign.py`, `run_campaign.py`, `run_openml_tabular_campaign.py`, `run_lr_synthetic_lg_ablation.py`, and `run_manager_comparison.py`.
- [ ] Justify the choice in the paper
- [ ] Re-run memory ablation on ResNet-trigger with the frontier model, ≥3 seeds per arm
- [ ] Optional: add a random-valid-edit baseline as a true floor
- [ ] Add a table of governance metrics × manager tier × memory mode
- [ ] State explicitly whether each finding survives the model substitution
- [ ] Remove any claim about "manager behavior" that is supported only by qwen2.5-coder:7b

### ☐ P0-4. Validate governance metrics as measurement instruments
- [x] Implement metric-validation summary exporter: `scripts/validate_governance_metrics.py`
- [x] Test-retest reliability smoke probe: ran `metric_validation_vectorization_same_seed_a` and `metric_validation_vectorization_same_seed_b`, 6 trials each; decision agreement = 1.00
- [ ] Run with different seeds, compare to same-seed runs (gives reliability ratio)
- [ ] Discrimination: plot each metric across the three manager tiers from P0-3
- [x] Implement kept-trial re-execution tool: `scripts/reexecute_kept_trials.py`
- [ ] Validity probe: re-execute 5 "kept" trials per campaign from audit records (current smoke: 2 kept MLAgentBench-adapter trials)
- [x] Report the current re-execution smoke fraction: 2/2 MLAgentBench-adapter kept trials reproduce within the noisy microbenchmark tolerance
- [x] Implement blind failure-taxonomy sample exporter: `scripts/export_failure_taxonomy_labels.py`
- [ ] Inter-rater reliability: have one author + one external person classify 30 failed trials into the six categories (CSV exported at `plan/metric_validation/failure_taxonomy_sample.csv`)
- [x] Implement Cohen's κ scorer: `scripts/score_failure_taxonomy_labels.py`
- [ ] Compute Cohen's κ for the taxonomy after two completed rater sheets exist
- [x] Add new subsection §4.5 "Metric Validation"
- [ ] Drop or caveat any metric that fails validation

### ☑ P0-5. Reference audit — COMPLETE (2026-05-16)
- [x] Verify every reference: work exists, title and authors correct, venue and date correct — all 22 cited entries audited, all Low risk
- [x] Verify every arXiv ID / DOI resolves — all arXiv IDs resolve; 3 optional DOI additions remain (non-blocking)
- [x] Specifically check SHARP [2] arXiv ID `2604.18752` — **VERIFIED**: paper exists, correct title "A Scientific Human-Agent Reproduction Pipeline", correct 6 authors (Birk, Kasieczka, Mishra-Sharma, Nachman, Noll, Wamorkar), submitted April 2026
- [x] Verify every blog post URL resolves and content matches the cited claim — all 8 blog/online entries verified
- [x] Replace blog posts with peer-reviewed citations where possible — assessed; blog posts cite practitioner content with no peer-reviewed equivalent; keeping as-is is appropriate
- [x] Build a verification table (citation key, verified Y/N, URL, verification date) — complete master table in `plan/reference_check.md`
- [x] Remove or replace any unverifiable reference — 4 orphan uncited entries removed; all cited entries verified
- [x] Confirm no 2026-dated references actually post-date submission — all checked; SHARP (April 2026) and FMLbench (Oct 2025) verified against arXiv
- [x] Fix `burtenshaw2026multiautoresearch` author from raw handle `{burtenshaw}` → `Burtenshaw, Ben` (verified via HuggingFace profile)

**Remaining optional (non-blocking):** add DOI to `huang2024mlagentbench`, `elsken2019nas`, `deng2023mind2web`

---

## P1 — Substantial Strengthening (needed for credibility)

### ☐ P1-1. Sharpen the contribution and demote the memory ablation
- [ ] Move failure taxonomy from §4.4 paragraph into its own §3 subsection
- [ ] Add: how categories were derived, exhaustiveness argument, extension procedure for new nodes
- [ ] Include inter-rater reliability result (from P0-4) in the taxonomy section
- [ ] Move memory ablation to an appendix or a single §5 paragraph
- [ ] Rewrite §1 to state the contribution as: lifecycle + taxonomy + external enforcement
- [ ] Confirm a reviewer can state the contribution in one sentence after reading abstract + §1

### ☑ P1-2. Wrap an external benchmark node
- [x] Pick a non-trivial node from MLAgentBench, MLE-bench, FML-bench, or MLGym: MLAgentBench `vectorization`
- [x] Adapt as a NodeSpec (frozen assets, run command, metric parser, editable scope): `mlagentbench_vectorization`
- [x] Run a 30-trial governed campaign with full governance metrics: `mlagentbench_vectorization_main_30`
- [x] Document whether the harness required code changes — adapter/registry/generic runner added; control-plane lifecycle logic was unchanged
- [x] Consider designating this as a holdout node (not used for harness development): treated as an external compatibility probe, not a holdout performance claim
- [x] Add the new node's results to §5

### ☐ P1-3. Formalize the lifecycle state machine
- [ ] Define states S, transition function δ, terminal states T, guard conditions
- [ ] List invariants the control plane enforces
- [ ] Argue exhaustiveness: under what conditions can the lifecycle hang?
- [ ] Add transition table and invariants list to §3.4
- [ ] Optional: appendix with proof sketch
- [ ] Confirm a reader can reconstruct the state machine from the paper alone

### ☐ P1-4. Report governance overhead (cost)
- [ ] Measure wall-clock overhead per trial vs. ungoverned baseline
- [ ] Measure additional LLM tokens consumed by the manager interface
- [ ] Measure storage overhead (ledger + patch artifacts)
- [ ] Report engineering cost (LoC for the control plane)
- [ ] Add a cost table to §6 Discussion

### ☐ P1-5. Empirical related-work comparison
- [ ] Wrap at least one external system (AIDE manager, MLAgentBench agent, or similar) as a manager backend
- [ ] Run it on `resnet_trigger` under the governed harness
- [ ] Report governance metrics for the external manager
- [ ] Add a paragraph in §5 and a row in the head-to-head table
- [ ] Replace AIDE positioning prose with empirical comparison

---

## P2 — Polish (final pass before submission)

### ☐ P2-1. Internal consistency
- [ ] Reconcile abstract AUC claim with Limitations caveat — either drop the number or qualify inline with seed CI
- [ ] Rewrite "100% provenance completeness" framing — say what it rules out, not just that it is 100%
- [ ] Confirm "governance metrics expose what task score hides" appears only where P0-1 supports it
- [ ] Check every quantitative claim in the abstract appears with CIs in §5

### ☐ P2-2. Tone and framing
- [ ] Pick a register — careful narrow contribution OR ambitious general protocol — and hold it throughout
- [ ] Remove "general governance evaluation protocol" language unless evidence supports the scope
- [ ] Ensure §6 Limitations does not directly contradict the abstract
- [ ] Pass the "would a hostile reviewer find a contradiction" test

---

## Minor Issues Noted in Review

### ☐ Smaller items
- [ ] Move the 2/3 replicate result out of the abstract
- [ ] Remove or rework Table 1 (currently reads as marketing)
- [ ] Clarify "auditable" vs. "audited" — the paper does not perform an actual audit
- [ ] State explicitly that the failure taxonomy was derived from observed failures on these four nodes, not from first principles
- [ ] Add a sentence on what proposal diversity / time-to-first-improvement / cost-per-kept-trial would add as future metrics
- [ ] Check: does "artifact evidence completeness" overlap with "provenance completeness"? If yes, merge them

---

## Submission Readiness Gate

Before submitting, confirm all of the following:

- [ ] All P0 items checked
- [ ] All P1 items checked (or explicitly deferred with a note in Limitations)
- [ ] All P2 items checked
- [ ] No quantitative claim in the abstract lacks a CI in §5
- [ ] No reference is unverified
- [ ] The contribution claim in §1 matches what §5 demonstrates
- [ ] At least one empirically surprising result is present (something a reader could not have predicted)

If the last item is not met, the paper is a workshop submission, not a main-track submission. Pick the venue accordingly.

---

## Quick Status Dashboard

| Section | Items | Done | Remaining |
|---------|-------|------|-----------|
| P0      | 5     |   3  |     2     |
| P1      | 5     |   1  |     4     |
| P2      | 2     |   0  |     2     |
| Minor   | 6     |   0  |     6     |
| **Total** | **18** | **4** | **14** |

_Last updated: 2026-05-16. P0-5 reference audit completed._

Update this table as items are checked off.
