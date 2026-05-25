# Revision 5 Checklist: Final Credibility and Page-Budget Use

## Positioning

The current paper is meaningfully stronger than the previous draft. The main credibility risks from Revision 3 have largely been addressed: the ResNet L40S s4/s5 reset issue is disclosed and superseded by a verified clean 5-seed rerun; the compound-AI-system citation is now canonical; the abstract no longer carries the 70x claim; the conclusion and limitations are aligned; trial counts and failure-taxonomy wording are reconciled; and the paper now compiles to 8 pages total.

The remaining goal is not a broad rewrite. Use the available page budget selectively to close reviewer objections that still affect methodological credibility. Ignore concerns about 2026-dated blog/GitHub citations for this revision pass.

## Highest-Leverage Additions

- [ ] Expand the ungoverned counterfactual if there is enough time to run new evidence.
  - Priority: highest.
  - Target location: Section 5.3 / stress and ungoverned counterfactual subsection.
  - Current weakness: the counterfactual is N=10, single node, single arm, single replicate, and must remain illustrative.
  - Preferred upgrade: run a matched governed-vs-ungoverned comparison with at least N=30 on `autoresearch_linux`, or N=30 on `autoresearch_linux` plus one additional node if feasible.
  - Analysis requirement: report bootstrap CIs and matched comparison framing.
  - Text requirement if completed: replace "small illustrative counterfactual" with a more precise matched-comparison description.
  - Text requirement if not completed: keep the illustrative framing and do not overclaim.

- [ ] Report human taxonomy agreement if a second rater can label the blind-label CSV.
  - Priority: high.
  - Target locations: taxonomy/validation subsection in Section 4 and limitations in Section 6.
  - Evidence needed: two-rater labels on the 30-record blind-label CSV.
  - Metric: Cohen's kappa, with n=30.
  - Text requirement if completed: add one sentence such as "Cohen's kappa = X.XX (n=30, two raters), indicating that the taxonomy is independently interpretable."
  - Text requirement if not completed: keep the current limitation that two-rater Cohen's kappa is not reported.

- [ ] Verify `autoresearch_linux` reset semantics and disclose the result.
  - Priority: high.
  - Target locations: Section 5.6 and Section 6 limitations.
  - Question to answer: did the `autoresearch_linux` campaigns use a fixed clean reset, mutable branch `HEAD`, or an unclear/mixed reset mechanism?
  - If fixed clean reset: add one sentence confirming reset integrity.
  - If mutable `HEAD`: disclose the risk and treat the summary-seed s2 anomaly cautiously.
  - If unclear/mixed: mark as unresolved in limitations and do not strengthen the memory interpretation.
  - Required caution: do not imply that the `autoresearch_linux` memory effect is general; keep it framed as governance under high failure rate.

## Medium-Priority Text Fixes

- [x] Add early trial-accounting guidance at the start of Section 5.
  - Target location: opening paragraph of results.
  - Purpose: prevent reviewer confusion when 1,445 appears in the abstract and 1,135 appears later.
  - Suggested sentence: "All counts derive from ledgers in `experiments/ledgers/`; the primary paper-relevant campaign set comprises 1,135 of the 1,445 total governed trials, with the remainder being development runs, smoke tests, and campaigns outside the primary identifier set."

- [x] Verify Table 6 L40S acceptance rate against the clean ledgers.
  - Target item: `resnet_trigger` L40S row, 225 trials, AR = 46.9%, IR = 0.4%.
  - Check completed: corrected AR from 46.7% to 46.9% based on 105 kept / 224 valid.
  - Text change: Table 6 only; surrounding governance claims unchanged.

- [x] Sharpen the ResNet L40S figure caption.
  - Target location: Figure 2 / L40S memory ablation figure caption in Section 5.5.
  - Suggested caption: "ResNet L40S memory ablation across 5 clean seeds (verified reset). Left: per-seed acceptance (green) and invalid (red) rates. Right: per-seed best AUC, showing sigma_none = 0.009 vs sigma_summary = sigma_rationale = 0.002 (4--5x reduction)."
  - Requirement: keep the 5-clean-seed framing and do not reintroduce seed-specific anomaly language as a current result.

- [x] Clarify the `edit_failed` relationship to the canonical taxonomy.
  - Target location: Section 3.5 taxonomy description.
  - Current issue: Section 3.5 lists five canonical guard categories, while Section 4.4 lists `edit_failed` among observed categories.
  - Suggested sentence: "When the ClawWorker free-form path is used and the coding agent fails to produce a valid patch, an additional `edit_failed` label is recorded; this is a worker-path artifact rather than a control-plane guard, and does not appear in the deterministic-patch LangGraph ablations."
  - Requirement: preserve the distinction between terminal lifecycle states and failure labels.

## Optional Additions If Space Remains

- [ ] Add a compact lifecycle invariant statement or proof sketch.
  - Target location: Section 3.4 state machine / lifecycle guard discussion.
  - Purpose: make the terminal-record guarantee more formal.
  - Suggested content: define states `{pending, executing, kept, discarded, failed_invalid}` and state the invariant that every opened budget slot reaches exactly one terminal ledger record.
  - Keep it short: half page maximum.
  - Do not add formalism if it crowds out empirical credibility fixes.

- [ ] Strengthen the AIDE / MLAgentBench related-work contrast if space remains.
  - Target location: Section 2.
  - Additions to consider:
    - AIDE lacks append-only ledger, failure taxonomy, and terminal-state guarantee.
    - MLAgentBench evaluates agents on tasks; this paper evaluates the harness around the agent and can wrap benchmark tasks via NodeSpec.
  - Constraint: keep each bold related-work block concise; do not undo the page-budget compression unless the final PDF remains within 9 pages.

- [ ] Decide whether to reuse "compound AI system" framing.
  - Target locations: Section 1, Section 3, conclusion.
  - Current issue: the term appears only once and may read like a one-off keyword.
  - Options:
    - Use it once more when introducing the manager-control-plane architecture.
    - Or remove/soften the phrase if it does not serve the paper's argument.
  - Requirement: keep Zaharia et al. 2024 as the canonical citation if the phrase remains.

## Reference-Note Consistency Checks

- [ ] Check whether the supplement actually contains the table described as "Table S1" for the Guides x Sensors taxonomy.
  - If the table exists: ensure the numbering/reference note is correct.
  - If it does not exist: remove or revise the reference note that says Bockeler's taxonomy is instantiated in Table S1 of this paper.

- [ ] Check whether the note for the Young/Anthropic reference says it supports a "failure-mode motivation table in Section 3."
  - If Section 3 has no such table: revise the note or align it with the actual table/prose.
  - Do not rename tables unless that improves clarity without disrupting the main structure.

## Items To Leave Alone

- [x] Do not worry about 2026-dated blog/GitHub citations in this pass.
  - Rationale: user explicitly said to ignore this issue because the reviewer LLM was trained in 2025.

- [x] Do not reintroduce the old 70x result.
  - Current status: removed from abstract and replaced by the clean 5-seed 4--5x variance statement in the body.

- [x] Do not make the autoresearch result memory-forward.
  - Current framing should remain "governance under high failure rates."

- [x] Do not perform a broad prose rewrite.
  - Only targeted additions/fixes should be made.

## Stop Criteria

- [ ] PDF compiles without fatal errors.
- [ ] Body remains within the 9-page KDD workshop limit.
- [ ] No new numeric claim is added without a ledger/notebook/script source.
- [ ] No new performance claim weakens the governance-first framing.
- [ ] Final text still ends on methodological contribution: governance metrics as evaluation criteria for agentic ML experimentation.
