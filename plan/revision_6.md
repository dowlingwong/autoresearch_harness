# Revision 6 Checklist: Submission Credibility Cleanup

## Positioning

The current draft is close to submission-ready. Revision 6 should not be a broad rewrite or a new-experiment pass. The goal is to close the remaining reviewer-ammunition gaps: reset disclosure, contribution/evidence alignment, stale anomaly framing, degenerate counterfactual wording, and a few numeric/text consistency checks.

Acceptance posture from the review: approximately 70--80% if the high-leverage fixes below are addressed; approximately 60--70% if they are left unresolved.

## Highest-Leverage Fixes

- [ ] Resolve and disclose the `autoresearch_linux` summary-seed s2 anomaly.
  - Priority: highest.
  - Target location: Section 5.6, with a short limitation note if needed.
  - Problem: Section 5.6 reports that summary seed s2 suffers total worker failure, but the paper does not say whether this was investigated after the ResNet reset-from-HEAD bug was found.
  - Question to answer: did `autoresearch_linux` use the same reset machinery as the ResNet run, a frozen-baseline restore protocol, or the original mutable git-HEAD reset?
  - If frozen/verified: state that the `autoresearch_linux` reset uses the verified baseline-restore protocol and that s2 remains unexplained or node-specific.
  - If git-HEAD-based: disclose that a clean rerun is not included because of cost, and frame s2 as unresolved.
  - If unknown: say so explicitly and avoid strengthening any memory-effect interpretation.
  - Suggested sentence shape: "Unlike the corrected ResNet rerun, the `autoresearch_linux` campaigns [used / did not use / have not yet been verified against] the fixed baseline-restore protocol; therefore the s2 total-failure seed is reported as an unresolved anomaly rather than evidence for a memory effect."

- [ ] Triple-check the Section 5.3 trial-accounting arithmetic.
  - Priority: highest.
  - Target location: Section 5.3 ungoverned-counterfactual paragraph and any related table/caption.
  - Current numbers to reconcile: 302 runtime-error trials, 163 precondition-caught trials, 500 valid-but-degrading trials, all out of 1,135 paper-relevant trials.
  - Check: 302 + 163 + 500 = 965, leaving 170 other trials. Confirm that the remaining count matches the kept/discarded/failed-invalid totals in Table 6 and the campaign ledgers.
  - Requirement: if the numbers do not balance, fix the counts or denominator rather than explaining around the mismatch.

- [ ] Reconcile the Section 1 contribution list with the evidence actually reported.
  - Priority: high.
  - Target location: Section 1 contributions.
  - Problem: contribution 3 lists private, synthetic, OpenML, and MLAgentBench-adapter nodes plus stress campaigns, while contribution 4 separately introduces `autoresearch_linux`; the ResNet L40S replication is not clearly named in the contributions.
  - Fix option A: list all six node families in contribution 3, then let contribution 4 focus narrowly on the `autoresearch_linux` high-failure demonstration.
  - Fix option B: restructure the bullets so every reported node is clearly covered by exactly one contribution.
  - Also rewrite contribution 4 so the contribution is the governance demonstration, not the fact that the no-memory arm failed.
  - Suggested shape: "a demonstration that the governance contract preserves provenance under high-failure regimes, validated on the `autoresearch_linux` LM-training node where 120/120 no-memory trials crashed."

- [ ] Get two-rater Cohen's kappa if feasible.
  - Priority: high, but skip rather than rush low-quality labels.
  - Target locations: Section 4.5 and Section 6.2.
  - Evidence needed: two raters label the 30-record blind-label CSV.
  - If completed: report Cohen's kappa with `n=30`, and remove or soften the "pending human validation" limitation.
  - If not completed: keep the limitation, but make the pending status consistent anywhere taxonomy validity is invoked.

## High-Severity Text Fixes

- [ ] Replace stale "anomalous seed divergence" framing in the abstract and conclusion.
  - Target locations: abstract and conclusion.
  - Problem: Section 5.5 now supersedes the anomalous ResNet seed divergence with a verified clean 5-seed rerun; the old anomaly was a protocol/reset contamination issue, not a node-specific failure mode.
  - Preferred fix: replace "anomalous seed divergence" with "node-state contamination revealed by per-trial reproducibility hashes" or remove the third example entirely.
  - Requirement: do not frame the reset bug as a harness-discovered node-specific failure mode unless the text explicitly says it was experimental-protocol contamination.

- [ ] Reframe "100% provenance completeness" as invariant validation, not empirical discovery.
  - Target locations: abstract, conclusion, and any headline result sentence.
  - Problem: Section 3.4 correctly says the invariant is structural, but the abstract/conclusion still lead with 100% completeness as if it were an ordinary empirical outcome.
  - Suggested abstract wording: "Across 1,445 governed trials, the control-plane invariant -- that every budget slot produces exactly one terminal record -- held without violation, including in high-failure campaigns where 100% of trials crashed."
  - Requirement: preserve the result, but make clear the empirical content is "no implementation violation of the invariant."

- [ ] Reframe Table 8's `autoresearch_linux` ungoverned CI as an extremal/definitional case.
  - Target location: Table 8 caption and Section 5.3 prose.
  - Problem: the ungoverned completeness CI `[0.00, 0.00]` for `autoresearch_linux` is degenerate because all 30 governed trials fail and therefore all silently drop in the ungoverned arm.
  - Fix: call the `autoresearch_linux` row an extremal case or definitional bound.
  - Requirement: make the statistical narrative lean on `openml_bank_marketing`, where IR = 55% and 13/20 drops provide the informative comparison.
  - Also check whether the "combined N=50" caption overstates the statistical evidence by mixing the degenerate and informative rows.

## Medium-Priority Consistency Fixes

- [ ] Add an explicit in-text reference to Figure 1 in Section 3.2.
  - Target location: replacement-principle/control-plane prose.
  - Problem: Figure 1 is useful but slightly orphaned.
  - Suggested sentence shape: "Figure 1 illustrates this replacement principle: managers and workers can vary, while the control plane owns validation, lifecycle state, budget, memory update, provenance IDs, and final decisions."

- [ ] Align the Figure 1 control-plane labels with Section 3.2 prose.
  - Target locations: Figure 1 source/caption and Section 3.2.
  - Problem: the figure lists scope validation, lifecycle state machine, keep/discard decision, provenance ID assignment, and memory context update; the prose lists validity, state transitions, budget, memory updates, provenance IDs, and final decision.
  - Fix: use the same terms in both places, and include budget in the figure if it remains in the prose.

- [ ] Qualify the Section 3.2 replacement-principle backend list.
  - Target location: Section 3.2.
  - Problem: the paper empirically demonstrates `prompt_manager` and `LangGraphManager`, but names baseline heuristic and AIDE-style tree search as swappable backends.
  - Fix option A: trim the list to demonstrated backends.
  - Fix option B: keep the broader architectural claim but add: "We empirically validate two backends (`prompt_manager` and `LangGraphManager`); baseline heuristics and AIDE-style tree search are architectural possibilities under the same contract."

- [ ] Align ResNet memory-ordering language to "none < summary approximately rationale."
  - Target locations: Section 5.5 and conclusion.
  - Problem: `0.938` vs `0.941` is a tiny mean difference relative to the reported sigma, so "summary < rationale" overstates the distinction.
  - Fix: use "none < summary approximately rationale" everywhere.
  - Requirement: preserve the stronger variance story: `sigma_none = 0.009` vs `sigma_summary = sigma_rationale = 0.002`.

- [ ] Decide whether to include future MDP/policy-optimization language.
  - Target location: Section 6.2 generalisation path.
  - Current clean option: leave it out.
  - If kept: add one explicit setup sentence such as "The augmented-state formulation also supports treating manager proposals as a policy over governance state, enabling future policy-optimization extensions."
  - Requirement: do not imply the current paper evaluates RL or policy optimization.

- [ ] Make caveats in the contribution list symmetric.
  - Target location: Section 1 contribution bullets.
  - Problem: contribution 1 has a parenthetical caveat about control-plane-assigned taxonomy labels, while other scoped claims are not caveated in the same place.
  - Fix option A: remove the contribution-1 parenthetical and rely on limitations for caveats.
  - Fix option B: add compact caveats to other contribution bullets, e.g. structural invariant, 5-seed memory result, and interface-specific `autoresearch_linux`.
  - Requirement: avoid making one contribution look uniquely uncertain when the others are also scoped.

## Low-Priority Polish

- [ ] Check Table 3 PDF layout.
  - Target: table header and first row spacing.
  - Problem: "From state | To state | Condition" appears visually tight against the first row.
  - Fix only if the final PDF still shows the issue.

- [ ] Clarify supplement structure around Cohen's kappa.
  - Target locations: Section 4.5, Section 6.2, supplement if applicable.
  - Problem: the exporter/scorer workflow is described, but the supplement path for labels or the missing kappa result is not clear.
  - If kappa is not completed: state exactly what exists, what is pending, and where the exporter output lives.

## Do Not Do In Revision 6

- [x] Do not expand the ungoverned counterfactual further.
  - Table 8 is sufficient; more rows risk new inconsistencies.

- [x] Do not add new experimental campaigns.
  - The draft is stable enough; new results at this point are more likely to introduce accounting drift.

- [x] Do not broadly rewrite the abstract or introduction.
  - Touch them only for contribution alignment, structural-invariant framing, and the stale anomalous-seed phrase.

- [x] Do not overclaim memory effects.
  - Keep memory as a diagnostic governance probe with node-specific behavior, not a general performance claim.

## Stop Criteria

- [ ] The `autoresearch_linux` reset semantics are explicitly disclosed.
- [ ] Section 5.3 counts reconcile exactly with the ledgers and tables.
- [ ] Contributions cover the reported evidence without omitting `autoresearch_linux` or the ResNet L40S replication.
- [ ] Abstract, Section 5.5, and conclusion no longer describe the superseded ResNet anomaly as a current node-specific failure mode.
- [ ] The structural provenance invariant is framed consistently across abstract, system design, results, and conclusion.
- [ ] Table 8 distinguishes the degenerate `autoresearch_linux` row from the informative `openml_bank_marketing` comparison.
- [ ] Final PDF compiles cleanly and table/figure references are consistent.
