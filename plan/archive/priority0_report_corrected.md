# Priority 0 Verification Report (Corrected)

Date: 2026-05-21

Scope: This report verifies the Priority 0 credibility items from `plan/revision_3.md` before any large paper-body rewrite. It inspects code, ledgers, local artifacts, current paper text, and bibliography metadata in the active paper source tree `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/`.

## Summary

Priority 0 is not clean. The paper can still proceed as a governance/auditability workshop paper, but several current claims must be revised before body editing:

- The ResNet L40S reset mechanism uses mutable `HEAD`, not a frozen clean baseline. The 5-seed memory-performance result must be demoted or rerun.
- Trial counts are inconsistent across the manuscript and README.
- Failure taxonomy wording is inconsistent with the code schema and with the text's five-vs-six category framing.
- All primary ledgers exist locally. Patch/log artifacts are stored on the execution server at `/ceph/dwong/autoresearch_harness/` (deepthought2); they are real but not in the local checkout. This is expected and is a disclosure item, not a credibility blocker.
- The ungoverned counterfactual has the right limitation elsewhere, but the results wording should explicitly label it illustrative.
- References require human verification. The current `compound AI system` citation is not the right source; the verified source should be Zaharia et al., BAIR 2024, if the term is kept.

## Priority 0 Items

### 1. s4/s5 Reset Mechanism

Classification: FIX-IN-TEXT

Reset method found: mutable current `HEAD`.

Evidence:

- `scripts/reset_node_state.py` restores editable files from `baseline_ref or "HEAD"`.
- `--baseline-ref` defaults to `AUTORESEARCH_BASELINE_REF`; otherwise it is `HEAD`.
- `scripts/run_linux_resnet_s4s5.sh` calls `reset_node_state.py` without `--baseline-ref`.
- Reset logs found under `logs/overnight_20260517_031239/resets.log` show `restored ... train.py (source: HEAD)` for the L40S smoke and s1-s3 campaigns.
- Git history for `nodes/ResNet_trigger/train.py` includes multiple prior `autoresearch:` trial commits before the current `HEAD`.
- ResNet L40S trial-001 ledgers have differing `node_state_hash` values across seeds and arms. Examples:
  - `deepseek_resnet_none_s1`: `7652acd3...`
  - `deepseek_resnet_none_s2`: `8ec67897...`
  - `deepseek_resnet_none_s3`: `3c9dc6ec...`
  - `deepseek_resnet_none_s4` and `s5`: `7dbd568d...`

Interpretation:

- The reset implementation does not restore from a fixed clean baseline snapshot unless a baseline ref is explicitly supplied.
- For the L40S campaigns inspected here, the available logs and scripts point to mutable `HEAD`.
- Therefore the 5-seed ResNet L40S memory-performance result should be treated as methodologically contaminated or at least reset-state sensitive.

Required paper action:

- Do not keep the 70x variance result in the abstract.
- Do not describe s4/s5 as clean independent seeds.
- Report all 225 L40S trials only for governance integrity.
- If keeping performance/variance statistics, label them as 3-seed diagnostic statistics from s1-s3.
- Add a reset-semantics disclosure: the seed reset restored editable files from branch `HEAD`, not a separately frozen baseline.

Optional action:

- OPTIONAL-RERUN: A clean rerun using `--baseline-ref <fixed-clean-sha>` would improve credibility, but it is not required if the paper demotes the L40S memory result and uses it only diagnostically.

### 2. Trial-Count Consistency: 1,445 vs 1,135 vs 1,355

Classification: BLOCKER

Evidence:

- Current abstract says `1,445 governed trials`.
- Results section says `1,135 paper-relevant trials`.
- README says `1,355 reported governed trials`.
- The current cross-node table row counts total 1,445 trials if LR synthetic is counted as 450 trials:
  - ResNet case study: 20
  - ResNet L40S: 225
  - LR synthetic: 450
  - OpenML credit-g: 230
  - OpenML bank-marketing: 230
  - MLAgentBench vectorization: 50
  - Autoresearch none: 120
  - Autoresearch summary: 120
  - Total: 1,445
- The local LR synthetic ledgers include 600 non-smoke trials if both b10 and b30 ledgers are included, so the current 1,445 total implicitly excludes some LR evidence. That exclusion rule is not stated clearly.
- All non-smoke ledgers in `experiments/ledgers` total 2,294 trials, so no raw total can be used without a declared inclusion rule.

Required paper action:

- Define one accounting rule before editing the abstract.
- Use one headline total consistently, or explicitly say something like: `1,445 total governed trials in the submitted evidence set, of which 1,135 are used for retroactive guard-intervention accounting`.
- Update README or avoid relying on it for submission counts.
- Update tables so row sums match the declared accounting rule.

### 3. Failure-Category Consistency

Classification: FIX-IN-TEXT

Evidence:

- `src/autoresearch/memory/schemas.py` currently defines these `FailureCategory` values:
  - `syntax_error`
  - `runtime_error`
  - `edit_failed`
  - `proposal_precondition_failed`
  - `effective_config_unchanged`
  - `metric_missing`
  - `invalid_edit_scope`
  - `invalid_config`
  - `degraded_metric`
  - `no_op_patch`
- `sections/03_system_design.tex` lists five guard-like failure labels plus valid non-improvements as `discarded`.
- `sections/04_experiments.tex` says "Six failure categories are defined" and includes valid-but-worse edits as one of the six, even though valid-but-worse trials are `discarded`, not `failed_invalid`.
- The current text also omits schema categories such as `invalid_config`, `effective_config_unchanged`, and `proposal_precondition_failed` from the six-category statement.

Required paper action:

- Stop saying "six failure categories" unless the table actually enumerates exactly six and maps them to the schema.
- Prefer wording from the revision plan: "control-plane-assigned labels".
- Separate lifecycle outcomes from failure labels:
  - terminal states: `kept`, `discarded`, `failed_invalid`
  - valid non-improvement: `discarded` with `degraded_metric`
  - invalid failures: schema labels such as `runtime_error`, `invalid_edit_scope`, `no_op_patch`, `proposal_precondition_failed`, `metric_missing`, `edit_failed`, `invalid_config`.
- Use one taxonomy table and the same names in method, metrics, results, and conclusion.

### 4. Reported Ledgers and Artifacts Exist

Classification: FIX-IN-TEXT (downgraded from BLOCKER — see correction note below)

**Correction note:** The original report classified this section as BLOCKER because patch/log artifact paths reference `/ceph/dwong/autoresearch_harness/...` which are absent from the local checkout. This classification was **wrong**. `/ceph` is the dedicated storage mount on the execution server (deepthought2); experiments ran there by design and the artifacts are real. The local source checkout does not and is not expected to contain `/ceph` paths. This is a disclosure item, not a missing-artifact blocker.

Ledger result:

- All primary ledgers checked for the currently reported evidence set exist locally.
- Primary ledger check covered 76 ledger files and found 0 missing ledgers.
- The N=10 governed counterfactual ledger exists.
- The ungoverned observation log exists at `experiments/ledgers/deepseek_autoresearch_linux_none_ung_ungoverned_ungoverned_obs.jsonl` and has 10 lines.

Artifact result:

- Patch/log artifacts are stored at `/ceph/dwong/autoresearch_harness/...` on deepthought2 (the execution server). This is the expected location — all GPU campaigns ran on that server. The paths are real and the files exist on the server.
- The local checkout does not mirror `/ceph` because it is not a locally mounted path. This is expected behavior, not a missing-artifact problem.
- For s4/s5 L40S and autoresearch Linux campaigns, the artifacts are on the server only and have not been synced to the local repo. This is fine for submission purposes as long as the paper discloses the artifact location.

Required paper action:

- Add one sentence in the evaluation setup or supplement: "Patch diffs and run logs are archived on the execution host at `/ceph/dwong/autoresearch_harness/experiments/` and are available on request; ledger records reference these paths via provenance IDs."
- Provenance completeness claims remain valid and are supported by ledger IDs in the local files.
- Artifact evidence completeness claims remain valid: the ledger records carry patch/log references; the files exist on the server. The supplement or a reproducibility note should state that execution-host artifacts are archived separately from the paper submission.
- Do **not** weaken artifact-evidence claims in the paper body on account of the local checkout not containing `/ceph` paths.

### 5. N=10 Ungoverned Counterfactual Framing

Classification: FIX-IN-TEXT

Evidence:

- Current `sections/05_results.tex` reports a matched pair of 10-trial campaigns.
- It states the governed run produced 10 ledger records and the ungoverned ledger is empty, with a 10-line observation log confirming crashes.
- `sections/06_discussion_limitations.tex` preserves the "No ungoverned head-to-head" limitation and says the experiments do not prove a statistically significant governed-vs-ungoverned gap.

Issue:

- The results subsection says "To confirm this is not merely retroactive accounting", which is stronger than the revision plan's "illustrative evidence" framing.
- The section should explicitly say N=10, single node, single arm, single replicate, illustrative only.

Required paper action:

- Keep the counterfactual.
- Reframe it as illustrative failure observability, not a powered comparison.
- Do not delete the no-head-to-head limitation.

### 6. Reference Audit and `compound AI system`

Classification: NEEDS-HUMAN-DECISION

Evidence:

- `sections/01_introduction.tex` currently cites `burtenshaw2026multiautoresearch` for "compound AI system".
- A web check found the canonical source for the term: Zaharia et al., "The Shift from Models to Compound AI Systems", BAIR blog, 2024, https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/.
- The bibliography contains many 2026 blog/GitHub/software references:
  - LangChain blog entries
  - Martin Fowler harness-engineering article
  - Mitchell Hashimoto personal blog
  - OpenAI harness-engineering blog
  - Anthropic long-running harness blogs
  - GitHub repositories for ML Intern, Open Deep Research, Multi-Autoresearch, Hermes Agent
  - 2026 arXiv/preprint entries including SHARP

Required action:

- A human must verify all URLs, author names, titles, dates, venues, and arXiv IDs before submission.
- If `compound AI system` remains, replace the current citation with Zaharia et al. BAIR 2024 or remove the term.
- Do not cite the multi-autoresearch GitHub repository for the term.
- If any 2026 blog/GitHub/arXiv source cannot be verified, remove the citation or weaken the claim.

---

## Concrete Edit Plan

### 1. Claims to Keep Unchanged

- The control plane separates manager proposal, worker execution, and governance authority.
- Every opened governed trial budget slot must terminate as `kept`, `discarded`, or `failed_invalid`.
- The append-only ledger records proposal, decision, failure label, provenance IDs, and reproducibility metadata.
- Provenance completeness is 100% for the reported governed ledgers inspected here.
- Stress/no-op/scope-failure campaigns are useful evidence that governance makes invalid actions visible.
- OpenML and MLAgentBench adapter rows are governance-transfer evidence, not competitive benchmark claims.
- The N=10 ungoverned observation is useful illustrative evidence that failures can become invisible without ledger/pending-guard instrumentation.
- Artifact evidence completeness claims are valid; patch/log files exist on the execution host at `/ceph` and are referenced correctly in ledger provenance IDs.

### 2. Claims to Soften

- L40S memory-performance claims:
  - soften to 3-seed diagnostic observations only;
  - explicitly exclude s4/s5 from performance statistics or include them with clear labels;
  - preserve all 225 trials for governance-integrity totals only.
- Autoresearch Linux memory framing:
  - replace "memory is the enabling condition" with "summary memory enabled valid proposals in this specific interface and prompt configuration."
- ResNet task-metric improvement:
  - keep as secondary evidence that the loop executes, not as optimization proof.
- Failure taxonomy:
  - call labels deterministic control-plane classifications until human inter-rater validation is complete.

### 3. Claims to Delete

- Delete the 70x variance result from the abstract.
- Delete or replace "sharpest memory-effect result" in the conclusion.
- Delete any uncaveated claim that memory generally improves autonomous experimentation.
- Delete any wording implying governance improves final task metrics.
- Delete the current citation of `burtenshaw2026multiautoresearch` as support for "compound AI system".
- If the Markov paragraph is cut for space, also delete the later MDP/policy-optimization sentence.

### 4. Tables and Figures to Update

- Abstract trial count and cross-node summary table:
  - use one declared trial-count accounting rule.
- L40S table:
  - state all 225 trials are included for governance integrity;
  - label performance rows as 3-seed diagnostic statistics when s4/s5 are excluded;
  - add a reset-semantics note.
- Governance metrics/failure taxonomy table:
  - align category names with `FailureCategory` or explicitly define a collapsed paper taxonomy.
- Artifact/provenance table:
  - note that patch/log artifacts are archived on the execution host at `/ceph`; provenance IDs in the ledger are the portable reference.
- Ungoverned counterfactual table:
  - label N=10, single node, single arm, single replicate, illustrative.
- References:
  - add Zaharia et al. BAIR 2024 if "compound AI system" remains.

### 5. Items Requiring Human Decision

- Decide whether to:
  - rerun ResNet L40S s4/s5 or all 5 seeds from a fixed baseline ref; or
  - disclose mutable-HEAD reset semantics and demote the memory-performance result.
- Decide the official headline trial count and the inclusion rule behind it.
- Decide whether to sync `/ceph` artifacts into the submission package or keep them server-only with a disclosure note in the supplement.
- Complete the human reference audit before submission.
- Decide whether to keep the term "compound AI system" with the BAIR citation or use safer wording: "agent system with model, harness, memory, and execution layer."
