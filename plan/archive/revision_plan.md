# Revision Plan: "Evaluating Governed LLM-Driven ML Experimentation"

**Purpose of this document.** This is a constructive, actionable revision plan addressing the reviewer-identified weaknesses in the current draft. It is written as a handoff to an implementing agent (or co-author). Each section states (a) the weakness, (b) the concrete experiment or rewrite required, (c) the deliverable, and (d) the acceptance criterion for "done."

The plan is organized by priority. Sections P0 are blocking — without them the paper does not meet bar. P1 substantially strengthen the contribution. P2 are polish items.

---

## P0-1. Add a head-to-head comparison: governed vs. ungoverned

### Weakness
The paper claims governance metrics "expose behaviours that task score hides," but never compares a governed run to an ungoverned baseline on the same node with the same manager. Every result is descriptive single-system reporting. There is no evidence that governance changes what we can see.

### Required experiment
Run a controlled A/B on `resnet_trigger` and on at least one OpenML node:

- **Arm G (governed):** current harness, full lifecycle, scope guards, append-only ledger.
- **Arm U (ungoverned):** same manager backend, same model, same prompt, same budget, but with the control plane removed — manager directly commits trial state, self-evaluates kept/discarded, no scope whitelist, no pending-trial guard, no append-only ledger (or write-through-overwrite logging only).

Run both arms with **N ≥ 5 independent seeds** per arm per node. Use the same LLM and temperature in both.

### Deliverable
A new table in §5 with columns: arm, kept count, claimed kept count vs. verifier-checked kept count (this is the headline metric), invalid-action rate, scope violations actually committed (Arm U) vs. caught pre-run (Arm G), reproducibility check (re-run a "kept" trial from each arm's records and report whether the metric matches).

### Acceptance criterion
At least one governance metric (most likely the discrepancy between self-claimed and verifier-checked kept trials, or the number of silent scope violations) shows a statistically significant gap between G and U with bootstrap 95% CIs that do not overlap. If the gap is not significant, the paper's central claim is empirically unsupported and the framing must change.

### Current-submission fix
Do not imply this A/B has already been run. The active draft has been revised to
frame the current result as lifecycle instrumentation and auditability evidence:
the governed harness records/rejects invalid, no-op, degraded, and kept trials.
It no longer claims a causal governed-vs.-ungoverned performance gap. The
explicit limitation now says that a true head-to-head requires matched governed
and ungoverned arms, replicated seeds, verifier-checked kept trials, and
bootstrap CIs.

### Estimated effort
Medium. Most of the work is implementing the "ungoverned" mode as a stripped-down adapter around the same manager. Compute cost is roughly 2× the current ablation budget.

---

## P0-2. Run larger budgets with bootstrap confidence intervals

### Weakness
Every governance metric is reported from 10–20 trials with no CIs, no significance tests, no power analysis. RBR moving from 0.78 to 0.44 on a budget of 10 is a 3-trial difference — well inside sampling noise for a stochastic LLM. The baseline AUC seed interval [0.775, 0.798] contains the "best" case-study AUC of 0.7827.

### Required experiment
Re-run the canonical campaigns at larger budget and with more seeds:

- ResNet-trigger case study: budget 50, 5 seeds (currently 20, 1 seed).
- Memory ablation (all three arms): budget 30, 5 seeds per arm per node (currently 10–20, 1 seed per arm).
- OpenML campaigns: budget 50, 3 seeds per node (currently 20, 1 seed).

For every reported governance metric, compute bootstrap 95% CIs (10,000 resamples) over trials within a seed and over seeds. For ordering claims (e.g., RBR(none) > RBR(summary)), report the bootstrap probability that the ordering holds.

### Deliverable
Every table cell that currently shows a point estimate must show "estimate [low, high]". Replace categorical claims ("reduces RBR") with probabilistic ones ("bootstrap P(RBR_summary < RBR_none) = 0.83").

### Acceptance criterion
No headline claim in the abstract or §5 rests on a non-overlapping CI gap of less than 0.1 in proportion-space, unless the underlying N is large enough that the gap is significant at p < 0.05.

### Current-submission fix
A ledger-only CI tool has been added:

```bash
uv run python3 scripts/bootstrap_governance_cis.py \
  --campaign kdd_resnet_scientific_20 --node resnet_trigger \
  --campaign openml_credit_g_main_20 --node openml_credit_g \
  --campaign openml_bank_marketing_main_20 --node openml_bank_marketing \
  --samples 10000 \
  --out A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/tables/governance_bootstrap_cis.csv
```

This is a **trial-level bootstrap**, useful as a diagnostic but not a substitute
for seed-level reruns. The active draft has been narrowed so headline claims do
not rest on short-run point estimates. Once budget-50 / multi-seed reruns are
available, rerun the same script and replace trial-level CIs with seed-level CIs
or hierarchical bootstrap intervals.

### Estimated effort
High in compute, low in code. Re-running campaigns at 5× current budgets and 5× seeds is roughly 25× compute. If compute is constrained, prioritize the head-to-head from P0-1 and the memory ablation; OpenML can use 3 seeds.

---

## P0-3. Use a frontier-class manager to control for model-capability confound

### Weakness
All LangGraph results use `qwen2.5-coder:7b` at temperature 0.7, locally deployed (implied Q4 quantization). This is a small, quantized, code-specialized model. Many "repeated-bad" failure modes may be artifacts of this specific model rather than properties of the harness. The paper does not address this.

### Required experiment
Repeat the memory ablation on ResNet-trigger (3 arms, ≥3 seeds) with at least two manager backends spanning capability:

- **Tier S (small, current):** qwen2.5-coder:7b.
- **Tier L (large):** DeepSeek-V4-Flash via the official DeepSeek API, model id `deepseek/deepseek-v4-flash`.

Optional third tier: a deterministic non-LLM baseline (random valid edit from the allowed range) as a true floor.

### Deliverable
A new table reporting governance metrics × manager tier × memory mode. Discuss whether the RBR ordering survives the model change. If it does, the harness claim strengthens. If it does not, the paper must say so explicitly and reframe.

### Acceptance criterion
The paper reports governance behavior for at least one frontier model and explicitly states whether each finding survives the model substitution. No claim about "manager behavior" should be made on the basis of qwen2.5-coder:7b alone.

### Current implementation note
Yes, the practical fix is to use paid API access unless you have local frontier
hardware. For this project we will use **DeepSeek-V4-Flash** because it is
available through an OpenAI-compatible API, is cheaper than most closed frontier
models, and is positioned by DeepSeek as a fast/economical agent-capable V4
model. Official DeepSeek docs list:

- OpenAI-compatible base URL: `https://api.deepseek.com`
- API key environment: `DEEPSEEK_API_KEY`
- model: `deepseek-v4-flash`
- optional model endpoint: `GET https://api.deepseek.com/models`
- optional thinking controls: `DEEPSEEK_THINKING=enabled|disabled` and
  `DEEPSEEK_REASONING_EFFORT=high|max`. DeepSeek defaults thinking to enabled;
  in thinking mode, `temperature` is accepted for compatibility but has no
  effect. For the first smoke test, use `DEEPSEEK_THINKING=disabled` to make
  strict JSON proposal formatting easier to debug.

The repo now supports provider-prefixed LangGraph models through LangChain. Set
the key and pass the model id:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_THINKING=disabled
uv run python3 scripts/run_kdd_memory_ablation.py \
  --node resnet_trigger \
  --memory-mode append_only_summary \
  --campaign-id deepseek_v4_flash_resnet_summary_seed1 \
  --node-root nodes/ResNet_trigger \
  --budget 10 \
  --manager langgraph_manager \
  --model deepseek/deepseek-v4-flash \
  --worker-model qwen2.5-coder:7b \
  --temperature 0.2
```

Use `--worker-model qwen2.5-coder:7b` so only the **manager** changes. The
worker path remains local/Ollama-compatible and therefore avoids mixing a
manager-tier substitution with a worker-tier substitution.

The same manager model can be used on the active public/synthetic runners:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_THINKING=disabled

# OpenML public tabular nodes; LocalWorker executes locally.
uv run python3 scripts/run_openml_tabular_campaign.py \
  --node all \
  --budget 20 \
  --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash \
  --temperature 0.2

# lr_synthetic memory ablation; LocalWorker executes locally.
uv run python3 scripts/run_lr_synthetic_lg_ablation.py \
  --budget 10 \
  --model deepseek/deepseek-v4-flash \
  --temperature 0.2

# Generic ResNet campaign; DeepSeek manager, local Qwen worker.
uv run python3 scripts/run_campaign.py \
  --node resnet_trigger \
  --campaign-id deepseek_v4_flash_resnet_generic \
  --budget 5 \
  --manager langgraph_manager \
  --memory-mode append_only_summary \
  --node-root nodes/ResNet_trigger \
  --model deepseek/deepseek-v4-flash \
  --worker-model qwen2.5-coder:7b \
  --host http://localhost:11434
```

One-trial smoke test before paying for full runs:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_THINKING=disabled
uv run python3 scripts/run_kdd_memory_ablation.py \
  --node resnet_trigger \
  --memory-mode append_only_summary \
  --campaign-id deepseek_v4_flash_smoke \
  --node-root nodes/ResNet_trigger \
  --budget 1 \
  --manager langgraph_manager \
  --model deepseek/deepseek-v4-flash \
  --worker-model qwen2.5-coder:7b \
  --temperature 0.2 \
  --no-export
```

Before committing to many runs, execute a one-trial smoke test for each
provider/model pair. If a provider's latest recommended model changes, update
the model id from the official provider model list and cite the snapshot used.

### Estimated effort
Low engineering, moderate API cost. The manager interface is already abstracted; swapping the backend is a config change. Budget perhaps 200–500 API calls per ablation arm.

---

## P0-4. Validate the governance metrics as measurement instruments

### Weakness
The paper introduces six governance metrics but never validates them as metrics. We do not know their test-retest reliability, whether they are discriminative across systems, or whether 100% provenance completeness (which never varies) means anything operational.

### Required analyses

1. **Test-retest reliability.** Re-run the same campaign with the same seed twice and report metric stability. Then re-run with different seeds and report seed-induced variance. The ratio is a reliability estimate.

2. **Discrimination.** Does any governance metric separate good managers from bad managers? Plot each metric across the three manager tiers from P0-3. A metric that does not vary across tiers may be uninformative.

3. **Validity probe for provenance completeness.** Sample 5 "kept" trials per campaign and actually re-execute them from the audit record. Report the fraction that reproduce the logged metric within tolerance. If 100% provenance completeness does not predict ≥90% reproduction, the metric is not measuring what it claims.

4. **Failure taxonomy inter-rater reliability.** Have one author and one external person independently classify 30 failed trials into the six categories. Report Cohen's κ. If κ < 0.6, the taxonomy categories are not well-defined.

### Deliverable
A new subsection §4.5 "Metric Validation" with results from the four analyses above.

### Acceptance criterion
At least three of the six governance metrics show: (a) test-retest reliability with CIs tighter than seed-induced variance, (b) variance across manager tiers, and (c) for provenance completeness specifically, empirical reproduction of kept trials. Metrics that fail validation must either be dropped or reported with an explicit caveat.

### Estimated effort
Medium. The reproduction probe is the most labor-intensive — it requires re-executing committed trials from audit records.

---

## P0-5. Audit and fix the reference list

### Weakness
The bibliography contains references dated 2026 (the submission year), arXiv IDs in suspicious formats (e.g., `2604.18752` for SHARP, which corresponds to a publication month that may not yet exist as of submission), and a large number of blog post citations with "Retrieved May 2026" notes. At least some of these may be hallucinated or misattributed. Given Dow's stated preference for verifiable citations, this is a credibility risk that could sink the paper independently of the science.

### Required action

1. For every reference, verify: (a) the work exists, (b) the title and authors are correct, (c) the venue and date are correct, (d) the arXiv ID or DOI resolves.
2. For blog post references (Böckeler, Trivedy, Hashimoto, Lopopolo, Rajasekaran, Young), verify the URL resolves and the dated content is actually there. Replace with peer-reviewed citations where one exists.
3. For SHARP [2]: re-verify the arXiv ID. If the ID is invalid or post-dates submission, find the correct ID or replace the reference.
4. Anything that cannot be verified must be removed or replaced.

### Deliverable
A reference-verification table (can live in supplementary): citation key, verified Y/N, source URL, verification date.

### Acceptance criterion
Every citation in the final bibliography has a verifiable source. No fabricated or misattributed references remain.

### Estimated effort
Low, mechanical, urgent. Should be done by a human, not an LLM, given the failure mode is LLM hallucination.

---

## P1-1. Sharpen the core contribution and trim the memory ablation

### Weakness
The paper's central conceptual move — separation of authority, append-only audit, scope whitelist — is standard software engineering reapplied to LLM agents. The novelty rests on combining these *and* on the failure taxonomy, but the taxonomy is buried and the memory ablation (which is a negative result reframed as a diagnostic) dominates the results section.

### Required rewrite

1. **Promote the failure taxonomy to a first-class contribution.** Move it from §4.4 (one paragraph) to its own subsection in §3. Discuss: how the categories were derived, whether they are exhaustive, what extension procedure a practitioner uses for a new node, the inter-rater reliability result from P0-4.

2. **Demote the memory ablation.** Move it to an appendix or a single paragraph in §5. It does not support the claim it was designed to support; keeping it as a major result invites reviewers to read the whole paper through its failure.

3. **Reframe the contribution claim.** The genuinely new piece is: a *failure taxonomy and lifecycle state machine for LLM-proposed code edits in scientific ML*, together with the empirical demonstration that these can be enforced externally without manager cooperation. This is more defensible than the current "governance metrics as first-class evaluation" framing, which over-promises.

### Deliverable
Rewritten §1 (Introduction), §3 (Governed Control Plane), and §5 (Results) with the taxonomy promoted, the memory ablation demoted, and the contribution claim tightened.

### Acceptance criterion
A reviewer reading only the abstract and §1 can state the single new thing this paper contributes in one sentence, and that sentence is supported by §5.

### Estimated effort
Medium. Mostly writing, but requires real rethinking of the framing.

---

## P1-2. Wrap an existing benchmark node to support the portability claim

### Weakness
The paper claims the governance contract transfers across nodes but tests it on four very lightweight nodes (one private scientific node, one synthetic logistic regression, two OpenML config-only tabular nodes). The OpenML nodes only permit `config.yaml` edits, which is a trivial edit surface. The portability claim is asserted, not demonstrated, on realistic agent code modification.

### Required experiment
Adapt one node from an existing ML-agent benchmark — MLAgentBench, MLE-bench, FML-bench, or MLGym — as a NodeSpec. Choose a task that requires non-trivial code edits (not just hyperparameter tuning). Run a 30-trial governed campaign and report the same governance metrics.

This is also the natural place to add a holdout node (a node never used during harness development) to address the optimize/holdout split issue called out in Limitations.

### Deliverable
A fifth node in §5 with full governance metrics. Honest discussion of whether the harness needed code changes to support the new node — if it did, "the control plane is fixed" is overstated and must be qualified.

### Acceptance criterion
At least one node from an external benchmark is wrapped with no changes to the control plane, and produces full lifecycle classification with complete provenance.

### Estimated effort
High. Adapting an external benchmark node is often more work than it appears. Budget two weeks.

---

## P1-3. Formalize the lifecycle state machine

### Weakness
§3 names a state machine (pending → executing → kept/discarded/failed_invalid) but does not formalize it. There is no transition table, no invariants, no exhaustiveness argument. For a paper whose contribution is the control plane, this is thin.

### Required addition

1. Formal definition: states S, transition function δ, terminal states T ⊆ S, guard conditions per transition.
2. Invariants the control plane enforces — for example: "every opened budget slot has exactly one terminal record" or "no kept trial exists without a patch hash and a parsed metric."
3. Exhaustiveness argument: under what conditions can the lifecycle hang or produce no record? The pending-guard mechanism is described but its sufficiency is not argued.

### Deliverable
A new subsection §3.4 "Lifecycle Formalization" with a transition table and an invariants list. Can be supported by an appendix proof sketch.

### Acceptance criterion
A reader can reconstruct the state machine from the paper alone without reference to the code.

### Estimated effort
Low, mostly writing. The state machine clearly already exists in code.

---

## P1-4. Account for governance overhead (cost)

### Weakness
Governance is not free — it adds compute, wall-clock, LLM token cost, and engineering overhead. The paper never reports these. A practitioner cannot decide whether to adopt the framework without knowing what it costs.

### Required analysis
Per governed trial, report: wall-clock overhead vs. ungoverned baseline (from P0-1), additional LLM tokens consumed by the manager interface, storage overhead from the append-only ledger and patch artifacts, engineering cost (LoC for the control plane).

### Deliverable
A new table in §6 (Discussion) with cost figures.

### Acceptance criterion
A reader can estimate whether governance is affordable for their use case.

### Estimated effort
Low. Most of this is instrumenting the existing runs.

---

## P1-5. Strengthen the related work positioning empirically

### Weakness
The paper distinguishes itself from AIDE, MLAgentBench, MLE-bench, FML-bench, and MLGym, but never empirically compares. The AIDE distinction in particular is a positioning claim ("AIDE couples search, evaluation, and summarization; we separate them") that could be tested by wrapping AIDE's manager in the harness.

### Required action
At minimum: wrap one external system (AIDE manager, MLAgentBench agent, or similar) as a manager backend, run it on `resnet_trigger`, and report governance metrics. This is the same experiment as P0-1 Arm U with a different "ungoverned" instantiation, and may share infrastructure.

### Deliverable
A paragraph in §5 reporting governance metrics for at least one external manager, plus a corresponding line in the head-to-head table.

### Acceptance criterion
At least one external system is empirically compared, not just contrasted in prose.

### Estimated effort
High if AIDE itself is wrapped, medium for a simpler external manager.

---

## P2-1. Internal consistency pass

The abstract and §5 currently lead with "improves validation AUC from 0.9220 to 0.9341" while Limitations admits AUC improvement should not be interpreted as meaningful optimization. Reconcile: either drop the AUC number from the abstract, or qualify it with the baseline seed interval inline.

Similar pass for: "100% provenance completeness" (currently presented as a strength, but a metric that never varies cannot discriminate — say what completeness rules out, not just that it is 100%), and "governance metrics expose behaviours that task score hides" (only true if P0-1 demonstrates the gap).

### Estimated effort
Half a day.

---

## P2-2. Tone and framing

The current draft alternates between strong claims in the abstract and concessive limitations in §6. Pick a register and hold it. The most credible version of this paper is a careful, narrow contribution paper: "we formalize a lifecycle and taxonomy for LLM-proposed ML code edits, demonstrate it on four nodes, and validate the metrics as measurement instruments." Avoid framing this as a general "governance evaluation protocol" until the evidence supports that scope.

### Estimated effort
Half a day.

---

## Suggested order of execution

If working serially, the dependency order is:

1. **P0-5 (reference audit)** — start immediately, no compute needed, blocks submission.
2. **P0-1 (head-to-head)** — produces the central new evidence; everything else builds on it.
3. **P0-3 (frontier manager)** — runs in parallel with P0-1 once the manager interface is set up.
4. **P0-2 (budgets and CIs)** — re-run campaigns with proper statistics, ideally combining with P0-1 and P0-3 runs.
5. **P0-4 (metric validation)** — uses data from P0-1 through P0-3.
6. **P1-1 (rewrite)** — start drafting once P0 results are in.
7. **P1-2, P1-3, P1-4, P1-5** — parallelizable with the rewrite.
8. **P2** — final pass before submission.

---

## Honest assessment of submission target

If P0 items are completed substantively, this becomes a credible workshop paper at KDD AAE or a similar venue. To reach the main KDD track or a top-tier ML venue (NeurIPS, ICML), the paper would additionally need P1-1, P1-2, and at least one finding that is empirically surprising — i.e., a result a reader would not have predicted. The current draft has no such result; the closest candidate is whatever P0-1 reveals about the gap between self-claimed and verifier-checked kept trials in the ungoverned arm.

If P0-1 shows no significant gap, the paper's framing must shift away from "governance reveals what task score hides" toward something more modest, such as "lifecycle formalization and failure taxonomy for LLM-proposed ML edits." That is still a publishable contribution at the workshop level, but it is a different paper.
