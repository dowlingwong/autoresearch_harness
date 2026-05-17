# 6. Discussion and Limitations

## Demonstrated Properties

The current artifact suite demonstrates that the control-plane protocol can run a bounded fixed-budget real-worker campaign, classify keep/discard/failed-invalid outcomes, preserve append-only provenance, recover per-trial artifact references, and expose governance metrics as paper tables and figures.

Specifically, the primary real-worker campaign demonstrates that:

- a `claw_style_worker` backed by `prompt_manager` can execute five real ResNet-trigger training runs under governance;
- two valid improving edits are kept with complete patch diffs, parsed metrics, and provenance chains;
- three runtime-invalid trials are classified, recorded, and exposed as first-class audit objects without corrupting node state or leaving a pending guard;
- provenance is complete for all five records (100% rate);
- the append-only ledger correctly reflects the true sequence of events.

The stress campaigns demonstrate that scope violations (`invalid_edit_scope`) and no-op patches (`no_op_patch`) are recorded as first-class audit objects with artifact references and unchanged git state.

## Scope and Non-Claims

The following claims are explicitly not made:

- **General autonomous scientist capability.** The system governs a fixed-scope hyperparameter search loop on one node. It is not a general scientific discovery engine.
- **Scientific discovery.** The net AUC gain of +0.000933 over five trials on the ResNet-trigger task is secondary evidence of real execution, not a claim of meaningful ML improvement.
- **Universal optimisation algorithm.** The harness controls the experimentation process; the proposal selector is a structured round-robin, not an optimisation method.
- **Backend-specific dependency.** The governance protocol does not depend on a specific coding agent, LLM, or worker implementation. Manager and worker are replaceable without changing the control-plane contract.
- **Broad benchmark coverage.** All real evidence uses the ResNet-trigger node. Results may not generalise to other ML tasks without additional evaluation.
- **Flagship-campaign lifecycle diversity.** The primary five-trial campaign includes `kept` and `failed_invalid` outcomes but no `discarded` valid-but-worse trial. Discarded trials are demonstrated in the extended campaigns, not in the flagship run.
- **Memory effect on repeated-bad rate.** The pre-stated strong ordering (none > summary > rationale) is not confirmed in full. In the v2 LangGraph ablation (`temperature=0.7`, `budget=10`, explicit avoidance prompt), the `append_only_summary` arm reduces repeated-bad rate to 0.44 vs. 0.78 for the no-memory baseline — confirming a partial memory effect. However, the `append_only_summary_with_rationale` arm matches the no-memory arm (0.78), yielding a non-monotonic result: none = rationale > summary. We do not claim that rationale memory reduces repeated-bad rate relative to no memory.

## Memory Ablation Analysis

The pre-stated hypothesis — repeated_bad_rate(none) > repeated_bad_rate(summary) > repeated_bad_rate(rationale) — is partially confirmed in the v2 LangGraph ablation but not in full. The finding is layered across three configurations.

**`prompt_manager` arm (design-constrained negative).** All three arms produce identical outcomes (2 kept, 3 failed-invalid, repeated_bad_rate = 0.40, best val_auc = 0.774711). The mechanism is in Section 4: the deterministic round-robin makes memory structurally irrelevant. Memory injection and repeated-bad detection work correctly; the limitation is the proposal generator.

**10-trial `prompt_manager` extension (partial positive).** Extending the same structured manager to 10 trials changes the conclusion. The no-memory arm records repeated_bad_rate = 0.60, while both memory arms record repeated_bad_rate = 0.40 and reach a higher best validation AUC. This supports the weaker claim that memory can reduce repeated poor proposals under a longer budget. It does not support the stronger ordering that rationale beats summary memory, because the two memory arms are tied.

**`langgraph_manager` v2 arm (stochastic LLM, `temperature=0.7`, `budget=10`, explicit avoidance prompt).** All 30 trials routed through the deterministic constant-patch path — zero `edit_failed`. The v2 configuration adds three changes versus v1: doubled budget (10 vs. 5), higher temperature (0.7 vs. 0.2), and an explicit AVOIDANCE RULE in the prompt (fires only when memory is present, instructing the LLM not to re-try parameter changes in the same direction that have already failed).

The repeated-bad rate result is non-monotonic: `append_only_summary` = 0.44, while `none` = `with_rationale` = 0.78. The `append_only_summary` arm is the only one to achieve a meaningful reduction in repeated-bad rate, and it is also the only arm to explore TRAIN\_FRACTION — the parameter that produced the run's largest AUC gains (best: 0.8212 vs. 0.774711 for both other arms).

The failure of the `with_rationale` arm to improve over no-memory is the most striking result. A plausible interpretation is that verbose per-trial rationale gives the LLM material to reconstruct justifications for already-tried parameter directions, partially counteracting the avoidance instruction. The interaction between rationale verbosity and avoidance effectiveness is an empirically testable hypothesis: it predicts that truncating or abstracting the rationale field would restore the ordering.

A pre-bridge LangGraph run produced 15/15 `edit_failed` outcomes because `LangGraphManager` did not populate the structured edit fields. The control plane correctly refused all baseline metrics as keeps, so the run remains failure-taxonomy evidence independent of the memory ablation.

**Summary.** The governance infrastructure — memory injection, SHA-256 verification, repeated-bad counting, fail-safe classification — functions correctly in all configurations. The pre-stated strict ordering is not confirmed: a partial memory effect is present (`summary` reduces repeated-bad rate), but `rationale` memory does not improve on no-memory. The non-monotonic result is a genuine empirical finding that warrants further study of how rationale verbosity interacts with avoidance prompting.

## Why Governance Metrics Matter

Final AUC alone cannot answer whether an autonomous experiment was scientifically credible. An agent could improve a metric while repeatedly attempting invalid edits, hiding failed runs, corrupting state, or keeping decisions that cannot be audited. Governance metrics make those behaviors visible.

Acceptance rate, invalid rate, repeated-bad rate, provenance completeness, artifact capture completeness, and failure categories describe the reliability of the experimentation process around the model. This distinction is the main contribution: the harness does not replace task metrics; it makes task metrics interpretable by attaching lifecycle, validity, and provenance context to every trial.

## Limitations

**Single node.** All real evidence uses the ResNet-trigger node. We cannot rule out that governance behavior reported here is specific to narrow hyperparameter editing on this node. Applying the framework across multiple nodes with holdout splits would strengthen the generalisation claim.

**No holdout evaluation node.** Better-Harness-style holdout evaluation nodes are not yet implemented. Without holdout nodes, it is possible for the proposal selector to overfit to the evaluation surface. This is future work.

**No discarded trial in main campaign.** The current five-trial campaign does not include a valid-but-worse trial. Full three-way lifecycle diversity (kept, discarded, failed-invalid) requires a longer campaign or a proposal that produces a valid metric below the current best.

**Memory ablation: partial positive, non-monotonic.** The `langgraph_manager` v2 ablation (`temperature=0.7`, `budget=10`, explicit avoidance prompt) shows a partial memory effect: `append_only_summary` reduces repeated-bad rate from 0.78 to 0.44 and achieves best val\_auc = 0.8212 by exploring TRAIN\_FRACTION. However, `append_only_summary_with_rationale` does not improve over the no-memory baseline (repeated-bad rate = 0.78, best val\_auc = 0.774711). This non-monotonic result — where mid-complexity memory outperforms richer memory — indicates that the beneficial effect of memory can be attenuated by verbose rationale context. A stronger test would: (a) evaluate rationale truncation vs. full rationale to test the verbosity hypothesis; (b) extend the metric set to include parameter-class entropy; (c) use a task with a wider editable surface so all arms have more parameters to explore.

**Stress artifact reproducibility.** Stress trial artifact files are generated deterministically by the stress workers and are byte-reproducible from the worker source code. The corresponding ledger records are unchanged and continue to serve as the authoritative audit objects.

**Task-metric noise.** Five baseline seeds produce validation AUCs from 0.771511 to 0.810711, with mean 0.784755 and bootstrap 95% CI [0.774809, 0.798329]. This variation is larger than the primary five-trial campaign's +0.000933 gain, so task-metric improvement should be treated as secondary evidence that the loop runs, not as an optimisation claim.

**Reproducibility and worker dependencies.** The `claw_style_worker` has two execution paths. The deterministic patch path — gated on `proposal.extra["deterministic_patch"]` and `proposal.extra["structured_edit"]` — applies constant-value edits directly and requires no external coding agent. The free-form path translates LLM proposals into patches by invoking a live AI coding agent (Claude Code or a claw-harness agent). The `edit_failed` failure mode (Section 5) arises when the free-form path is taken without an agent present: training runs on the unmodified source, and the control plane correctly refuses to keep the result.

The `LangGraphManager` deterministic patch bridge (`_extract_structured_edit`, implemented in the harness) eliminates this dependency for proposals that include structured `param`/`old_value`/`new_value` fields. Any replication attempt using `LangGraphManager` with a model that produces structured proposals will route through the deterministic path without requiring a coding agent.

For fully dependency-free replication — no Ollama, no coding agent — the `LocalWorker` (`src/autoresearch/worker/local_worker.py`) parses "Change PARAM from X to Y" directives from any proposal text and applies them directly. This worker exercises the full control-plane lifecycle, including scope validation, metric parsing, and ledger recording, without external tool dependencies. Reviewers wishing to verify the governance protocol can run a three-trial dry campaign with `LocalWorker` or `DryRunWorker` using only standard Python dependencies.

## Generalisation Path

The NodeSpec YAML pattern generalises the harness to new ML experiments without code changes; each spec is a harness template for a class of experiments. A stronger follow-up evaluation would:

- add multiple node specs, including a holdout node not used for development;
- rerun the memory ablation using `LangGraphManager` with a live coding agent present to close the edit loop, so that free-form proposals are actually applied and the behavioral effect of memory on valid-trial outcomes can be assessed;
- compare multiple manager backends (baseline, prompt, LangGraph) under equivalent budgets and worker modes;
- report bootstrap confidence intervals for governance metrics across repeated seeds.
