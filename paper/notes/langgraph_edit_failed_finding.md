# LangGraph Ablation: `edit_failed` Failure Mode — Finding and Patch

Date: 2026-05-12  
Status: Patch implemented (`langgraph_manager.py` deterministic-patch bridge). Ablation rerun pending.

---

## Finding

The Priority 8b LangGraph stochastic ablation (`lg_ablation_none`,
`lg_ablation_append_only_summary`, `lg_ablation_append_only_summary_with_rationale`)
produced 15/15 `failed_invalid` trials, all with `failure_category = edit_failed`.

**Root cause.** `claw_style_worker` has two code paths:

1. **Deterministic patch path** — activated when `proposal.extra["deterministic_patch"] == True`
   and `proposal.extra["structured_edit"]` is a `{symbol, old, new, path}` dict. Applies
   the edit directly via `_replace_python_constant`, no AI coding agent required.

2. **Legacy loop path** — all other proposals. Packages the `objective` string into
   `generated_packet.json` and invokes the claw-harness legacy loop, which in turn
   expects a live AI coding agent (Claude Code / claw agent) to apply the edit.

`prompt_manager` populates `extra["deterministic_patch"]` and `extra["structured_edit"]`
via `select_structured_hyperparameter_edit`. `LangGraphManager` (prior to the patch) only
populated `context_sha256`, `raw_proposal_sha256`, `raw_proposal_chars` — so every
LangGraph proposal fell through to the legacy loop.

Without a live coding agent, the legacy loop ran `uv run train.py` against the
**unmodified** `train.py`, reported "worker did not modify train.py", and returned
`recommended_status: discard`. The control plane correctly classified these as
`edit_failed` and did not accept the baseline metric (`val_auc = 0.774711` present in
the run log) as a keep. This is correct governance behaviour — the system fails safe.

**Evidence from ledgers:**
- 15/15 trials across 3 arms: `failed_invalid / edit_failed`
- All 15 context SHA-256 hashes are unique (memory injected correctly, accumulates per trial)
- `append_only_summary` arm diverged at T3 (`increase-learning-rate` vs batch-size in the
  other arms) — confirms stochastic LLM is influenced by memory context
- `repeated_bad_rate = 0.80` for all three arms (hypothesis not confirmed, but for the
  wrong reason: proposals never ran, so behavioural effect on valid outcomes is unmeasurable)

---

## Governance Contribution

This is a positive governance result, not just a failure. The control plane:
- Detected that no edit was applied despite training running and producing a valid metric
- Classified the trial as `edit_failed` rather than silently accepting the baseline AUC
- Preserved the run artifact (log, parsed metrics) as a first-class audit object
- Did not advance `best_metric_so_far` from an uneditied run

This demonstrates **fail-safe classification**: the governance layer enforces correctness
even when a downstream component (the AI coding agent) is absent.

---

## Paper Framing (for Sections 05 and 06)

The `edit_failed` finding should be presented as a two-part result:

**Part 1 (governance correctness):** The control plane correctly rejected 15/15 trials
where training ran but no edit was applied. Final AUC is not a sufficient acceptance
criterion — the harness additionally requires that a real code change was made and
captured as a patch artifact.

**Part 2 (ablation limitation):** Because all LangGraph trials failed at the edit stage,
the ablation cannot measure whether memory context influences the quality of *applied*
proposals. The repeated-bad rate (0.80 flat) reflects proposal repetition at the
generation level, not at the execution level. A fair behavioral ablation requires closing
the edit loop.

---

## Patch: LangGraph Deterministic-Patch Bridge

**File:** `src/autoresearch/manager/langgraph_manager.py`

**Change 1 — Prompt (`_prepare_context`):**
- Include the list of available Python constants from `status.current_constants`
- Request two additional JSON fields: `"param"` (exact constant name) and `"new_value"`
  (proposed value). `"old_value"` is optional but helpful.
- Update the example to show all fields.

**Change 2 — Extraction helper (`_extract_structured_edit`):**
- Priority 1: read `param` / `old_value` / `new_value` directly from parsed JSON
- Priority 2: regex fallback on `objective` string: `change SYMBOL from OLD to NEW`
- Case-insensitive symbol lookup against `current_constants`
- Returns `{symbol, old, new, path}` or `None` if extraction fails

**Change 3 — Proposal assembly (`_validate_proposal`):**
- Call `_extract_structured_edit` after JSON parsing
- If result is not None: add `deterministic_patch: True` and `structured_edit: {...}`
  to `proposal.extra`
- Worker then routes through `_run_deterministic_constant_trial` — no coding agent needed

**Benefit:** LangGraph proposals with recognizable "change PARAM from X to Y" objectives
are applied as deterministic patches, exactly as `prompt_manager` proposals are. The only
difference is the proposal *generator* (stochastic LLM vs round-robin); the patch
application and governance layer are identical. This makes the two managers directly
comparable in a rerun of the memory ablation.

---

## Ablation Rerun Plan (after patch)

Use `scripts/run_p8b_lg_ablation.sh` (already written). The three arms will now produce
proposals that are applied as deterministic patches, enabling measurement of:
- Whether memory mode changes the distribution of kept / discarded / failed-invalid outcomes
- Whether repeated-bad rate differs across modes when proposals actually run
- Whether the `append_only_summary` arm's T3 divergence (`increase-learning-rate`) is
  reproducible and whether that proposal is kept or discarded

Expected: some trials will be kept (if the LLM proposes a valid improving change), some
discarded (valid but non-improving), some failed-invalid (bad value or precondition fails).
This is the data needed to test the hypothesis.
