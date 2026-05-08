# KDD AAE Execution Chunks

> Chunk the `KDD_AAE_refinement_plan_v2.md` into small, independently executable blocks.
> Each block has: **Goal**, **Why it matters**, **Exact implementation steps**, **Acceptance criterion**, **Effort estimate**, and **Dependencies**.
>
> Work through chunks in order within each phase. Phases can partially overlap, but Phase 1 must be substantially complete before running Phase 2 experiments.

---

## Phase 1 — Infrastructure & Pre-Flight

*Complete before running any real experiment. These are fast, low-risk, and unlock everything else.*

---

### Chunk 1.1 — Fix Package Dependencies

**Goal:** Add `langgraph`, `langchain-core`, `langchain-ollama` to `pyproject.toml` so the project installs reproducibly without `.venv` path hacks.

**Why:** Reviewers expect a reproducible install. Script-level path injection is a red flag.

**Steps:**
1. Open `pyproject.toml`.
2. Under `[project.dependencies]` (or `[tool.poetry.dependencies]`), add:
   ```toml
   langgraph = ">=0.2"
   langchain-core = ">=0.3"
   langchain-ollama = ">=0.2"
   ```
3. Remove any `sys.path.insert(..., ".venv/...")` lines from scripts.
4. Run `pip install -e ".[dev]"` in the project venv; verify no import errors.
5. Run `python -c "import langgraph; import langchain_core; import langchain_ollama; print('ok')"`.

**Acceptance criterion:** `pip install -e .` succeeds from a clean venv without any manual path manipulation.

**Effort:** 30 minutes.

**Dependencies:** None.

---

### Chunk 1.2 — Implement `scripts/reset_node_state.py`

**Goal:** A single command resets the ResNet-trigger node to a clean, reproducible state before each campaign.

**Why:** The memory ablation requires that all three modes start from identical node state. Without this, ablation comparisons are invalid.

**Steps:**
1. Create `scripts/reset_node_state.py`.
2. Implement the following logic:
   ```python
   # reset_node_state.py --node resnet_trigger [--campaign-id <id>]
   # 1. git checkout -- nodes/ResNet_trigger/train.py
   # 2. rm -rf experiments/artifacts/<campaign_id>  (if --campaign-id given)
   # 3. rm -f experiments/ledgers/<campaign_id>_trials.jsonl
   # 4. Print confirmation: "Node resnet_trigger reset to clean state."
   ```
3. Accept `--node` (required) and `--campaign-id` (optional) arguments.
4. Add a `--dry-run` flag that prints what would be deleted without doing it.
5. Test: run the script, verify `train.py` is back to original, ledger file is gone.

**Acceptance criterion:** Running the script and then `git status` shows no unexpected changes. Running twice is idempotent.

**Effort:** 1 hour.

**Dependencies:** None.

---

### Chunk 1.3 — Add No-Op Patch Guard

**Goal:** The control plane detects when a worker produces a patch that changes nothing (byte-identical before/after) and marks it `failed_invalid / no_op_patch` instead of executing training.

**Why:** No-op repeats contaminate the memory ablation and make the agent look weak. This is also a new failure taxonomy entry.

**Steps:**
1. Locate the control-plane execution path in `src/autoresearch/control_plane/` (likely `campaign.py` or `trial_runner.py`).
2. After the patch is generated but before `run_training` is called, add:
   ```python
   if patch_ref is None or patch_is_empty(patch_ref):
       record = build_trial_record(
           decision=TrialDecision.FAILED_INVALID,
           failure_category=FailureCategory.NO_OP_PATCH,
           validity_status=ValidityStatus.INVALID,
           ...
       )
       store.append(record)
       continue  # skip execution
   ```
3. Add `NO_OP_PATCH = "no_op_patch"` to `FailureCategory` enum in `src/autoresearch/memory/schemas.py`.
4. Implement `patch_is_empty(path: Path) -> bool`: returns True if patch file is missing, zero bytes, or produces no diff when applied.
5. Add a unit test in `tests/test_permissions.py` or a new `tests/test_noop_guard.py`.

**Acceptance criterion:** A dry-run campaign where the worker returns an empty patch results in a `FAILED_INVALID / no_op_patch` ledger entry. No training command is invoked.

**Effort:** 2 hours.

**Dependencies:** Chunk 1.1 (clean install).

---

### Chunk 1.4 — Add Seed / Config Logging to TrialRecord

**Goal:** Every trial record logs the seeds and config hashes needed to reproduce or challenge the result.

**Why:** The AUC gain is small; reviewers will ask if it's noise. Seed logging lets you say "all seeds logged; results are reproducible."

**Steps:**
1. Open `src/autoresearch/memory/schemas.py`.
2. Add optional fields to `TrialRecord` (all `Optional[str]`, default `None`):
   ```python
   training_seed: Optional[str] = None
   fast_config_hash: Optional[str] = None
   node_state_hash: Optional[str] = None
   patch_hash: Optional[str] = None
   ```
3. In the worker or campaign runner, compute and populate:
   - `fast_config_hash`: SHA256 of the fast-training config YAML.
   - `node_state_hash`: SHA256 of `train.py` before the patch.
   - `patch_hash`: SHA256 of the patch diff file.
4. Update `TrialRecord.to_dict()` to include new fields.
5. Verify `test_trial_schema.py` still passes (new optional fields should not break existing tests).

**Acceptance criterion:** A real trial's JSONL ledger entry includes `fast_config_hash` and `node_state_hash`. These values are identical for trials starting from the same reset state.

**Effort:** 2 hours.

**Dependencies:** Chunk 1.2 (reset script, for verifying node_state_hash consistency).

---

### Chunk 1.5 — Add Pending-Trial Recovery Path

**Goal:** If the campaign runner crashes mid-trial, the recovery tool lists and resolves stale pending guards safely.

**Why:** The ablation runs multiple campaigns. A crash could leave a `*_pending.json` that blocks future runs. Recovery must be safe and auditable.

**Steps:**
1. Create `scripts/recover_pending_trial.py`.
2. Commands:
   - `--list`: print all pending guards under `experiments/ledgers/`.
   - `--inspect <campaign_id>`: show contents of the pending guard file.
   - `--mark-failed <campaign_id> --reason <text>`: append a `FAILED_INVALID` ledger record for the pending trial, then delete the guard.
3. Implement guard location logic (reuse the existing pending-guard pattern in the control plane).
4. The `--mark-failed` option must write a valid `TrialRecord` to the ledger before deleting the guard — never leave the ledger without a record for a pending trial.

**Acceptance criterion:** Manually creating a fake `*_pending.json` file, then running `--mark-failed`, produces a ledger entry with `decision=failed_invalid` and no pending file left behind.

**Effort:** 2 hours.

**Dependencies:** Chunk 1.1.

---

## Phase 2 — Experiment Execution

*Run in order. Each experiment builds evidence for the paper. Start fresh from a reset node state for each campaign.*

---

### Chunk 2.1 — Ablation Smoke Test (1 trial × 3 memory modes)

**Goal:** Verify the campaign runner produces valid ledgers for all three memory modes before running the full 5-trial ablation.

**Why:** The full ablation is expensive. A smoke test catches integration bugs in 3 short trials instead of 15.

**Steps:**
1. Run `scripts/reset_node_state.py --node resnet_trigger`.
2. For each memory mode `[none, append_only_summary, append_only_summary_with_rationale]`:
   ```bash
   python3 scripts/run_kdd_memory_ablation.py \
     --node resnet_trigger \
     --budget 1 \
     --memory-mode <mode> \
     --campaign-id smoke_<mode> \
     --dry-run  # or real if runner is ready
   ```
3. After each run, verify:
   - [ ] Ledger file exists at `experiments/ledgers/smoke_<mode>_trials.jsonl`.
   - [ ] Ledger entry includes `memory_mode` field matching the mode.
   - [ ] No pending guard file remains.
   - [ ] `repeated_bad_count` field is present (even if 0 for 1 trial).
4. If any check fails: fix the runner integration before proceeding to Chunk 2.2.

**Acceptance criterion:** Three ledger files exist, each with one valid `TrialRecord`, `memory_mode` set correctly, no pending guards.

**Effort:** 2–4 hours (including debugging).

**Dependencies:** Chunks 1.1, 1.2, 1.3.

---

### Chunk 2.2 — Run 5-Trial Main Campaign

**Goal:** Produce the primary paper result: a real 5-trial governed campaign demonstrating complete lifecycle behavior.

**Why:** The current result is 1 baseline + 1 agent trial. This is insufficient for KDD AAE. 5 trials shows lifecycle distribution (kept / discarded / failed_invalid).

**Setup (run before starting):**
```bash
python3 scripts/reset_node_state.py --node resnet_trigger --campaign-id kdd_main_5trial
```

**Run command:**
```bash
python3 scripts/run_kdd_main_campaign.py \
  --node resnet_trigger \
  --budget 5 \
  --manager prompt_manager \
  --memory-mode append_only_summary_with_rationale \
  --campaign-id kdd_main_5trial \
  --node-root nodes/ResNet_trigger \
  --model qwen2.5-coder:7b \
  --host http://localhost:11434
```

**Post-run checks:**
- [ ] Ledger has exactly 5 records (excluding the baseline record if separate).
- [ ] At least one `kept` and at least one `discarded` or `failed_invalid` record.
- [ ] All records have `provenance` fully populated.
- [ ] Export with: `python3 scripts/export_kdd_tables.py --campaign-id kdd_main_5trial`.

**If all 5 trials are kept:** The fast-training config may be too easy. Add a constraint or increase budget to 10 and look for diversity.

**Acceptance criterion:** 5-record JSONL ledger with ≥1 non-kept decision, complete provenance on every record, `main_campaign_summary.csv` exported.

**Effort:** 4–8 hours (including training time).

**Dependencies:** Chunks 1.1, 1.2, 2.1 (smoke test passed).

---

### Chunk 2.3 — Run Forced Failure Stress Trial

**Goal:** Demonstrate that the control plane catches and records a `failed_invalid` trial with a specific failure category.

**Why:** KDD AAE reviewers will expect failure evidence. Governance systems that only show successes are not credible.

**Best stress case:** Invalid edit scope.

**Setup:**
```bash
python3 scripts/reset_node_state.py --node resnet_trigger --campaign-id kdd_stress_scope
```

**Implementation options:**
- Option A: Temporarily modify the `prompt_manager` prompt to request an edit to a frozen file (e.g., `model.py`), then run 1 trial.
- Option B: Create a special `StressWorker` that always produces a patch touching a forbidden file, then run 1 trial using it.
- Option C: Use a special `--stress-mode scope_violation` flag in the campaign runner.

**Recommended:** Option B (cleanest — doesn't require modifying prompts).

**Post-run checks:**
- [ ] Ledger entry: `decision = failed_invalid`, `failure_category = invalid_edit_scope`.
- [ ] `patch_ref` is present (the patch was generated but rejected).
- [ ] `git_commit_after` is identical to `git_commit_before` (no state corruption).
- [ ] No pending guard remains.

**Acceptance criterion:** One `failed_invalid / invalid_edit_scope` JSONL record with complete provenance and no state corruption.

**Effort:** 2–3 hours.

**Dependencies:** Chunk 1.3 (no-op guard already adds precedent for pre-execution failure).

---

### Chunk 2.4 — Run Full Memory Ablation (5 trials × 3 modes)

**Goal:** Produce the core KDD AAE result: whether rationale memory reduces repeated-bad proposals compared to no memory or summary-only memory.

**Why:** This is the headline experiment. Without it, the paper cannot claim its main governance-memory contribution.

**Pre-state the hypothesis in a file before running:**
Create `paper/notes/ablation_hypothesis.md`:
```markdown
# Memory Ablation Hypothesis

Pre-stated before running Experiment B (per scientific practice).

We hypothesise that:
  repeated_bad_rate(none) > repeated_bad_rate(append_only_summary)
                          > repeated_bad_rate(append_only_summary_with_rationale)

Grounded in:
- OpenAI (2026): "give the agent a map, not a manual" — structured failure context
  outperforms raw history.
- Böckeler (2026): feedforward guides (rationale) + feedback sensors (repeated-bad
  detector) together reduce recurrence more than feedback alone.

If the result is flat or reversed, it is a valid negative finding and will be
reported as such.
```

**Run sequence:**
```bash
for mode in none append_only_summary append_only_summary_with_rationale; do
  python3 scripts/reset_node_state.py --node resnet_trigger --campaign-id ablation_${mode}
  python3 scripts/run_kdd_memory_ablation.py \
    --node resnet_trigger \
    --budget 5 \
    --memory-mode ${mode} \
    --campaign-id ablation_${mode} \
    --manager prompt_manager \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b \
    --host http://localhost:11434
done
```

**Post-run:**
```bash
python3 scripts/export_kdd_tables.py --experiment memory_ablation
```

**Post-run checks:**
- [ ] Three ledger files, each with 5 records.
- [ ] Each record has `memory_mode` field set.
- [ ] `repeated_bad_count` and `repeated_bad_rate` populated in summary CSV.
- [ ] `memory_ablation_summary.csv` exported.

**If noisy:** Scale to 10 trials/mode and add a note: *"We increased budget to 10 trials/mode after a 5-trial pilot showed inconclusive variance."*

**Acceptance criterion:** Three complete ledgers, `memory_ablation_summary.csv` with `repeated_bad_rate` for each mode, hypothesis file committed before experiment ran.

**Effort:** 8–16 hours (including training time for 15 total real trials).

**Dependencies:** Chunks 2.1 (smoke test), 2.2 (main campaign proves runner is stable).

---

### Chunk 2.5 — Manager Comparison (optional, Priority 13)

**Goal:** Show that governance metrics are consistent across managers.

**Why:** Strengthens the claim that the control plane is manager-agnostic. Optional — only run if Chunks 2.2–2.4 are complete and time permits.

**Run:**
```bash
python3 scripts/run_manager_comparison.py \
  --node resnet_trigger \
  --budget 5 \
  --memory-mode append_only_summary_with_rationale \
  --managers baseline_manager prompt_manager
```

**Report:** Show that `provenance_completeness` and `artifact_capture_completeness` are both 100% for all managers. Do not lead with which manager got better AUC.

**Acceptance criterion:** `manager_comparison_summary.csv` exported; both managers show ≥90% provenance completeness.

**Effort:** 4–6 hours.

**Dependencies:** Chunk 2.2.

---

## Phase 3 — Analysis, Export, and Verification

*After experiments complete. Produce all paper-ready tables and figures.*

---

### Chunk 3.1 — Export All Paper Tables

**Goal:** Generate all CSV files for Tables 1–5.

**Steps:**
```bash
python3 scripts/export_kdd_tables.py \
  --main-campaign kdd_main_5trial \
  --ablation-campaigns ablation_none ablation_append_only_summary ablation_append_only_summary_with_rationale \
  --stress-campaign kdd_stress_scope \
  --output-dir paper/tables/
```

**Verify each file:**
- [ ] `main_campaign_summary.csv` — Table 1 columns present.
- [ ] `memory_ablation_summary.csv` — `repeated_bad_rate` non-null for all modes.
- [ ] `failure_taxonomy.csv` — at least one `invalid_edit_scope` row.
- [ ] `provenance_chain.csv` — kept/discarded/failed rows with completeness %.
- [ ] `accepted_discarded_invalid_counts.csv` — counts sum to budget for each campaign.

**Acceptance criterion:** All 5 CSVs exist with non-null values in all required columns.

**Effort:** 2 hours.

**Dependencies:** Chunks 2.2, 2.3, 2.4.

---

### Chunk 3.2 — Generate Paper Figures

**Goal:** Produce Figures 1–4 as publication-ready SVGs.

**Steps:**
1. **Figure 2 (repeated-bad rate bar chart):**
   ```bash
   python3 scripts/export_kdd_figures.py --figure repeated_bad_rate \
     --input paper/tables/memory_ablation_summary.csv \
     --output paper/figures/fig2_repeated_bad_rate.svg
   ```
   X-axis: 3 memory modes. Y-axis: `repeated_bad_rate`. Add error bars if 10-trial data available.

2. **Figure 3 (decision breakdown stacked bar):**
   ```bash
   python3 scripts/export_kdd_figures.py --figure decision_breakdown \
     --input paper/tables/accepted_discarded_invalid_counts.csv \
     --output paper/figures/fig3_decision_breakdown.svg
   ```

3. **Figure 4 (campaign trajectory):**
   ```bash
   python3 scripts/export_kdd_figures.py --figure trajectory \
     --input paper/tables/campaign_trajectory.csv \
     --output paper/figures/fig4_trajectory.svg
   ```

4. **Figure 1 (architecture diagram):** Draw manually or using a diagramming tool (Mermaid, draw.io, or LaTeX tikz). Should show: Manager → Control Plane ↔ Memory/Ledger; Control Plane → Worker → Training → Metric Parser → Decision.

**Acceptance criterion:** Three auto-generated SVGs render correctly in a browser; architecture diagram is available as a file.

**Effort:** 3–4 hours.

**Dependencies:** Chunk 3.1.

---

### Chunk 3.3 — Verify Artifact Completeness

**Goal:** Verify that every trial in every campaign has all expected artifacts captured.

**Steps:**
```bash
python3 scripts/check_kdd_artifact_completeness.py \
  --campaigns kdd_main_5trial ablation_none ablation_append_only_summary \
              ablation_append_only_summary_with_rationale kdd_stress_scope \
  --output paper/tables/artifact_completeness_report.txt
```

Expected output: completeness % per trial per campaign.

**What to check:**
- [ ] All `patch_ref` paths exist as files.
- [ ] All `raw_log_ref` paths exist.
- [ ] All `parsed_metrics` dicts are non-empty for valid trials.
- [ ] All `decision_id` fields are non-null.

**If any artifact is missing:** investigate and either re-run the trial or document the gap in Limitations.

**Acceptance criterion:** Report shows ≥90% artifact completeness across all campaigns; any gaps are documented.

**Effort:** 2 hours.

**Dependencies:** Chunks 2.2–2.4.

---

### Chunk 3.4 — Create Artifact Manifest

**Goal:** Create `artifact_manifest.json` — a machine-readable index of every campaign, ledger, artifact, table, and figure used in the paper.

**Why:** Reviewers can use this to verify reproducibility. It is also the `nice-to-have` from the checklist that takes only 30 minutes.

**Steps:**
1. Create `artifact_manifest.json` in the repo root.
2. Structure:
   ```json
   {
     "paper": "KDD AAE 2026",
     "generated": "2026-MM-DD",
     "campaigns": [
       {"id": "kdd_main_5trial", "ledger": "experiments/ledgers/kdd_main_5trial_trials.jsonl", "trials": 5}
     ],
     "tables": ["paper/tables/main_campaign_summary.csv", ...],
     "figures": ["paper/figures/fig2_repeated_bad_rate.svg", ...],
     "environment": {"python": "3.11", "model": "qwen2.5-coder:7b"},
     "run_commands": ["python3 scripts/run_kdd_main_campaign.py ..."]
   }
   ```
3. Commit and reference from `README.md`.

**Acceptance criterion:** `artifact_manifest.json` is valid JSON; all referenced files exist.

**Effort:** 1 hour.

**Dependencies:** Chunk 3.1 (tables exist before manifest references them).

---

## Phase 4 — Paper Writing

*Write in section order. Complete experiments before writing Results. Introduction and Abstract go last.*

---

### Chunk 4.1 — Write Related Work Section

**Goal:** A Related Work section that positions the paper against AIDE, harness engineering literature, and experiment trackers — and wins.

**Key content:**
1. **AIDE (Jiang et al. 2025) — closest related:** Insert Table 5 (AIDE comparison, §7.3 of refinement plan v2). Conclude: *"AIDE optimises the ML metric; we govern the experimentation process. These are complementary goals."*
2. **Harness engineering literature (1 paragraph):** Cite Trivedy (Mar 2026), Böckeler (Apr 2026), OpenAI (2026), Anthropic (2025). Use: *"Designing the environments, feedback loops, and control systems is now recognized as the primary engineering challenge for agentic AI."* These sources validate the project's framing.
3. **Experiment trackers (MLflow, W&B):** One sentence: they record outcomes but do not govern the agent loop or enforce bounded execution.
4. **AutoML (NAS, HPO):** One sentence: they optimise within a pre-defined search space; we audit the agent's self-directed exploration.
5. **Other agent systems (ml-intern, multiautoresearch, deep-research, Hermes):** These motivate implementation patterns but do not define an evaluation protocol for governed autonomous experimentation.
6. **Mind2Web (Deng et al. 2023):** Contrast: *"Unlike benchmark-driven evaluation across many domains, we prioritise a single real task node with complete provenance."*

**Writing rule:** End Related Work with the gap: *"None of the above provides the full governed control plane: scope enforcement, append-only audit ledger, failure taxonomy, and memory ablation for evaluating autonomous experimentation."*

**Acceptance criterion:** Related Work cites AIDE with the comparison table; cites ≥3 harness engineering sources; clearly states the gap the paper fills.

**Effort:** 3–4 hours.

**Dependencies:** None (can be written before experiments finish).

---

### Chunk 4.2 — Write System Design Section (Section 3)

**Goal:** Describe the governed control plane architecture clearly, using the guides/sensors vocabulary.

**Content to include:**
1. Manager/worker separation and the replacement principle.
2. Trial lifecycle state machine (diagram or pseudocode).
3. Editable-scope enforcement.
4. Pending-trial guard.
5. Append-only ledger.
6. Memory modes.
7. Keep/discard/failed-invalid decision authority — *and why it lives in the control plane, not the manager* (cite Anthropic/Rajasekaran 2026 on self-evaluation leniency).
8. **Add: Guides/Sensors 2×2 table** (§9.6 of refinement plan v2).
9. **Add: Failure mode motivation table** (§9.7 of refinement plan v2).

**Key sentence for decision authority:**
> *"We keep the keep/discard decision in the control plane, not delegated to the manager. Rajasekaran et al. (2026) found that LLM self-evaluation is systematically lenient — agents confidently praise mediocre work. A deterministic, externally-owned decision criterion addresses this."*

**Acceptance criterion:** Section 3 contains both tables; clearly states that managers cannot commit trial state.

**Effort:** 4–5 hours.

**Dependencies:** Chunk 4.1 (guides/sensors vocabulary introduced in Related Work).

---

### Chunk 4.3 — Write Experiments Section (Section 4)

**Goal:** Describe the four experiments with enough detail for reproducibility.

**Content:**
1. ResNet-trigger benchmark: what it is, why it's a real scientific ML node (not a toy), what `train.py` does, why val_AUC is the metric.
2. Fixed-budget campaign protocol: budget, manager, memory mode, worker, acceptance rule.
3. Memory ablation design: three modes, equal conditions, **pre-stated hypothesis** (copy from `ablation_hypothesis.md`), minimum budget, how repeated-bad rate is computed.
4. Stress trial: what kind of violation, what the control plane is expected to do.
5. Governance metrics: definitions for each metric used in results.
6. Failure taxonomy: Table 2 (all 6 categories with definitions).

**Key instruction:** Pre-state the memory ablation hypothesis *in the Experiments section*, before Results. This is required for scientific credibility.

**Acceptance criterion:** Hypothesis is stated before results; failure taxonomy table is in the Experiments section (not buried in an appendix); a reader can reproduce the experimental setup from the text alone.

**Effort:** 4–5 hours.

**Dependencies:** Chunks 2.1–2.4 complete (you need final experimental design to describe it accurately).

---

### Chunk 4.4 — Write Results Section (Section 5)

**Goal:** Present results in governance-first order. Never open with AUC.

**Order:**
1. **Main campaign governance** (Table 1): lifecycle distribution, acceptance rate, provenance completeness, artifact completeness.
2. **Memory ablation** (Figure 2 + Table 3): repeated-bad rate by mode; does the pattern match the hypothesis?
3. **Decision breakdown** (Figure 3): kept/discarded/failed_invalid across modes.
4. **Failure taxonomy** (Table 2): counts, categories, control-plane response.
5. **Provenance chain** (Table 4): completeness across trial types.
6. **Task metric** (Figure 4, secondary): AUC trajectory. Write: *"The accepted edit also improved validation AUC by 0.002845; we report this as secondary evidence that the governed loop can execute meaningful real experiments."*

**What to do if ablation result is flat:**
> *"Repeated-bad rate was [X]% across all memory modes, suggesting that the manager is insensitive to memory content under the current budget and node. This negative finding indicates that governance memory effects may require larger budgets or more failure opportunities to manifest."*

**Acceptance criterion:** Results section contains all 4 tables + 3 figures; AUC is the last result presented, not the first.

**Effort:** 3–4 hours.

**Dependencies:** Chunks 3.1, 3.2.

---

### Chunk 4.5 — Write Discussion and Limitations (Section 6)

**Goal:** Pre-empt every reviewer concern proactively.

**Content:**
1. What is proven (bounded execution, auditability, failure classification, real execution).
2. What is not proven (generalisation to other nodes, memory effects at scale, manager superiority).
3. Why governance metrics matter beyond AUC.
4. **Limitations — use these four paragraphs verbatim from §8 of refinement plan v2:**
   - Single benchmark node.
   - No holdout evaluation node (cite Better-Harness / Trivedy Apr 2026).
   - Dry-run tests are not full evidence.
   - Future backend extensions.
5. Generalisation path: *"The NodeSpec YAML pattern generalises the harness to new ML experiments without code changes; each spec is a harness template for a class of experiments."*

**Acceptance criterion:** Discussion includes the holdout gap as a named limitation; non-claims paragraph from §2.3 appears verbatim.

**Effort:** 2–3 hours.

**Dependencies:** Chunk 4.4 (you know what was proven).

---

### Chunk 4.6 — Rewrite Abstract and Introduction

**Goal:** Use the harness-first framing. Write last, after all results are known.

**Abstract:**
- Use post-results template from §9.2 of refinement plan v2.
- Fill in `[BEST_VAL_AUC]`, `[INITIAL_VAL_AUC]`, `[X]%`, `[A]%`, `[B]%` from actual results.
- Key phrase to include: *"In the Agent = Model + Harness decomposition, our harness owns trial lifecycle, scope enforcement, and audit; the LLM manager is a replaceable component."*

**Introduction:**
- Open with the evaluation problem (§9.4 hook from refinement plan v2).
- Use: *"Designing the environments, feedback loops, and control systems that close this gap is now recognized as a primary engineering challenge for agentic AI (OpenAI 2026; Böckeler 2026)."*
- State contributions as a numbered list.
- State non-claims (§2.3 of refinement plan v2) explicitly in Introduction — do not hide them in Discussion.

**Acceptance criterion:** Abstract fills in all bracketed values; Introduction opens with the evaluation gap, not with the system.

**Effort:** 2–3 hours.

**Dependencies:** All previous writing chunks.

---

## Phase 5 — Repository Hygiene and Reproducibility

*Parallel with Phase 4. Can be done any time after Phase 2.*

---

### Chunk 5.1 — Implement LangChain Proposal Backend (Phase 1 of integration)

**Goal:** Add `LangChainProposalBackend` so the paper can claim backend-agnosticism with evidence.

**Why:** Without this, the backend-agnosticism claim is a design claim only. With it, you can run one real campaign and verify the JSONL schema is identical.

**Steps:**
1. Create `src/autoresearch/llm/langchain_client.py`.
2. Implement `LangChainProposalBackend` with interface:
   ```python
   class LangChainProposalBackend:
       def __init__(self, model: str, host: str): ...
       def propose(self, campaign_state: CampaignState) -> Proposal: ...
   ```
3. Use `ChatOllama` from `langchain-ollama` for local endpoint.
4. Use structured output (Pydantic `ExperimentProposal` model) for the proposal.
5. Save raw prompt/response as artifact refs.
6. Add `--llm-backend langchain` flag to campaign runner.
7. Run one real 1-trial campaign with LangChain backend.
8. Verify JSONL ledger schema is identical to native backend.

**Acceptance criterion:** One ledger entry generated with `--llm-backend langchain`; schema diff against native backend is empty.

**Effort:** 4–6 hours.

**Dependencies:** Chunk 1.1 (deps installed).

---

### Chunk 5.2 — Update README and Documentation

**Goal:** Make real campaigns the documented canonical path. Remove or demote legacy Stage 1 descriptions.

**Steps:**
1. In `README.md`:
   - Add "Real Campaign Quickstart" section (based on Chunk 2.2 run command).
   - Add reference to `artifact_manifest.json`.
   - Add "Reproduce KDD AAE results" section with ordered steps.
2. In `docs/architecture.md`:
   - Add Guides/Sensors 2×2 table.
   - Update control-plane flow diagram to reflect current state.
3. In `docs/stage_2_current_structure.md` (if exists):
   - Mark Stage 2 real campaigns as the canonical path.
   - Move legacy Stage 1 description to a "Historical" section.

**Acceptance criterion:** A new contributor can run a real campaign by following README alone, without reading any other docs.

**Effort:** 2–3 hours.

**Dependencies:** Chunk 2.2 (need real campaign command to document).

---

### Chunk 5.3 — Create Paper LaTeX Structure

**Goal:** Create the `paper/kdd_aae_2026/` directory with section stubs so writing is modular.

**Steps:**
```bash
mkdir -p paper/kdd_aae_2026/sections
mkdir -p paper/kdd_aae_2026/tables
mkdir -p paper/kdd_aae_2026/figures
touch paper/kdd_aae_2026/main.tex
touch paper/kdd_aae_2026/sections/{introduction,related_work,system,experiments,results,discussion,conclusion}.tex
```

2. In `main.tex`, add `\input{sections/...}` for each section file.
3. Copy final CSV tables to `paper/kdd_aae_2026/tables/`.
4. Copy SVG/PDF figures to `paper/kdd_aae_2026/figures/`.

**Acceptance criterion:** `pdflatex paper/kdd_aae_2026/main.tex` compiles without errors (even if sections are stubs).

**Effort:** 1 hour.

**Dependencies:** Chunks 3.1, 3.2.

---

## Appendix: Chunk Dependency Graph

```
Phase 1 (Infra):
  1.1 → 1.2 → 1.3
  1.1 → 1.4
  1.1 → 1.5

Phase 2 (Experiments):
  1.1 + 1.2 + 1.3 → 2.1 (smoke)
  2.1 → 2.2 (main campaign)
  1.3 → 2.3 (stress trial)
  2.1 + 2.2 → 2.4 (full ablation)
  2.2 → 2.5 (manager comparison, optional)

Phase 3 (Analysis):
  2.2 + 2.3 + 2.4 → 3.1 (export tables)
  3.1 → 3.2 (figures)
  2.2 + 2.3 + 2.4 → 3.3 (artifact completeness)
  3.1 → 3.4 (manifest)

Phase 4 (Writing):
  independent → 4.1 (related work)
  4.1 → 4.2 (system design)
  2.4 → 4.3 (experiments)
  3.1 + 3.2 → 4.4 (results)
  4.4 → 4.5 (discussion)
  4.5 → 4.6 (abstract + intro)

Phase 5 (Hygiene):
  1.1 → 5.1 (langchain backend)
  2.2 → 5.2 (docs)
  3.1 + 3.2 → 5.3 (latex structure)
```

---

## Appendix: Readiness Level Tracker

Update this table as chunks complete:

| Chunk | Description | Status |
|-------|-------------|--------|
| 1.1 | Fix packaging | ✅ done |
| 1.2 | reset_node_state.py | ✅ done |
| 1.3 | No-op patch guard | ✅ done |
| 1.4 | Seed logging | ✅ done |
| 1.5 | Pending-trial recovery | ⬜ pending |
| 2.1 | Ablation smoke test | ⬜ pending |
| 2.2 | 5-trial main campaign | ⬜ pending |
| 2.3 | Stress trial | ⬜ pending |
| 2.4 | Full memory ablation | ⬜ pending |
| 2.5 | Manager comparison | ⬜ optional |
| 3.1 | Export tables | ⬜ pending |
| 3.2 | Generate figures | ⬜ pending |
| 3.3 | Artifact completeness check | ⬜ pending |
| 3.4 | Artifact manifest | ⬜ pending |
| 4.1 | Write Related Work | ⬜ pending |
| 4.2 | Write System Design | ⬜ pending |
| 4.3 | Write Experiments | ⬜ pending |
| 4.4 | Write Results | ⬜ pending |
| 4.5 | Write Discussion + Limitations | ⬜ pending |
| 4.6 | Rewrite Abstract + Intro | ⬜ pending |
| 5.1 | LangChain backend | ⬜ pending |
| 5.2 | Update README + docs | ⬜ pending |
| 5.3 | LaTeX structure | ⬜ pending |

**Minimum for Level 2 (acceptable workshop paper):** Chunks 1.1–1.3, 2.1–2.4, 3.1–3.2, 4.1–4.6.
**Target for Level 3 (strong workshop paper):** All of the above + 2.5, 3.3–3.4, 5.1–5.3.
