# KDD AAE Refinement Plan v2 — `autoresearch_harness`

> **Skeleton:** `kdd_aae_refinement_plan_md` (Sections 0–18)
> **Enriched with:** `harness_literature_synthesis.md` (AIDE comparison, Guides/Sensors taxonomy, Anthropic self-eval finding, Better-Harness holdout gap) and `checklist_stage3.md` (packaging, LangGraph, artifact capture, Stage 2 promotion)
> **Last revised:** 2026-05-08

---

## Framing Principle

> **Agent = Model + Harness** (Trivedy 2026). The governed control plane *is* the harness. Its defining property is that it owns trial lifecycle, scope enforcement, and audit — the LLM manager is a replaceable, interchangeable component.

Use this framing in every section, every sentence. If a sentence does not reinforce it, question whether it belongs.

---

## 0. Current Status

### Already Strong

- Real ResNet-trigger full-loop demonstration.
- Actual baseline metric and post-agent metric.
- Bounded edit: `LEARNING_RATE = 1e-3` → `5e-4`.
- Valid accepted trial with complete artifact chain (proposal, packet, patch, run log, parsed metrics, JSONL ledger, paper CSV exports).
- Governance metrics already exported: acceptance rate, invalid rate, editable-scope violations, command failure rate, metric parsing failure rate, artifact capture completeness.
- Notebook and SVG plots generated.

### Current Paper-Facing Result

| Item | Value |
|------|-------|
| Benchmark node | `resnet_trigger` |
| Campaign id | `resnet_real_incremental` |
| Manager | `baseline_manager` |
| Memory mode | `append_only_summary_with_rationale` |
| Worker | `claw_style_worker` |
| Baseline val_AUC | `0.779911` |
| Final val_AUC | `0.782756` |
| Net gain | `+0.002845` |
| Records | 2 total (one baseline, one agent trial) |
| Decision | `kept` |
| Provenance completeness | complete |

### Current Evidence Level: Level 1 — Weak/Borderline

Supports: "Stage 2 converts an autoresearch-style worker loop into an auditable, governed experiment protocol."

Does **not** yet support: "The governance mechanism improves agent behavior across repeated trials."

---

## 1. Target Paper Identity

### Wrong Identity
> A local LLM agent improved a ResNet trigger by changing the learning rate.

### Correct Identity
> A governed harness for autonomous ML experimentation, where the control plane owns trial lifecycle, enforces bounded execution, records decisions append-only, and reports auditable governance metrics.

### One-Sentence Thesis
> Autonomous ML experimentation becomes scientifically credible only when the agent loop is governed by an explicit control plane that owns trial lifecycle, enforces bounded execution, records decisions append-only, and reports auditable governance metrics — independent of the specific LLM manager or worker backend.

### Target Contribution Type: Evaluation Methodology + Governed Harness Design

Not: AutoML method, ResNet-trigger optimizer, coding-agent benchmark, detector-physics result, general autonomous scientist.

### Harness Literature Alignment
The harness engineering literature (Trivedy 2026, Böckeler 2026, Rajasekaran 2026, OpenAI 2026, Anthropic 2025, Hashimoto 2026) independently converges on the same insight: **the governance/control layer is the hard open problem, not model capability**. The paper's positioning is exactly correct and increasingly well-supported in the field.

---

## 2. Claims, Conditional Claims, and Non-Claims

### 2.1 Claims Currently Supported

1. **Bounded execution improves auditability.** The real run records proposal, bounded edit, training command, metric parsing, validity, decision, and artifacts.
2. **Explicit lifecycle control makes autonomous experiments inspectable.** The control plane owns state transition and records authoritative decisions.
3. **Governed experimentation can execute a real scientific ML node.** The ResNet-trigger campaign is a real training execution, not a dry run.
4. **The framework is backend-agnostic.** Claw-style/Ollama/Qwen is the current worker; the architecture is described as manager/worker-agnostic.

### 2.2 Claims Requiring New Evidence

1. **Rationale memory reduces repeated poor proposals.** — Requires real memory ablation. *Hypothesis (grounded in OpenAI 2026 and Böckeler 2026): rationale > summary > none on repeated-bad rate.*
2. **Governance behavior holds across multiple trials.** — Requires ≥5-trial real campaign.
3. **Control plane catches invalid/failed actions.** — Requires one forced failure or stress trial.
4. **Governance is manager-agnostic.** — Requires manager comparison, or remains a design claim.

### 2.3 Explicit Non-Claims (include verbatim in paper)

> This work does not claim to build a general autonomous scientist, prove scientific discovery, introduce a universal optimization algorithm, or depend on a specific coding-agent backend. The ResNet-trigger task is used as a real scientific ML case study for evaluating governed autonomous experimentation; it is not claimed to represent all ML optimization tasks.

---

## 3. Core Reviewer Risks and Fixes

| Risk | Fix |
|------|-----|
| "One-trial demo" | Run 5–10 real trials; report lifecycle distribution |
| "AUC gain too small" | Lead with governance metrics; AUC is secondary evidence |
| "Memory claim unproven" | Run real ablation; make repeated-bad rate the headline |
| "Too domain-specific" | Frame ResNet-trigger as controlled scientific ML case study; minimize detector details |
| "Weak related work" | Add AIDE comparison table (see §2 of harness synthesis); cite Böckeler, OpenAI, Anthropic |
| "No holdout evaluation" *(new from synthesis)* | Acknowledge in Limitations: single node, no holdout. Cite Better-Harness (Trivedy Apr 2026) |

---

## 4. Required Experiments

### Experiment A — Real Main Campaign

**Purpose:** Upgrade one-trial demo to multi-trial governed campaign.

**Minimum:** 5 real trials, `prompt_manager`, `append_only_summary_with_rationale`, clean node state.

**Stronger:** 10 real trials with ≥1 discarded trial or invalid/failure case, full trajectory exported.

**Metrics to report:**

| Metric | Why It Matters |
|--------|---------------|
| Initial / best / final val_AUC | Task performance evidence |
| Net gain, gain per budget unit | Efficiency under fixed budget |
| Kept / discarded / invalid count | Lifecycle behavior |
| Acceptance rate | Decision behavior |
| Invalid rate | Safety behavior |
| Complete provenance rate | Auditability |
| Artifact capture completeness | Reproducibility |
| Total wall-clock seconds | Practical cost |

**Success criterion:** Complete governed lifecycle behavior over multiple trials, not a large AUC gain.

---

### Experiment B — Memory / Governance Ablation *(the core KDD AAE experiment)*

**Purpose:** Test whether memory changes agent behavior (specifically repeated poor proposals).

**Modes:**

| Mode | Expected Role |
|------|--------------|
| `none` | Weak baseline |
| `append_only_summary` | Intermediate baseline |
| `append_only_summary_with_rationale` | Main method |

**Fixed conditions:** same node, manager, worker, model, budget, fast training config, acceptance rule, editable scope, starting state.

**Budget:** 5 trials/mode minimum (15 total real trials); 10/mode if noisy (30 total).

**Pre-stated hypothesis** *(required before results — grounded in OpenAI 2026, Böckeler 2026)*:
> Rationale-augmented memory will reduce the repeated-bad rate more than raw summaries because it gives the manager a targeted failure signal rather than history noise.

**Primary metric:**
```
RepeatedBadRate = repeated_bad_proposals / total_proposals
```

**Operational definition of repeated-bad:** A proposal repeats the same edit target and edit mechanism as a prior rejected/invalid/degraded trial *without* adding a new corrective rationale, constraint, or justification.

**Desired pattern:** `none > append_only_summary > append_only_summary_with_rationale` on repeated-bad rate. Task metric need not follow the same pattern.

**If noisy:** increase to 10 trials/mode, add bootstrap intervals, state that memory effects are preliminary.

---

### Experiment C — Forced Failure / Stress Trial

**Purpose:** Governance systems must show they catch invalid actions.

**Best single case:** Invalid edit scope (unambiguous governance demonstration).

```
worker attempts forbidden edit
  → control plane detects scope violation
  → trial marked failed_invalid / invalid_edit_scope
  → ledger records failure category
  → no state corruption
```

**Other valid stress cases:** metric missing, syntax error, command failure, no-op patch.

**Paper framing:** Failed trials are first-class audit objects, not embarrassments. Write: *"Unlike conventional experiment trackers that mainly record successful runs, the control plane records invalid and failed trials as first-class audit objects."*

---

### Experiment D — Manager Comparison *(optional; Priority 13)*

**Purpose:** Show governance is not tied to one manager.

**Compare:** `baseline_manager` vs. `prompt_manager` (same node, memory mode, budget, worker).

**Do not emphasize:** which manager gets better AUC.

**Emphasize:** governance metrics are produced consistently regardless of manager.

---

## 5. Metrics to Refine

### 5.1 Repeated-Bad Rate — Definition

> We define a repeated-bad proposal as a proposal that repeats the same edit target and edit mechanism as a prior rejected, invalid, or degraded trial without adding a new corrective rationale. We compute the rate over all proposals in a fixed-budget campaign.

### 5.2 Failure Taxonomy *(present as Table 2 in the paper)*

| Category | Definition |
|----------|-----------|
| `invalid_edit_scope` | Patch touched a disallowed file or region |
| `syntax_error` | Code cannot be parsed or imported |
| `runtime_error` | Training command exits nonzero |
| `metric_missing` | Run completes but expected metric cannot be parsed |
| `degraded_metric` | Valid run, worse than current best — discarded (not failed_invalid) |
| `no_op_patch` | Worker produced no effective change |

**Key distinction:**
- `failed_invalid`: violates validity or cannot produce metric
- `discarded`: valid but not kept because metric did not improve
- `kept`: valid and accepted

**Note:** This is the first formal taxonomy of autonomous ML experimentation failure modes derived from real campaigns. Present it as a contribution.

### 5.3 Provenance Completeness

```
ProvenanceCompleteness = completed_required_artifact_links / total_required_artifact_links
```

Required fields: `proposal_id`, `worker_packet_ref`, `patch_ref`, `run_log_ref`, `parsed_metrics_ref`, `decision_id`, `ledger_record_id`.

Report separately for kept / discarded / failed-invalid trials.

### 5.4 Artifact Capture Completeness

```
ArtifactCaptureCompleteness = captured_artifacts / expected_artifacts
```

Expected: proposal JSON, worker packet, patch diff, raw run log, parsed metrics, control-plane decision record, ledger entry.

---

## 6. System / Implementation Refinements

### 6.1 No-Op Patch Guard *(new failure category)*

```python
if patch_is_empty or patch_matches_current_state:
    decision = "failed_invalid"
    failure_category = "no_op_patch"
    write_ledger_record(...)
    skip_execution()
```

### 6.2 Reproducible State Reset

Before each ablation campaign:
```bash
git checkout -- nodes/ResNet_trigger/train.py
rm -rf experiments/artifacts/<campaign_id>
rm -f experiments/ledgers/<campaign_id>_trials.jsonl
```

Or implement `scripts/reset_node_state.py --node resnet_trigger`.

Required paper statement: *"Each ablation campaign was initialized from the same node state and executed under the same fixed budget and training configuration."*

### 6.3 Seed and Configuration Logging

Log per trial: `random_seed`, `training_seed`, `data_seed`, `model_seed`, `fast_config_hash`, `node_state_hash`, `patch_hash`, `command_hash`.

### 6.4 Bootstrap / Repeated Baseline *(if time allows)*

Run 3 repeated baselines under the original config (different seeds). Report `baseline mean ± std` vs. best governed result. If gain is within variance, frame as: *"The optimization gain is treated as secondary; the primary result is that the control plane produces complete and auditable lifecycle records under real execution."*

### 6.5 Memory Ablation Runner Stability

Before the 5-trial ablation, run a 1-trial smoke across all three memory modes:
1. Verify each mode creates a ledger.
2. Verify each ledger includes `memory_mode`.
3. Verify repeated-bad export is non-zero when expected.
4. Only then run 5- or 10-trial version.

### 6.6 Fix Packaging *(from checklist_stage3)*

Add to `pyproject.toml` runtime dependencies: `langgraph`, `langchain-core`, `langchain-ollama`. Remove script-level `.venv` site-package path hacks.

### 6.7 Artifact Capture Completeness *(from checklist_stage3)*

Ensure real-run artifact capture records:
- generated packet path
- patch diff
- raw log
- commit before/after
- parsed metric payload
- manager raw output or hash

### 6.8 Pending-Trial Recovery *(from checklist_stage3)*

Add recovery path:
- list pending campaigns
- inspect pending guard
- mark failed and append failure record
- clear stale guard safely

---

## 7. Related Work and External System Lessons

### 7.1 What to Take from Each External System

| System | Useful Pattern | Add to Plan | Do Not Claim |
|--------|---------------|-------------|--------------|
| `dzhng/deep-research` | Iterative loop with breadth/depth controls, accumulated learnings, final report | Breadth/depth campaign params; per-trial learning object; `campaign_report.md` | Web research or literature search as contribution |
| `burtenshaw/multiautoresearch` | Multi-track project organization | Multi-node roadmap as future work; clean `NodeSpec` YAML | Broad multi-benchmark coverage until more nodes run |
| `huggingface/ml-intern` | Event model, headless/interactive modes, local models, trace export | Headless campaign mode for paper; event taxonomy; optional notification hooks | Full ML engineer capability |
| `NousResearch/hermes-agent` | Long-lived memory, skills/routines, trajectory compression | Skill/routine registry as future work; trajectory compression for memory summaries | Self-improving system in Hermes sense |
| `langchain-ai/langchain` | Model/tool abstractions, LangGraph orchestration, LangSmith observability | LangChain for model/tool interface; LangGraph optional backend; LangSmith optional tracing | Replace append-only ledger with LangSmith/LangGraph checkpoints |

### 7.2 Harness Literature Lessons *(new from synthesis — cite in Related Work)*

| Source | Key Lesson | Application |
|--------|-----------|-------------|
| Trivedy (Mar 2026) — Anatomy | Agent = Model + Harness; harness includes hooks/middleware for deterministic execution | Frame control plane as the harness by this canonical definition |
| Böckeler (Apr 2026) — HE for coders | Guides (feedforward) × Sensors (feedback) × Computational/Inferential taxonomy | Label each component in Section 3; add 2×2 table (see §9.5) |
| Böckeler (Feb 2026) — HE first thoughts | Increasing trust required constraining solution space; harnesses become service templates | Frame editable-scope restriction as reliability mechanism, not simplification |
| Hashimoto (Feb 2026) | Harness engineering = fixing agent mistakes permanently; two forms: AGENTS.md + programmed tools | Cite as practitioner validation; failure taxonomy is the structured form |
| OpenAI/Lopopolo (Feb 2026) | "Give agent a map, not a manual"; failure = signal → fix what's missing and make it enforceable | Memory rationale mode is the formal version of this ad-hoc process |
| Anthropic/Young (Nov 2025) | Four agent failure modes all have ML experimentation analogs; git commit per session = clean state | Map failure modes to harness fixes in §9.4 table |
| Anthropic/Rajasekaran (Mar 2026) | Self-evaluation is systematically lenient; separate generator from evaluator | Validates keeping keep/discard in control plane, not delegated to manager |
| Trivedy (Apr 2026) — Better-Harness | Evals are training data for harnesses; holdout sets prevent overfitting | Acknowledge single-node gap in Limitations; cite as future work methodology |
| Jiang et al. (2025) — AIDE | Closest academic system; merges control layer with LLM; lacks scope enforcement, audit, failure taxonomy | Must compare explicitly in Related Work |
| Deng et al. (2023) — Mind2Web | Broad benchmark at cost of reproducibility | Contrast: we choose deep reproducibility over breadth |

### 7.3 AIDE Comparison *(required in Related Work)*

**AIDE (Jiang et al. 2025)** is the most directly related academic system. Both use LLMs to propose atomic code changes evaluated by a hard-coded metric. The decisive structural difference:

| Dimension | AIDE | autoresearch_harness |
|-----------|------|---------------------|
| LLM role | Proposes code + selects next node | Proposes change only |
| Control plane | Hard-coded tree search + evaluator (monolithic) | Governed lifecycle state machine (separated) |
| Keep/discard owner | Hard-coded metric h(s) | Control plane (deterministic, auditable) |
| Scope enforcement | None | Editable-path whitelist |
| Audit ledger | None | Append-only JSONL with provenance |
| Failure taxonomy | None | 5-category taxonomy |
| Memory / context | Summarization operator Σ(T) | 3-mode ablation experiment |
| Reproducibility claim | Performance on Kaggle | Full provenance from proposal to commit |

**Positioning:** *"AIDE optimises the ML metric; autoresearch_harness governs the experimentation process. These are complementary goals."*

### 7.4 LangGraph / LangChain Integration Plan

**Architecture principle:** LangChain provides model/tool abstraction. LangGraph optionally orchestrates the manager-worker flow. The control plane remains the governance authority. The append-only ledger remains ground truth — not LangSmith checkpoints.

**Phase 1 (implement first):** `LangChainProposalBackend` in `src/autoresearch/llm/langchain_client.py` — initialize chat model, produce structured proposal JSON, log raw prompt/response as artifact, return `ManagerProposal`.

**Phase 2:** Structured proposal output via `ExperimentProposal(BaseModel)` — `edit_target`, `edit_intent`, `expected_effect`, `risk_level`, `allowed_paths`, `rationale`, `repeat_check`.

**Phase 3 (after Phase 1–2 stable):** Optional LangGraph runner at `src/autoresearch/orchestration/langgraph_runner.py`. Flag `--orchestrator native|langgraph`. Governance result must be identical in both modes.

**Phase 4:** Optional LangSmith tracing as secondary observability only.

**Paper wording:** *"Our framework sits below common agent orchestration layers. Managers may use direct LLM calls, LangChain abstractions, or a LangGraph workflow. In all cases, the control plane owns trial validity, keep/discard decisions, append-only ledger updates, and paper-facing governance metrics."*

---

## 8. Limitations *(state proactively — makes paper more credible)*

### 8.1 Single Benchmark Node
> We evaluate on one real scientific ML node. This demonstrates real governed execution, but does not claim broad benchmark coverage.

**Harness literature alignment:** Better-Harness (Trivedy Apr 2026) shows holdout sets are essential to prevent harness overfitting; our single-node evaluation cannot rule out overfitting to this domain. Generalisation across node types is future work.

### 8.2 No Holdout Evaluation Node *(new, from synthesis)*
> All experiments use the ResNet-trigger node. We cannot rule out that reported improvements overfit to this evaluation domain. Applying the harness hill-climbing methodology of Trivedy (2026) across multiple evaluation nodes with holdout splits would strengthen the governance claims.

### 8.3 Memory Ablation Stability
> The memory ablation is a planned validation of the governance-memory claim. Real ablation results require verifying each mode starts from the same node state and produces real campaign ledgers.

### 8.4 Dry-Run Tests Are Not Full Evidence
> Dry-run tests validate control-plane contracts; reported empirical results use real worker campaigns unless explicitly marked as smoke tests.

### 8.5 Future Backend Extensions
> Orchestration backends, cloud execution, and UI layers are future extensions. This work focuses on the control-plane protocol and its audit metrics.

---

## 9. Paper Structure

### 9.1 Recommended Title

**Best:** *Evaluating Governed Autonomous Experimentation with Bounded Execution and Auditable Memory*

Alternatives:
- *A Governance Harness for Evaluating Agentic AI in Autonomous ML Experimentation*
- *Harness-Mediated Evaluation of Autonomous Research Agents*
- *Auditable Autonomous Experimentation with Explicit Lifecycle Control*

### 9.2 Abstract (Before Ablation Results)

```
Autonomous ML experimentation agents can propose code edits, execute training
runs, and select follow-up trials, but their evaluation remains difficult because
conventional task metrics do not capture invalid actions, repeated failures,
auditability, or lifecycle governance. We present a governed harness for autonomous
ML experimentation. In the Agent = Model + Harness decomposition, our harness owns
trial lifecycle, scope enforcement, and audit; the LLM manager is a replaceable
component. The framework separates proposal, execution, validation, and decision
authority; enforces editable-scope constraints; records append-only trial ledgers;
and exports governance metrics including invalid-action rate, repeated-bad rate,
artifact completeness, and provenance completeness. We demonstrate the framework on
a ResNet-trigger scientific ML benchmark with a real full-loop campaign that executes
a bounded code edit and records complete provenance. We further define a
memory-ablation protocol — consistent with harness engineering practice (Trivedy
2026; Böckeler 2026) — comparing no memory, summary memory, and rationale-linked
append-only memory under fixed budget. The study argues that agentic
experimentation should be evaluated not only by final task performance, but also by
whether agent behavior is bounded, auditable, failure-aware, and reproducible.
```

### 9.3 Abstract (After Ablation Results)

```
[Replace last two sentences with:]
On a ResNet-trigger scientific ML benchmark, a real governed campaign achieves
[BEST_VAL_AUC] from [INITIAL_VAL_AUC] while maintaining [X]% provenance
completeness. A memory ablation across no-memory, summary-memory, and
rationale-memory modes shows repeated-bad rate decreasing from [A]% to [B]% under
equal budget. These results show that governance metrics expose agent behavior
beyond final task performance and provide a reproducible basis for evaluating
autonomous experimentation systems.
```

### 9.4 Introduction Hook

> Autonomous agents are increasingly able to plan experiments, edit code, execute training runs, and select follow-up actions. However, evaluating these systems by final task score alone is insufficient: an agent may reach a good result while repeatedly attempting invalid edits, hiding failed runs, corrupting experimental state, or making decisions that cannot be audited. This creates a gap between autonomous experimentation and scientifically credible evaluation. Designing the environments, feedback loops, and control systems that close this gap is now recognized as a primary engineering challenge for agentic AI (OpenAI 2026; Böckeler 2026).

Then introduce the control plane as the answer.

### 9.5 Recommended Paper Outline

**Section 1 — Introduction**
- Autonomous experimentation is hard to evaluate.
- Final metric is insufficient; governance metrics are needed.
- Introduce the control plane as a governed harness.
- State contributions and non-claims explicitly.

**Section 2 — Background and Related Work**
- Agentic AI evaluation (KDD AAE framing).
- Autonomous coding/research agents: AIDE (closest; see comparison table §7.3), ml-intern, multiautoresearch, deep-research.
- Experiment tracking (MLflow, W&B) — gap: no lifecycle governance.
- AutoML — gap: no auditability or failure taxonomy.
- Harness engineering literature: Trivedy, Böckeler, OpenAI, Anthropic, Hashimoto.
- Gap: none of the above provide the full governed control plane (scope enforcement + append-only ledger + failure taxonomy + memory ablation).

**Section 3 — Governed Control Plane (System Design)**
- Manager/worker separation.
- Trial lifecycle state machine.
- Editable-scope enforcement.
- Pending-trial guard.
- Append-only ledger.
- Memory modes.
- Keep/discard/failed-invalid decision authority.
- **Add: Guides/Sensors 2×2 Table** (see below).
- **Add: Failure mode motivation table** (agent failure modes → harness fixes).

**Section 4 — Evaluation Protocol**
- ResNet-trigger benchmark node + NodeSpec.
- Fixed-budget campaign protocol.
- Memory ablation design (pre-stated hypothesis).
- Stress/forced failure trial design.
- Governance metrics and their definitions.
- Failure taxonomy (Table 2).

**Section 5 — Results**
- Order: governance lifecycle demo → memory ablation → failure taxonomy → provenance completeness → task metric (secondary).
- Main campaign governance (Table 1).
- Memory ablation repeated-bad rate (Figure 2 + Table 2).
- Decision breakdown (Figure 3).
- Provenance chain (Table 4).
- Task metric trajectory (Figure 4, secondary).

**Section 6 — Discussion**
- What is proven; what is not proven.
- Why governance metrics matter beyond AUC.
- Limitations (§8 verbatim).
- Generalisation path.

**Section 7 — Conclusion**
- Restate governed harness contribution.
- Emphasize evaluation methodology.

### 9.6 Guides/Sensors 2×2 Table *(add to Section 3)*

| | **Guide (feedforward — steers before act)** | **Sensor (feedback — corrects after act)** |
|---|---|---|
| **Computational** *(deterministic, fast)* | Node spec (defines valid scope), scope validator (enforces editable paths), budget cap, editable-path whitelist | Metric parser (val_bpb→val_AUC), state machine (enforces transition legality), pending-trial guard (detects crash) |
| **Inferential** *(semantic, LLM-based)* | Manager system prompt + memory injection (steers next proposal) | Repeated-bad detector (Jaccard similarity + parameter-direction extraction flags semantically redundant proposals) |

*Taxonomy from Böckeler (2026), instantiated for autonomous ML experimentation.*

### 9.7 Failure Mode Motivation Table *(add to Section 3)*

| Agent Failure Mode | ML Experimentation Analog | Harness Fix |
|---|---|---|
| Declares victory too early | Manager stops after one good trial | Budget enforcer + fixed trial count |
| Leaves broken state | Failed patch leaves repo dirty | Pending-trial guard + git commit check |
| Marks feature done prematurely | Manager claims improvement without valid metric | `FAILED_INVALID` state + metric parser |
| Doesn't know how to run the app | Manager proposes out-of-scope edit | Node spec + editable-path whitelist |

*Failure mode pattern from Anthropic/Young (2025), mapped to ML experimentation context.*

---

## 10. Figures and Tables Required

### Table 1 — Main Campaign (optimization + governance)
Columns: node, manager, memory mode, budget, initial/best/final val_AUC, net gain, kept/discarded/invalid, acceptance rate, invalid rate, provenance completeness, artifact completeness.

### Table 2 — Failure Taxonomy *(position as a contribution)*
Columns: failure category, count, example trial ID, control-plane response, state committed (yes/no).

### Table 3 — Memory Ablation (primary result)
Columns: memory mode, budget, repeated-bad count, repeated-bad rate, kept, discarded, failed-invalid, best val_AUC, context chars, compression ratio.

### Table 4 — Provenance Chain
Rows: kept / discarded / failed_invalid. Columns: proposal present, patch present, run log present, parsed metric present, decision present, ledger entry present, completeness %.

### Table 5 — AIDE Comparison *(new, required for Related Work)*
See §7.3 above.

### Figure 1 — Architecture Diagram
Control plane as center; Manager → Proposal → Control Plane ↔ Memory/Ledger; Control Plane → Worker → Training Command → Metric Parser → Validity Check → Decision. Caption: *"Managers and workers are pluggable and cannot directly commit trial state."*

### Figure 2 — Repeated-Bad Rate by Memory Mode *(visual centerpiece)*
X-axis: `none | append_only_summary | append_only_summary_with_rationale`. Y-axis: `repeated_bad_rate`.

### Figure 3 — Decision Breakdown
Stacked bars (kept / discarded / failed_invalid) by memory mode.

### Figure 4 — Campaign Trajectory *(secondary)*
val_AUC by trial index.

---

## 11. Writing Rules

1. **Governance first, metric second.** Never open a results sentence with AUC.
2. **Do not overclaim from small gains.** "The task metric confirms real execution; optimization quality is not the primary claim."
3. **Make failures look valuable.** "Failed trials are first-class audit objects, measuring invalid-action rate, failure category, and recovery behavior."
4. **Always separate agent and control plane.** "The agent proposes; the control plane validates and decides."
5. **State non-claims explicitly.** Include the non-claims paragraph verbatim (§2.3).
6. **Pre-state hypotheses before presenting results.** Required for memory ablation.
7. **Cite harness engineering literature.** Trivedy, Böckeler, OpenAI, Anthropic, AIDE must appear in Related Work.

---

## 12. Complete Implementation Checklist

### Infra / Pre-Flight (must do before any experiment)
- [ ] Add `langgraph`, `langchain-core`, `langchain-ollama` to `pyproject.toml`
- [ ] Implement `scripts/reset_node_state.py --node resnet_trigger`
- [ ] Add no-op patch guard (`no_op_patch` failure category)
- [ ] Add seed/config logging to `TrialRecord` fields
- [ ] Add pending-trial recovery path

### Experiments (must complete before final paper writing)
- [ ] Run 1-trial smoke ablation across all 3 memory modes (verify runner integration)
- [ ] Run 5-trial main campaign (`prompt_manager` + `append_only_summary_with_rationale`)
- [ ] Run forced invalid-scope stress trial
- [ ] Run full memory ablation (5 trials/mode × 3 modes)
- [ ] If noisy: increase to 10 trials/mode and add bootstrap intervals

### Analysis / Export (must complete)
- [ ] Export `main_campaign_summary.csv`
- [ ] Export `governance_metrics.csv`
- [ ] Export `memory_ablation_summary.csv`
- [ ] Export `repeated_bad_idea_rates.csv`
- [ ] Export `accepted_discarded_invalid_counts.csv`
- [ ] Export `campaign_trajectory.csv`
- [ ] Commit all real JSONL ledgers and artifact directories

### LangChain Integration (must-have if claiming backend-agnosticism)
- [ ] Implement `LangChainProposalBackend` with structured output
- [ ] Add `--llm-backend langchain` option
- [ ] Run one real campaign with LangChain backend; verify JSONL schema identical
- [ ] Save raw LangChain prompt/response as artifacts

### Paper Writing (after experiments)
- [ ] Write Related Work: AIDE comparison table, harness engineering citations
- [ ] Write Section 3: add 2×2 guides/sensors table, failure mode motivation table
- [ ] Write Experiments: pre-state memory ablation hypothesis
- [ ] Write Results: governance first, AUC last
- [ ] Write Discussion + Limitations (include holdout gap, single-node caveat)
- [ ] Rewrite Abstract using post-results template
- [ ] Add explicit non-claims paragraph

### Should-Have (if time allows)
- [ ] Manager comparison (baseline vs. prompt)
- [ ] Repeated baseline seeds (mean ± std)
- [ ] Artifact manifest (`artifact_manifest.json`)
- [ ] Architecture diagram (Figure 1)
- [ ] Pseudocode for control-plane lifecycle
- [ ] Optional LangSmith tracing (secondary observability)

### Nice-to-Have
- [ ] Second benchmark node or toy node
- [ ] Public GitHub release tag + Zenodo DOI
- [ ] Minimal reviewer quickstart (`make reproduce-kdd-aae`)
- [ ] One-page artifact appendix

---

## 13. Suggested Repository Structure Changes

### New Scripts
```
scripts/reset_node_state.py
scripts/run_kdd_main_campaign.py
scripts/run_kdd_memory_ablation.py
scripts/run_kdd_stress_trial.py
scripts/run_kdd_ablation_smoke.py
scripts/export_kdd_tables.py
scripts/export_kdd_figures.py
scripts/check_kdd_artifact_completeness.py
```

### New Source Files
```
src/autoresearch/llm/langchain_client.py
src/autoresearch/orchestration/langgraph_runner.py  (optional, Phase 3)
```

### New Paper Files
```
paper/kdd_aae_2026/main.tex
paper/kdd_aae_2026/sections/{introduction,related_work,system,experiments,results,discussion}.tex
paper/kdd_aae_2026/tables/
paper/kdd_aae_2026/figures/
artifact_manifest.json
```

### Docs to Update *(from checklist_stage3)*
- [ ] `docs/stage_2_current_structure.md` — make real campaigns the primary path
- [ ] `docs/architecture.md` — add guides/sensors table, promoted campaign architecture
- [ ] `README.md` — add real campaign quickstart; replace legacy Stage 1 descriptions

---

## 14. Acceptance-Oriented Priority Ranking

### Highest Priority (do first)
1. Real memory ablation
2. Forced failure / stress trial
3. Multi-trial real campaign (≥5 trials)
4. Repeated-bad rate table and figure
5. AIDE comparison table in Related Work
6. Clear non-claims paragraph

### Medium Priority
7. Manager comparison
8. Repeated baseline seeds
9. Artifact manifest
10. Guides/Sensors 2×2 table in paper

### Lower Priority
11. LangGraph runner (Phase 3)
12. Second benchmark node
13. Large AUC improvement
14. Extensive optimization tuning

---

## 15. Submission Readiness Levels

| Level | Description | Target |
|-------|-------------|--------|
| 0 | Only dry-run ledgers or code-only claims | Not ready |
| 1 | One real baseline + one agent trial | **Current status** |
| 2 | 5-trial campaign + real ablation + stress trial + Tables 1–4 | Minimum acceptable |
| 3 | 10-trial campaign + 10 trials/mode + clean repeated-bad pattern + manager comparison + reproducibility package | Strong workshop paper |
| 4 | Second node + repeated seeds + public artifact package + reviewer quickstart | Very strong |

**Target Level 2. Aim for Level 3 if time allows.**

---

## 16. Immediate Next Steps (ordered)

1. Fix packaging (add deps to `pyproject.toml`).
2. Implement `reset_node_state.py`.
3. Add no-op patch guard to control plane.
4. Run 1-trial smoke ablation over all 3 memory modes.
5. Run 5-trial main campaign with `prompt_manager`.
6. Run forced invalid-scope stress trial.
7. Run memory ablation (5 trials/mode).
8. Inspect repeated-bad rates; if noisy, scale to 10/mode.
9. Export all KDD tables and figure CSVs.
10. Write paper starting from governance, not AUC.
11. Add AIDE comparison table to Related Work.
12. Add non-claims paragraph.
13. Add explicit limitations (single node, no holdout).

---

## 17. The Core Filter

For every experiment, paragraph, table, and figure, ask:

> Does this help a reviewer evaluate whether the autonomous agent loop is **bounded, auditable, failure-aware, reproducible, and behaviorally improved by governance memory**?

If yes: keep it. If it only says "AUC improved slightly": move to secondary evidence.
