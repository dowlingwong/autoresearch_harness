# Stage 2 Deliverable Summary
## Making autoresearch_harness a KDD AAE 2026 Submission

---

## 1. What This Paper Claims (and Doesn't)

**The thesis in one sentence:**

> Autonomous ML experimentation becomes scientifically credible only when the agent loop is governed by an explicit control plane — one that owns trial lifecycle, enforces bounded execution, records decisions append-only, and reports auditable governance metrics.

**Title (use one of these):**
- *Auditable Autonomous Experimentation with Bounded Execution and Explicit Keep/Discard Control*
- *A Governed Control Plane for Autonomous ML Experimentation: Lifecycle, Budget, and Audit Metrics*

**Claims the paper makes:**
1. Bounded execution (editable-scope enforcement + pending-trial guard) prevents the most common class of silent experiment corruption in autonomous agent loops.
2. Explicit keep/discard decisions owned by the framework — not the agent — produce auditable, reproducible records.
3. Append-only memory with rationale reduces repeated-bad proposals under fixed budget compared to no-memory or summary-only modes.
4. Governed experimentation produces measurable, provenance-linked improvements on a real ML benchmark.

**Claims the paper explicitly does NOT make:**
- This is not a general autonomous scientist.
- This is not proof of scientific discovery or novel ML insight.
- This is not a universal hyperparameter optimizer.
- This is not dependent on any specific coding agent, LLM, or orchestration framework.
- The ResNet-trigger benchmark is not claimed to be representative of all ML optimization tasks.

The paper's audience is the evaluation/governance track of KDD AAE, not the ML systems optimization track. Frame it as **an evaluation methodology contribution**, not as a search algorithm.

---

## 2. Why KDD AAE Is the Right Venue

The KDD 2026 Agentic AI Evaluation workshop explicitly prizes:

| Workshop priority | How this project addresses it |
|---|---|
| Agentic AI evaluation frameworks | The governed control plane is itself an evaluation harness |
| Multi-step planning and tool use | Manager → proposal → worker → metric → decision chain |
| Monitoring under evolving conditions | Append-only ledger, memory ablation across a campaign |
| Standardized metrics and logging protocols | Typed `TrialRecord`, JSONL ledger, deterministic CSV exports |
| Lifecycle and governance frameworks | Formal state machine, pending-trial guard, scope enforcement |
| Auditability and liability attribution | Full provenance chain: proposal → patch → run → metric → decision |
| Production-style monitoring | Artifact capture completeness, command failure rate, parsing failure rate |

The positioning is: *"We built the governance layer that makes autonomous experimentation safe enough to evaluate and report."* The LLM and the coding worker are pluggable backends, not the contribution.

---

## 3. Experiment Matrix: What to Run

Run three experiments in this order. Do not proceed to Experiment 2 until Experiment 1 produces real ledgers.

### Experiment 1 — Governed Main Campaign (Required)

**Purpose:** Demonstrate that the control plane works end-to-end on a real ML benchmark and produces measurable improvement with complete provenance.

**Setup:**
- Node: `resnet_trigger`
- Manager: `prompt_manager` (memory-aware, no LLM required)
- Memory mode: `append_only_summary_with_rationale`
- Budget: 5–10 real trials (3 at minimum)
- Worker: `ClawWorker` (claw-code + Ollama backend)

**Command:**
```bash
RESNET_TRIGGER_FAST_SEARCH=1 \
RESNET_TRIGGER_FAST_N_SIGNAL=1000 \
RESNET_TRIGGER_FAST_N_NOISE=1000 \
RESNET_TRIGGER_FAST_TRACE_LEN=4096 \
RESNET_TRIGGER_FAST_BATCH_SIZE=64 \
RESNET_TRIGGER_FAST_EPOCHS=5 \
RESNET_TRIGGER_FAST_SKIP_TEST=1 \
RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 \
RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 \
RESNET_TRIGGER_DEVICE=cpu \
python3 scripts/run_campaign.py \
    --node resnet_trigger \
    --campaign-id main_campaign \
    --budget 10 \
    --manager prompt_manager \
    --memory-mode append_only_summary_with_rationale \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b \
    --host http://localhost:11434
```

**Must also include one stress trial** — a trial where the worker intentionally edits a frozen file or produces no metric. This proves the control plane catches failures correctly. Run a separate 1-trial campaign:
```bash
# ... same env vars ...
python3 scripts/run_campaign.py \
    --node resnet_trigger \
    --campaign-id stress_scope \
    --budget 1 \
    --manager baseline_manager \
    --memory-mode none \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b \
    --host http://localhost:11434
```
(Then manually verify the resulting `TrialRecord` shows `decision=failed_invalid` and `failure_category=invalid_edit_scope`.)

**What to report from this experiment:**
- Initial, best, and final accepted val_auc
- Net gain and gain per budget unit
- Acceptance rate, invalid rate, command failure rate
- Complete provenance rate
- Total wall-clock time
- The patch diff for the best accepted trial

---

### Experiment 2 — Memory/Governance Ablation (Required, Core Claim)

**Purpose:** Show that richer memory reduces repeated-bad proposals. This is the primary ablation and the strongest paper claim.

**Setup:**
- All three memory modes under equal budget
- Same node, same worker, same manager (`prompt_manager`)
- Budget: 5 trials per mode (15 total)

**Command:**
```bash
# ... same env vars as above ...
python3 scripts/run_memory_ablation.py \
    --node resnet_trigger \
    --budget 5 \
    --execute-real-campaigns \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b \
    --host http://localhost:11434
```

This runs three campaigns automatically:
- `resnet_trigger_none_ablation` (memory_mode=none)
- `resnet_trigger_append_only_summary_ablation`
- `resnet_trigger_append_only_summary_with_rationale_ablation`

**What to report from this experiment:**
- Repeated-bad proposal count and rate per mode
- Acceptance rate per mode
- Best metric per mode
- Context length fed to manager per mode (compression ratio)
- Recovery-after-failure behavior (did the manager avoid repeating the failed pattern?)

**Expected result pattern:** The `none` mode should show the highest repeated-bad rate because the manager has no memory of prior failures. The `with_rationale` mode should show the lowest. If results are noisy across 5 trials, increase budget to 10.

---

### Experiment 3 — Manager Comparison (Optional, Strengthens Paper)

**Purpose:** Show the control plane is manager-agnostic — different managers produce comparable governance behavior under the same constraints.

**Command:**
```bash
python3 scripts/run_manager_comparison.py \
    --node resnet_trigger \
    --budget 5 \
    --memory-mode append_only_summary_with_rationale \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b \
    --host http://localhost:11434
```

**What to report:** Side-by-side table of `baseline_manager` vs `prompt_manager` on governance metrics (not optimization metrics — the claim is that governance holds regardless of manager).

---

## 4. Rubrics: How to Organize the Paper Metrics

The paper should foreground **governance metrics** as first-class results, with optimization metrics as secondary evidence. This is counterintuitive for an ML paper but correct for a KDD AAE evaluation paper.

### Table 1 — Main Campaign Results (Experiment 1)

| Metric | Value |
|---|---|
| Node | resnet_trigger |
| Manager | prompt_manager |
| Memory mode | append_only_summary_with_rationale |
| Budget (trials) | N |
| Initial val_auc | X.XXXXXX |
| Best val_auc | X.XXXXXX |
| Net gain | +X.XXXXXX |
| Gain per budget unit | X.XXXXXX |
| Total wall-clock (s) | XXXX |
| Kept / discarded / invalid | K / D / I |
| Acceptance rate | XX% |
| Invalid rate | XX% |
| Complete provenance rate | 100% (target) |
| Editable-scope violations | 0 (target) |
| Command failure rate | XX% |
| Metric-parse failure rate | XX% |
| Artifact capture completeness | 100% (target) |

### Table 2 — Memory Ablation (Experiment 2, Primary Result)

| Memory mode | Repeated-bad rate | Acceptance rate | Best val_auc | Context chars | Compression ratio |
|---|---|---|---|---|---|
| none | XX% | XX% | X.XXXX | ~92 | ~0.003 |
| append_only_summary | XX% | XX% | X.XXXX | ~3,800 | ~0.10 |
| append_only_summary_with_rationale | XX% | XX% | X.XXXX | ~7,100 | ~0.20 |

**Key interpretation:** If `with_rationale` shows lower repeated-bad rate, that is the evidence for the paper's memory claim. The optimization improvement is secondary.

### Table 3 — Failure Taxonomy (Required)

Every `failed_invalid` trial must be categorized. Report counts:

| Failure category | Count | % of total trials |
|---|---|---|
| `invalid_edit_scope` | N | XX% |
| `syntax_error` | N | XX% |
| `runtime_error` / command failure | N | XX% |
| `metric_missing` | N | XX% |
| `degraded_metric` (discarded) | N | XX% |

This table proves the control plane correctly classifies failures — a key governance claim.

### Table 4 — Provenance Completeness (Required)

For every kept trial, the paper must show the provenance chain is complete:

| Provenance field | Present in all kept trials |
|---|---|
| `proposal_id` | ✓ |
| `patch_id` | ✓ |
| `run_id` | ✓ |
| `metric_id` | ✓ |
| `decision_id` | ✓ |
| `patch_ref` (diff file) | ✓ |
| `raw_log_ref` (run log) | ✓ |
| `parsed_metrics_ref` | ✓ |

### Figure 1 — Campaign Trajectory

Plot `val_auc` vs trial index, with kept trials in green and discarded/invalid in red. One line per memory mode (Experiment 2). Generated from `paper/figures/campaign_trajectory.csv`.

### Figure 2 — Decision Breakdown

Stacked bar chart: kept / discarded / failed_invalid per memory mode. Generated from `paper/figures/accepted_discarded_invalid_counts.csv`.

### Figure 3 — Repeated-Bad Rate Comparison

Bar chart of repeated-bad proposal rate across three memory modes. This is the visual proof of the memory claim. Generated from `paper/figures/repeated_bad_idea_rates.csv`.

---

## 5. Paper Sections and What Goes in Each

### Abstract (≤150 words)
Autonomous ML experimentation lacks governance: agents edit files freely, decisions are informal, and audits are impossible. We present a governed control plane for autonomous experimentation — a six-layer framework with explicit trial lifecycle, editable-scope enforcement, append-only memory, and deterministic provenance. We evaluate it on the ResNet-trigger near-threshold detector benchmark. Over N real trials with budget B, the framework achieves val_auc X.XXXX (net gain +X.XXXX), acceptance rate AA%, and 100% provenance completeness. A memory ablation over three modes shows that rationale-augmented memory reduces repeated-bad proposals from XX% to XX%. The framework is manager-agnostic and worker-agnostic; all governance properties hold regardless of the underlying coding agent or LLM.

### 1. Introduction
- The problem: autonomous agent loops have no native governance
- The gap: existing systems (AI Scientist, etc.) optimize but do not audit
- The contribution: a governed control plane that makes autonomous experimentation inspectable
- One paragraph on KDD AAE fit: this is a framework for *evaluating* agentic optimization, not for doing it better

### 2. Related Work
- AI Scientist, AlphaCode, Eureka — autonomous experimentation, no governance layer
- LangGraph, CrewAI — orchestration, not evaluation
- MLflow, W&B — experiment tracking, not agent governance
- Key distinction: those systems track what happened; ours enforces what can happen

### 3. System Architecture
Walk through the six layers with a diagram. One paragraph per layer. Emphasize:
- Control plane owns state transitions (managers and workers cannot commit trial state)
- Pending-trial guard prevents orphaned state
- Editable-scope allowlist enforced before execution
- Append-only ledger: no record can be overwritten

### 4. Experiment Design
- Node spec (ResNet-trigger near-threshold detector)
- Fixed-budget protocol
- Three memory modes and how they differ
- Failure taxonomy (what failure categories exist and how they're classified)
- Metric: val_auc (converted from val_bpb, higher is better)
- Acceptance rule: `candidate_metric > current_best_metric`

### 5. Results
5.1 Main Campaign (Table 1 + Figure 1)
5.2 Memory Ablation (Table 2 + Figures 2–3) — **lead with this**
5.3 Failure Taxonomy (Table 3)
5.4 Provenance Completeness (Table 4)
5.5 Manager Comparison (optional, if Experiment 3 was run)

### 6. Discussion
- What the results prove and don't prove
- Why governance metrics matter more than optimization gain for this paper
- Limitations: single benchmark node, CPU smoke configuration, one model (Qwen)
- Future: additional nodes, multi-worker campaigns, cloud deployment

### 7. Conclusion
Restate thesis. The control plane is the contribution; the LLM is a pluggable component. Anyone building autonomous experimentation systems should make governance explicit.

---

## 6. How to Generate All Paper Artifacts

Run these commands after your real campaigns complete:

```bash
# Export all paper tables from the main campaign
python3 scripts/export_paper_tables.py --campaign-id main_campaign

# Export figure data CSVs from the main campaign
python3 scripts/export_paper_figures.py --campaign-id main_campaign

# Write a human-readable markdown report
python3 -c "
from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.reporting.write_report import write_campaign_report
summary = load_campaign_summary('experiments/ledgers/main_campaign_trials.jsonl')
path = write_campaign_report(summary, 'paper/notes')
print('Report:', path)
"

# Summarize the memory ablation
python3 scripts/summarize_campaign.py \
    --campaign-id resnet_trigger_none_ablation \
    --records experiments/ledgers/resnet_trigger_none_ablation_trials.jsonl

python3 scripts/summarize_campaign.py \
    --campaign-id resnet_trigger_append_only_summary_ablation \
    --records experiments/ledgers/resnet_trigger_append_only_summary_ablation_trials.jsonl

python3 scripts/summarize_campaign.py \
    --campaign-id resnet_trigger_append_only_summary_with_rationale_ablation \
    --records experiments/ledgers/resnet_trigger_append_only_summary_with_rationale_ablation_trials.jsonl
```

Generated files (commit all of these):
```
paper/tables/main_campaign_summary.csv
paper/tables/governance_metrics.csv
paper/tables/memory_ablation_summary.csv
paper/figures/campaign_trajectory.csv
paper/figures/accepted_discarded_invalid_counts.csv
paper/figures/repeated_bad_idea_rates.csv
paper/figures/gain_per_budget_unit.csv
paper/notes/main_campaign_report.md
experiments/ledgers/*.jsonl   ← all real ledgers
```

---

## 7. Reproducibility Package

The paper must include or reference these assets for reproducibility:

| Asset | Location |
|---|---|
| Node contract (YAML) | `configs/nodes/resnet_trigger.yaml` |
| Campaign config | `configs/campaigns/resnet_trigger_smoke.json` |
| All trial JSONL ledgers | `experiments/ledgers/` |
| Generated artifacts | `experiments/artifacts/` |
| Paper tables | `paper/tables/` |
| Figure CSVs | `paper/figures/` |
| Run commands | This file, Section 6 |
| Dependency lockfile | `pyproject.toml` + `uv.lock` or `requirements.txt` |
| Install instructions | `README.md` Quickstart section |

To generate a `requirements.txt` for reviewers:
```bash
uv pip freeze > requirements.txt
```

---

## 8. Submission Readiness Checklist

Work through this top to bottom. Do not start writing the paper until at least Experiments 1 and 2 are complete.

### Infrastructure (done)
- [x] Six-layer framework implemented (`src/autoresearch/`)
- [x] `run_real_campaign()` with pending-trial guard
- [x] Append-only JSONL ledger
- [x] Editable-scope enforcement
- [x] Three memory modes
- [x] Repeated-bad detection
- [x] Paper metric exporters
- [x] Test suite (48+ tests, all passing)
- [x] `pyproject.toml` with declared dependencies

### Experiments (to complete)
- [ ] Experiment 1: real main campaign, ≥5 trials, `prompt_manager`
- [ ] Experiment 1: one stress trial demonstrating `failed_invalid` governance
- [ ] Experiment 2: real memory ablation, all three modes, equal budget
- [ ] Experiment 3 (optional): manager comparison dry-run at minimum
- [ ] All JSONL ledgers committed to repo

### Paper artifacts (to complete)
- [ ] `paper/tables/main_campaign_summary.csv` from real ledger
- [ ] `paper/tables/governance_metrics.csv` from real ledger
- [ ] `paper/tables/memory_ablation_summary.csv` from real ablation
- [ ] All figure CSVs generated and committed
- [ ] Markdown campaign report generated (`write_campaign_report`)

### Paper writing (to complete)
- [ ] Abstract drafted (≤150 words, lead with governance)
- [ ] Architecture diagram (can be a simple text diagram or hand-drawn)
- [ ] Table 1: main campaign governance + optimization metrics
- [ ] Table 2: memory ablation — repeated-bad rate is the headline number
- [ ] Table 3: failure taxonomy from real runs
- [ ] Table 4: provenance completeness
- [ ] Figures 1–3: trajectory, decision breakdown, repeated-bad rate
- [ ] Explicit non-claims paragraph in Introduction or Discussion

### Submission judgment
- **Not ready:** ledgers are all dry-run; real memory ablation not run
- **Borderline:** one real campaign, no ablation
- **Acceptable workshop paper:** Experiments 1 + 2 complete with real runs, Tables 1–4 present
- **Strong submission:** Experiments 1 + 2 + 3, real ablation shows memory effect, failure taxonomy included
- **Strongest:** second benchmark node added (not required but differentiating)

---

## 9. The One Sentence to Keep in Mind

Every section of the paper should be answerable with this test:

> "Does this section help a reader understand, evaluate, or reproduce the *governance* behavior of the framework — not just whether it improved a metric?"

If a section is only about the metric improvement, it is in the wrong venue. If it is about how the framework made that improvement auditable, reproducible, and bounded, it is in the right place.
