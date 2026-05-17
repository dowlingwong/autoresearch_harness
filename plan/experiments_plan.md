# Experiments Plan
_Last updated: 2026-05-17 (added rounds breakdown + 50% budget expansion)_

This document is the single source of truth for all planned and completed
experiments in the autoresearch harness. It maps every run to the specific
paper weakness it addresses (P0-2, P0-3, etc.), gives current status, and
specifies exact commands and budget choices.

---

## 1. Node Inventory and Positioning

Every node in the harness has a distinct role in the paper's argument.
Running the same harness across qualitatively different nodes is the
portability claim. Each node below answers a different "what does the
harness prove here?" question.

| # | Node | Domain | Worker | Runtime/trial | Metric | Direction |
|---|---|---|---|---|---|---|
| 1 | `resnet_trigger` | Waveform classification (ResNet) | ClawWorker + Ollama | 5–10 min fast-search | val_auc | maximize |
| 2 | `autoresearch_macos` | GPT LM training (nanochat) | ClawWorker + Ollama | 5 min fixed | val_bpb | **minimize** |
| 3 | `lr_synthetic` | NumPy logistic regression | LocalWorker | seconds | val_score | maximize |
| 4 | `mlp_synthetic` | NumPy one-hidden-layer MLP | LocalWorker | ~1 sec | val_score | maximize |
| 5 | `openml_credit_g` | Tabular classification (OpenML 31) | LocalWorker | 2–10 sec | val_auc | maximize |
| 6 | `openml_bank_marketing` | Tabular classification (OpenML 1461) | LocalWorker | 3–15 sec | val_auc | maximize |
| 7 | `mlagentbench_vectorization` | NumPy convolution (MLAgentBench) | LocalWorker | 0.1–3 sec | speed_score | maximize |

### Manager and worker rounds per trial

Every trial has exactly one manager round and one worker round. The table
below shows what those rounds cost in terms of API calls and subprocesses.

| Node | Manager LLM calls | Worker type | Ollama calls/trial | Subprocess/trial | Wall time/trial |
|---|---|---|---|---|---|
| `resnet_trigger` | 1 (DeepSeek, ~2–5 s) | ClawWorker | 1–2 (qwen, code edit + optional retry) | 1 (`uv run train.py`) | 5–10 min (fast-search) |
| `autoresearch_macos` | 1 (DeepSeek, ~2–5 s) | ClawWorker | 1–2 (qwen, code edit + optional retry) | 1 (`uv run train.py > run.log 2>&1`) | ~5 min (fixed budget) |
| `lr_synthetic` | 1 (DeepSeek, ~2–5 s) | LocalWorker | 0 | 1 (NumPy eval in-process) | < 1 s |
| `mlp_synthetic` | 1 (DeepSeek, ~2–5 s) | LocalWorker | 0 | 1 (NumPy eval in-process) | < 1 s |
| `openml_credit_g` | 1 (DeepSeek, ~2–5 s) | LocalWorker | 0 | 1 (sklearn fit) | 2–10 s |
| `openml_bank_marketing` | 1 (DeepSeek, ~2–5 s) | LocalWorker | 0 | 1 (sklearn fit) | 3–15 s |
| `mlagentbench_vectorization` | 1 (DeepSeek, ~2–5 s) | LocalWorker | 0 | 1 (NumPy eval) | 0.1–3 s |

**Key invariants:**
- The LangGraph manager executes a straight graph: `prepare_context → generate_proposal → validate_proposal → END`. There is no retry loop inside the manager. One trial = exactly one DeepSeek API call.
- LocalWorker applies the manager's text directive (e.g. "Change LR from 0.01 to 0.05") directly — no LLM involvement. One trial = one `run_command` subprocess.
- ClawWorker (ResNet only) calls Ollama to translate the DeepSeek proposal into a code patch. `retry_limit=1` means up to 2 Ollama calls if the first edit fails to parse. In practice ≈ 1.1 Ollama calls/trial on average.
- The control plane's keep/discard/failed_invalid decision is purely algorithmic (no LLM call).

**Why it is always 1 manager round × 1 worker round — not e.g. 3 manager × 5 worker:**

The harness is designed around a strict invariant: **one proposal → one execution → one atomic `TrialRecord`**. Breaking this 1:1 ratio would cause three problems:

1. *Ledger integrity.* Each `TrialRecord` carries a `provenance` chain: `proposal_id → patch_id → run_id → metric_id → decision_id`. If one manager call produced 5 candidate patches and the worker ran all 5, you'd need to decide which execution "is" the trial — or write 5 records per proposal. The append-only ledger would lose its clean one-proposal-to-one-decision mapping, breaking the audit trail the paper is built on.

2. *Memory feedback.* The manager's next proposal is conditioned on every prior trial result via `build_memory_context()`. Running 5 workers from one stale proposal wastes the learning signal: 4 of those 5 executions are informed only by trial N-1, not by trials N-1 through N+3. The sequential loop is the mechanism that makes the memory ablation meaningful.

3. *Decision ownership.* The control plane owns keep/discard, not the manager. Parallel candidates would require the control plane to adjudicate between simultaneous results — a fundamentally different state machine than the one the paper formalises.

Systems like Optuna (parallel workers), PBT (population-based training), and ASHA do run N workers per decision round — they are optimizing for sample efficiency under parallelism, not for auditability. This harness optimises for the latter; the sequential 1:1 structure is a feature, not a limitation.

**Per-campaign API call totals** (budget × 1 DeepSeek call, rounded):

| Campaign group | Trials | DeepSeek calls | Ollama calls | Notes |
|---|---|---|---|---|
| `deepseek_lr_*` (9 arms, 3 seeds) | 90 ✅ | 90 | 0 | Done |
| `deepseek_lr_*` (6 arms, s4+s5) | **+60** | 60 | 0 | Fast: seconds |
| `deepseek_mlp_summary_s1` | 10 ✅ | 10 | 0 | Done |
| `deepseek_mlp_none_s1` + `mlp_rationale_s1` | **+20** | 20 | 0 | Fast: seconds |
| `deepseek_openml_*` (6 campaigns, 3 seeds) | 120 ✅ | 120 | 0 | Done |
| `deepseek_openml_*_s4` (2 nodes) | **+40** | 40 | 0 | ~20 min |
| `deepseek_mlagentbench_s1` | **10** ❌ | 10 | 0 | ~2 min |
| `deepseek_mlagentbench_s2` | **+10** | 10 | 0 | ~2 min |
| `deepseek_resnet_*` (9 arms, budget 15) | **135** ⚠️ rerun | 135 | ~150 | Overnight, GPU |
| `deepseek_autoresearch_*` (3 seeds) | **60** ❌ | 60 | ~66 | ~5 hr |
| `deepseek_autoresearch_s4` | **+20** | 20 | ~22 | +~2 hr |
| **Grand total** | **575** | **575** | **~238** | **≈ 51% more than original 380** |

---

### Node positioning: what each one proves

**`resnet_trigger` — Flagship scientific node**
The primary case study. Real GPU training, private H5 waveform data, full
ClawWorker code-edit loop. The only node where the worker modifies real
Python source code (not config.yaml). Demonstrates governance under the
conditions the paper is targeting: autonomous ML experimentation on a real
scientific task. val_auc is secondary evidence — governance metrics
(lifecycle completeness, failure taxonomy, RBR) are the result.
- Proves: full three-way lifecycle (kept / discarded / failed_invalid) under
  real training; failure taxonomy in the wild; memory-ablation RBR diagnostic;
  DeepSeek vs. Qwen manager comparison (P0-3 core experiment).

**`autoresearch_macos` — Origin story + domain transfer node**
Karpathy's nanochat GPT training loop (autoresearch-macos fork). Fixed
5-minute wall-clock training budget. val_bpb (bits per byte, lower is better)
is vocab-size-independent and meaningful across architecture changes. This is
the only minimize-direction node and the only LM training node. The editable
scope spans architecture (DEPTH, ASPECT_RATIO, HEAD_DIM) and optimization
(MATRIX_LR, WARMDOWN_RATIO, etc.) — richer than config-only nodes.
The project conceptually originated from this node.
- Proves: governance transfers to LM training domain; minimize-direction
  decision logic is correct; harness is not classification-specific; strong
  narrative anchor ("governing the node that inspired the project").

**`lr_synthetic` — Ablation workhorse**
Pure NumPy logistic regression, seconds per trial, 4 editable hyperparameters.
Cheap enough to run 5 seeds × 3 arms × 30 trials overnight. The primary node
for the memory ablation (none / append_only_summary / append_only_summary_with_rationale).
Also the best place to demonstrate that RBR is a diagnostic tool, not just a
point estimate — running it across two manager tiers (Qwen vs. DeepSeek) and
multiple seeds is the main evidence for P0-3 on fast nodes.
- Proves: memory mode effect on repeated-bad rate; whether the non-monotonic
  summary < none ≈ rationale finding holds under model substitution.

**`mlp_synthetic` — Manager-tier discrimination check**
One-hidden-layer NumPy MLP. Slightly harder search space than lr_synthetic
(nonlinear, one extra hyperparameter HIDDEN_DIM). Used specifically for
P0-4 discrimination: do governance metrics (kept rate, RBR) vary between
manager tiers, or are they constant? If constant, the metrics are uninformative.
- Proves: governance metrics are discriminative across manager tiers;
  provides a second local node for the discrimination table.

**`openml_credit_g` — Public reproducibility, discard-dominant pattern**
OpenML dataset 31, sklearn pipeline, config.yaml hyperparameter edits.
20-trial history shows 1 kept, 19 discarded, 0 failed-invalid — almost all
valid proposals are non-improving. This cleanly demonstrates the discard path
and shows the control plane does not conflate "valid but worse" with "invalid".
- Proves: governance transfers to public benchmark; discard classification
  is operational; public reproducibility of governance evidence.

**`openml_bank_marketing` — Public reproducibility, mixed lifecycle**
OpenML dataset 1461. Mixed lifecycle: some kept (AUC improves), some
failed-invalid (out-of-range config proposals). Shows failure taxonomy
(invalid_config) in action on a public node and across seeds.
- Proves: same as credit_g + failure taxonomy diversity on public data;
  the harness catches invalid config proposals pre-execution.

**`mlagentbench_vectorization` — External benchmark compatibility probe**
MLAgentBench vectorization task adapted as a NodeSpec. The control-plane
lifecycle logic was unchanged; only a registry adapter and metric parser were
added. Treated as an external compatibility probe, not a holdout performance
node. 30-trial campaign exists.
- Proves: governance wraps an external benchmark without changing the control
  plane (P1-2 done); positions the paper against MLAgentBench-style systems.

---

## 2. What Each Experiment Phase Proves for the Paper

| Phase | Revision item | Core claim tested |
|---|---|---|
| Phase A — Smoke tests | Infrastructure | API + worker pipeline functional |
| Phase B — DeepSeek fast ablations | P0-3 (model confound) | Governance behavior is not an artifact of qwen2.5-coder:7b |
| Phase C — Multi-seed reruns (fast nodes) | P0-2 (CIs) | RBR ordering claim has CIs tighter than seed noise |
| Phase D — ResNet DeepSeek ablation | P0-3 (flagship) | DeepSeek manager on scientific node, same harness |
| Phase E — autoresearch_macos campaign | Portability | Governance transfers to LM training domain |
| Phase F — Large budget reruns | P0-2 (deferred) | Seed-level CIs replace trial-level CIs in all tables |
| Phase G — Bootstrap CI export | P0-2 analysis | Final CI tables for paper |
| Phase H — P0-4 validation | Metric validity | Governance metrics are discriminative and reliable |

---

## 3. Experiment Matrix

### 3A — DeepSeek Manager (P0-3 core): fast nodes

Goal: Show that governance metrics (RBR, kept rate, failure taxonomy) are
not artifacts of the small local qwen2.5-coder:7b model. Run the same
memory ablation with DeepSeek-V4-Flash as manager.

API involved: DeepSeek (manager proposals)
Worker: LocalWorker (no GPU needed)
Prereq: `export DEEPSEEK_API_KEY=... && export DEEPSEEK_THINKING=disabled`

| Campaign ID pattern | Node | Arms | Seeds | Budget/arm | Trials total | Status |
|---|---|---|---|---|---|---|
| `deepseek_smoke_lr` | lr_synthetic | smoke | 1 | 1 | 1 | ✅ Done |
| `deepseek_lr_{arm}_s{1-3}` | lr_synthetic | none / summary / rationale | 3 | 10 | 90 | ✅ Done |
| `deepseek_lr_{arm}_s{4-5}` ← **new** | lr_synthetic | none / summary / rationale | 3 | 10 | **+60** | ❌ Not started |
| `deepseek_mlp_summary_s1` | mlp_synthetic | summary | 1 | 10 | 10 | ✅ Done |
| `deepseek_mlp_none_s1` ← **new** | mlp_synthetic | none | 1 | 10 | **+10** | ❌ Not started |
| `deepseek_mlp_rationale_s1` ← **new** | mlp_synthetic | rationale | 1 | 10 | **+10** | ❌ Not started |
| `deepseek_openml_cg_s{1-3}` | openml_credit_g | summary | 3 | 20 | 60 | ✅ Done |
| `deepseek_openml_bm_s{1-3}` | openml_bank_marketing | summary | 3 | 20 | 60 | ✅ Done |
| `deepseek_openml_cg_s4` ← **new** | openml_credit_g | summary | 1 | 20 | **+20** | ❌ Not started |
| `deepseek_openml_bm_s4` ← **new** | openml_bank_marketing | summary | 1 | 20 | **+20** | ❌ Not started |
| `deepseek_mlagentbench_s1` | mlagentbench_vectorization | summary | 1 | 10 | 10 | ❌ Not started |
| `deepseek_mlagentbench_s2` ← **new** | mlagentbench_vectorization | summary | 1 | 10 | **+10** | ❌ Not started |

**Remaining commands to complete Phase B/Step 3:**
```bash
# openml_credit_g seed 3 (reset + rerun; existing 11-trial partial ledger will be cleared)
uv run python3 scripts/run_openml_tabular_campaign.py \
  --node openml_credit_g --campaign-id deepseek_openml_cg_s3 \
  --budget 20 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash --temperature 0.2

# openml_bank_marketing seed 3
uv run python3 scripts/run_openml_tabular_campaign.py \
  --node openml_bank_marketing --campaign-id deepseek_openml_bm_s3 \
  --budget 20 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash --temperature 0.2

# mlagentbench_vectorization — fills the last gap in full-node DeepSeek coverage
# (~2 min, 10 API calls; no GPU needed)
uv run python3 scripts/run_local_node_campaign.py \
  --node mlagentbench_vectorization \
  --campaign-id deepseek_mlagentbench_s1 \
  --budget 10 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash --temperature 0.2
```

---

### 3B — DeepSeek Manager (P0-3 core): ResNet scientific node

Goal: The central P0-3 experiment. DeepSeek proposes, Qwen worker executes.
Show that governance behavior on the flagship scientific node survives
frontier model substitution.

API involved: DeepSeek (manager)
Worker: ClawWorker + Ollama qwen2.5-coder:7b (code execution)
Prereq: Ollama running locally with qwen2.5-coder:7b pulled, GPU or fast-search env vars set.

| Campaign ID pattern | Node | Arms | Seeds | Budget/arm | Trials total | Est. time | Status |
|---|---|---|---|---|---|---|---|
| `deepseek_resnet_smoke` | resnet_trigger | summary | 1 | 1 | 1 | ~10 min | ❌ Not started |
| `deepseek_resnet_{arm}_s{1-3}` | resnet_trigger | none / summary / rationale | 3 | **15** | **135** | ~12–18 hr | ⚠️ Rerun needed (sklearn fix; budget raised to 15) |

**Commands:**
```bash
# 4a. Smoke first (fast-search mode, CPU)
RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_EPOCHS=3 RESNET_TRIGGER_DEVICE=cpu \
uv run python3 scripts/run_kdd_memory_ablation.py \
  --node resnet_trigger --memory-mode append_only_summary \
  --campaign-id deepseek_resnet_smoke \
  --budget 1 --manager langgraph_manager \
  --model deepseek/deepseek-v4-flash \
  --worker-model qwen2.5-coder:7b --temperature 0.2 --no-export

# 4b. Full ablation (leave overnight, GPU or fast-search mode)
for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s1 s2 s3; do
    uv run python3 scripts/run_kdd_memory_ablation.py \
      --node resnet_trigger --memory-mode ${ARM} \
      --campaign-id deepseek_resnet_${ARM}_${SEED} \
      --node-root nodes/ResNet_trigger --budget 10 \
      --manager langgraph_manager \
      --model deepseek/deepseek-v4-flash \
      --worker-model qwen2.5-coder:7b --temperature 0.2
  done
done
```

---

### 3C — autoresearch_macos: Karpathy node governance

Goal: Demonstrate governance transfers to LM training domain. Different task,
different metric direction (minimize val_bpb), richer editable scope.
One-time prereq: `cd nodes/autoresearch-macos && uv sync && uv run prepare.py`

API involved: DeepSeek (manager) + Ollama qwen2.5-coder:7b (worker code edits)
Worker: ClawWorker (Ollama generates patch for train.py, same as ResNet)
Runtime: Fixed 5 min/trial → budget 20 = ~2 hours

| Campaign ID | Node | Seeds | Budget | Trials | Est. time | Status |
|---|---|---|---|---|---|---|
| `autoresearch_smoke_1` | autoresearch_macos | 1 | 1 | 1 | 5 min | 🔄 In setup |
| `deepseek_autoresearch_s{1-3}` | autoresearch_macos | 3 | 20 | 60 | ~5 hr | ❌ Not started |
| `deepseek_autoresearch_s4` ← **new** | autoresearch_macos | 1 | 20 | **+20** | ~2 hr | ❌ Not started |

**Commands:**
```bash
# Smoke (1 trial, fast — verifies ClawWorker + Ollama + metric parser work)
RESNET_TRIGGER_FAST_SEARCH=1 \
uv run python3 scripts/run_kdd_memory_ablation.py \
  --node autoresearch_macos \
  --campaign-id autoresearch_smoke_1 \
  --node-root nodes/autoresearch-macos \
  --budget 1 --manager baseline_manager \
  --memory-mode append_only_summary \
  --worker-model qwen2.5-coder:7b --no-export

# Full 3-seed campaign with DeepSeek manager + Ollama worker
for SEED in s1 s2 s3; do
  uv run python3 scripts/run_kdd_memory_ablation.py \
    --node autoresearch_macos \
    --campaign-id deepseek_autoresearch_${SEED} \
    --node-root nodes/autoresearch-macos \
    --budget 20 --manager langgraph_manager \
    --memory-mode append_only_summary \
    --model deepseek/deepseek-v4-flash \
    --worker-model qwen2.5-coder:7b --temperature 0.2
done

# Expanded: seed 4
uv run python3 scripts/run_kdd_memory_ablation.py \
  --node autoresearch_macos \
  --campaign-id deepseek_autoresearch_s4 \
  --node-root nodes/autoresearch-macos \
  --budget 20 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash \
  --worker-model qwen2.5-coder:7b --temperature 0.2
```

---

### 3D — Multi-seed large-budget reruns (P0-2 deferred)

Goal: Replace trial-level CIs with seed-level CIs in all result tables.
Extends existing 3-seed budget-10 runs to 5 seeds at larger budgets.
Run ONLY after Phase B/C/D results are validated.

**Budget rationale:** The revision plan asks for budget 50 (ResNet) and
budget 30 (ablation). For a workshop paper, budget 20 with 5 seeds gives
seed-level CIs adequate to show ordering claims. Full budget 50 on ResNet
requires ~40+ GPU hours and is deferred to camera-ready or a follow-up.

| Node | Arms | Seeds | Budget/arm | Total trials | Est. time | Priority |
|---|---|---|---|---|---|---|
| `lr_synthetic` | 3 | 5 | 30 | 450 | ~3 hr (fast) | Medium |
| `mlp_synthetic` | 1 | 3 | 30 | 90 | ~15 min | Low |
| `openml_credit_g` | 1 | 5 | 30 | 150 | ~2 hr | Medium |
| `openml_bank_marketing` | 1 | 5 | 30 | 150 | ~3 hr | Medium |
| `resnet_trigger` | 3 | 5 | 20 | 300 | ~25 hr | High (overnight) |
| `autoresearch_macos` | 1 | 5 | 20 | 100 | ~8 hr | Medium |

Campaign ID convention: `ds_lr_{arm}_s{4,5}` for extra seeds, or
`ds_lr_{arm}_b30_s{1-5}` for full reruns at budget 30.

**Decision gate:** Run Phase D only if Phase B–C results show CIs are
too wide to support ordering claims. If 3-seed results already give
non-overlapping CIs for the main claims, skip to Phase E (analysis).

---

## 4. Execution Order and Dependencies

```
[Phase A] Smoke tests
    → all pass?
        ↓ YES
[Phase B] DeepSeek fast ablations (lr_synthetic, mlp, openml)   ← IN PROGRESS
    → complete openml_cg_s3 + openml_bm_s3
        ↓
[Phase C] ResNet smoke test (DeepSeek + Qwen)
    → passes?
        ↓ YES
[Phase D] ResNet full ablation 3-arm × 3-seed × budget 10       ← OVERNIGHT
        ↓
[Phase E] autoresearch_macos smoke + 3-seed campaign            ← TOMORROW
        ↓
[Phase F] Compute interim bootstrap CIs
    → CIs too wide? → Phase G (large budget reruns)
    → CIs acceptable? → Phase H (paper integration)
        ↓
[Phase G] Large-budget reruns (DEFERRED, budget 30/50)
        ↓
[Phase H] P0-4 metric validation (discrimination, re-execution)
        ↓
[Phase I] Export final tables, update paper
```

**Critical dependency:** ResNet DeepSeek ablation (Phase D) requires:
1. DEEPSEEK_API_KEY set
2. Ollama running with `ollama pull qwen2.5-coder:7b`
3. ResNet smoke trial passing (Phase C)

**No dependency between:** lr_synthetic / mlp / openml campaigns —
these are fully independent and can run in any order or in parallel.

---

## 5. Current Status Summary

| Phase | Description | Trials target | Status |
|---|---|---|---|
| A | Smoke tests | 1 | ✅ Complete |
| B — lr (original) | lr_synthetic 3-arm × 3-seed | 90 | ✅ Complete |
| B — lr (expanded) | lr_synthetic 3-arm × s4+s5 | +60 | ❌ Not started |
| B — mlp (original) | mlp_synthetic summary × 1-seed | 10 | ✅ Complete |
| B — mlp (expanded) | mlp_synthetic none + rationale arms | +20 | ❌ Not started |
| B — openml (original) | OpenML 3-seed × 2-node | 120 | ✅ Complete |
| B — openml (expanded) | OpenML s4 × 2-node | +40 | ❌ Not started |
| B — mlagent s1 | mlagentbench_vectorization 1-seed | 10 | ❌ Not started |
| B — mlagent s2 (expanded) | mlagentbench_vectorization 2nd seed | +10 | ❌ Not started |
| C | ResNet smoke (DeepSeek + Qwen) | 1 | ❌ Not started |
| D | ResNet 3-arm × 3-seed × budget **15** | 135 | ⚠️ Sklearn fix committed; re-run |
| E (original) | autoresearch_macos 3-seed | 60 | ❌ Not started |
| E (expanded) | autoresearch_macos s4 | +20 | ❌ Not started |
| F | Interim bootstrap CIs | — | ❌ Blocked on D |
| G | Large-budget reruns (P0-2 deferred) | — | ⏸ Deferred |
| H | P0-4 metric validation | — | ❌ Blocked on D |
| I | Paper table export + integration | — | ❌ Blocked on H |
| **Total** | | **575** | **210/575 done** |

---

## 6. +50% Budget Expansion: Where to Add It and Why

Current planned total: **380 trials** across all nodes and phases.
Target addition: **~190 trials** (≈ 51%) allocated where they most strengthen
specific paper claims, not spread uniformly.

### Allocation table

| Node | Current | Addition | New total | Campaign IDs to add | Why here |
|---|---|---|---|---|---|
| `lr_synthetic` | 90 (3 seeds) | **+60** (2 seeds) | 150 | `deepseek_lr_{arm}_s4`, `deepseek_lr_{arm}_s5` × 3 arms | RBR ablation ordering is the central governance finding. Going 3→5 seeds cuts CI width ≈35%. The non-monotonic summary<none≈rationale claim needs to survive the extra seeds. |
| `resnet_trigger` | 90 (budget 10) | **+45** (budget 10→15) | 135 | Extend each `deepseek_resnet_{arm}_s{1-3}` to 15 trials | Flagship node. More trials = more lifecycle observations per arm, richer failure taxonomy sample. Directly addresses P0-2 (larger budget). One of the cheapest ways to strengthen the scientific node evidence. |
| `openml_*` (both) | 120 (3 seeds) | **+40** (1 seed each) | 160 | `deepseek_openml_cg_s4`, `deepseek_openml_bm_s4` | Public data reproducibility claim. Adding a 4th seed raises n from 60→80 per node and provides an independent replication the reviewer can verify. Cheap (seconds/trial). |
| `mlp_synthetic` | 10 (1 arm) | **+20** (2 arms) | 30 | `deepseek_mlp_none_s1`, `deepseek_mlp_rationale_s1` | MLP is currently 1 arm (summary only) — it cannot contribute to the memory-mode ablation table. Adding the missing arms makes it a proper 3-arm node for the manager-tier discrimination table (P0-4). Cost: near-zero (< 1 s/trial). |
| `autoresearch_macos` | 60 (3 seeds) | **+20** (1 seed) | 80 | `deepseek_autoresearch_s4` | Only minimize-direction node, only LM-training node. A 4th seed strengthens the domain-transfer claim and gives better coverage of the lifecycle state distribution on a qualitatively different task. |
| `mlagentbench_vectorization` | 10 (1 seed) | **+10** (1 seed) | 20 | `deepseek_mlagentbench_s2` | External benchmark compatibility probe. 2 seeds instead of 1 avoids the paper claim resting on a single run. Cost: ~2 min. |
| **Total** | **380** | **+195** | **575** | | **≈ 51% more** |

### Why these four nodes first (priority order)

1. **LR +60 trials** — highest impact per compute dollar. The ablation finding (memory mode reduces RBR) is the paper's most novel claim; CI width directly determines whether that claim survives peer review.
2. **ResNet +45 trials** — the only node where governance meets real GPU training. Reviewers will scrutinize this node most heavily. Deeper per-arm evidence (15 vs 10 trials) makes the failure taxonomy and lifecycle distribution tables more credible.
3. **MLP +20 trials (new arms)** — qualitatively different from adding seeds: it turns MLP from a single data point into a full ablation node. The P0-4 discrimination table currently has only ResNet and LR; adding MLP at 3 arms × 1 seed is the minimum to claim the table is general.
4. **OpenML +40 / autoresearch +20 / mlagentbench +10** — public reproducibility and portability rounding. All three are cheap; together they add 70 trials that a reviewer can independently verify or reproduce.

### Commands for the +195 additional trials

```bash
# LR synthetic: 2 extra seeds per arm (6 new campaigns, ~30 sec total)
for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s4 s5; do
    uv run python3 scripts/run_local_node_campaign.py \
      --node lr_synthetic --campaign-id deepseek_lr_${ARM}_${SEED} \
      --budget 10 --manager langgraph_manager \
      --memory-mode ${ARM} \
      --model deepseek/deepseek-v4-flash --temperature 0.2
  done
done

# MLP: add missing arms (2 new campaigns, ~10 sec total)
uv run python3 scripts/run_local_node_campaign.py \
  --node mlp_synthetic --campaign-id deepseek_mlp_none_s1 \
  --budget 10 --manager langgraph_manager --memory-mode none \
  --model deepseek/deepseek-v4-flash --temperature 0.2

uv run python3 scripts/run_local_node_campaign.py \
  --node mlp_synthetic --campaign-id deepseek_mlp_rationale_s1 \
  --budget 10 --manager langgraph_manager \
  --memory-mode append_only_summary_with_rationale \
  --model deepseek/deepseek-v4-flash --temperature 0.2

# OpenML: 1 extra seed per node (~20 min total)
uv run python3 scripts/run_openml_tabular_campaign.py \
  --node openml_credit_g --campaign-id deepseek_openml_cg_s4 \
  --budget 20 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash --temperature 0.2

uv run python3 scripts/run_openml_tabular_campaign.py \
  --node openml_bank_marketing --campaign-id deepseek_openml_bm_s4 \
  --budget 20 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash --temperature 0.2

# MLAgentBench: seed 2 (~2 min)
uv run python3 scripts/run_local_node_campaign.py \
  --node mlagentbench_vectorization --campaign-id deepseek_mlagentbench_s2 \
  --budget 10 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash --temperature 0.2

# autoresearch_macos: seed 4 (~2 hr, Apple Silicon + Ollama)
uv run python3 scripts/run_kdd_memory_ablation.py \
  --node autoresearch_macos --campaign-id deepseek_autoresearch_s4 \
  --node-root nodes/autoresearch-macos \
  --budget 20 --manager langgraph_manager \
  --memory-mode append_only_summary \
  --model deepseek/deepseek-v4-flash \
  --worker-model qwen2.5-coder:7b --temperature 0.2

# ResNet: extend budget 10→15 for each existing arm-seed
# Run AFTER the original 10-trial arms complete and the sklearn fix is committed.
# Approach: new campaign IDs at budget 15 replace the old ones in the final table.
for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s1 s2 s3; do
    uv run python3 scripts/run_kdd_memory_ablation.py \
      --node resnet_trigger --memory-mode ${ARM} \
      --campaign-id deepseek_resnet_${ARM}_${SEED} \
      --node-root nodes/ResNet_trigger --budget 15 \
      --manager langgraph_manager \
      --model deepseek/deepseek-v4-flash \
      --worker-model qwen2.5-coder:7b --temperature 0.2
  done
done
```

> **Note on ResNet budget:** Since the original 9 arm-seed campaigns all failed (sklearn bug), we reset and re-run at budget 15 directly. No need to separately run 10 then extend to 15 — just use `--budget 15` from the start on the clean re-run. This saves one full overnight cycle.

### Sequencing

Run the cheap fast-node expansions (LR, MLP, OpenML, MLAgentBench) immediately — they take < 1 hour total and can run while the ResNet overnight is queued. The autoresearch seed 4 runs concurrently with ResNet if you have a second terminal.

---

## 7. What We Are NOT Running (and why)

**P0-1 ungoverned A/B:** Deferred. Requires building a stripped-down
"ungoverned" adapter around the same manager — new code, not a config
change. High engineering cost; paper is already reframed to not claim this
result. Revisit for camera-ready or venue upgrade.

**Budget 50 ResNet (full P0-2):** Deferred to Phase G. Budget 50 × 5 seeds
× 3 arms = 750 ResNet trials at ~10 min each = ~5 days of GPU time.
Impractical before a workshop deadline. Budget 20 × 5 seeds gives adequate
CIs for workshop submission.

**mlagentbench_vectorization DeepSeek reruns:** Added to Phase B.
A 10-trial DeepSeek run (`deepseek_mlagentbench_s1`) fills the final gap
so every node has consistent DeepSeek coverage. This makes the manager-tier
discrimination table complete across all 7 nodes and removes the last hole
in the "one frontier manager, all nodes" claim.

**P1-5 AIDE wrapper:** High engineering cost; requires reverse-engineering
AIDE's manager interface. Not feasible before submission. Listed as a
limitation in the paper.

---

## 8. Key Metrics to Report per Node

After all experiments complete, the paper needs these numbers per node:

| Metric | ResNet | lr_synth | openml_* | autoresearch |
|---|---|---|---|---|
| Acceptance rate | ✓ | ✓ | ✓ | ✓ |
| Repeated-bad rate (RBR) | ✓ (by memory arm) | ✓ (by arm × tier) | — | — |
| Failure taxonomy breakdown | ✓ | — | ✓ | ✓ |
| Provenance completeness | ✓ | ✓ | ✓ | ✓ |
| Bootstrap 95% CI (seed-level) | ✓ | ✓ | ✓ | ✓ |
| Best task metric | secondary | secondary | secondary | secondary |
| Manager-tier comparison | ✓ (qwen vs. DS) | ✓ (3-arm × 2-tier) | — | — |
