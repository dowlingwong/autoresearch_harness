# Artifact: A Governed Harness for Auditable LLM-Driven ML Experimentation

**Paper:** "Evaluating Governed LLM-Driven ML Experimentation: Lifecycle, Provenance,
and Failure Metrics"  
**Venue:** KDD 2026 ETAAI Workshop  
**Commit:** `ff3717b729652d4ba1de50848d8533078ccbcdfa`  
**Repository:** https://github.com/dowlingwong/autoresearch_harness

---

## What this artifact contains

```
autoresearch-harness-artifact/
├── ARTIFACT_README.md          ← this file
├── pyproject.toml              ← harness dependencies (Python ≥ 3.11)
├── uv.lock                     ← locked dependency graph
├── .python-version             ← pinned Python version
├── src/autoresearch/           ← full control-plane source code
│   ├── control_plane/          ← campaign loop, decision, permissions, state machine
│   ├── memory/                 ← ledger store, summarizer, similarity detector
│   ├── manager/                ← baseline, prompt, langgraph managers
│   ├── worker/                 ← DryRunWorker, LocalWorker, ClawWorker
│   ├── nodes/                  ← NodeSpec loader and registry
│   └── evaluation/             ← CampaignMetrics, ablation summaries
├── configs/nodes/              ← YAML node specs for all paper nodes
├── scripts/                    ← campaign runners and export scripts
├── tests/                      ← unit + integration test suite
├── experiments/ledgers/        ← append-only JSONL trial records (paper evidence)
├── paper_figures.ipynb         ← notebook that regenerates all paper figures/tables
└── nodes/                      ← per-node training environments
    ├── ResNet_trigger/         ← Physics CNN (requires GPU + HDF5 data files)
    ├── openml_bank_marketing/  ← OpenML 1461, CPU-only
    ├── openml_credit_g/        ← OpenML 31, CPU-only
    ├── lr_synthetic/           ← Synthetic LR, CPU-only
    ├── mlp_synthetic/          ← Synthetic MLP, CPU-only
    └── mlagentbench_vectorization/  ← MLAgentBench, CPU-only
```

---

## Quick start — verify governance metrics without any GPU

The paper's primary claims (acceptance rate, invalid rate, repeated-bad rate,
provenance completeness) are all recomputable from the archived ledgers without
re-running any training:

```bash
# 1. Install the harness
pip install -e ".[dev]"          # or: uv sync

# 2. Run the test suite to verify the control-plane contracts
pytest -p no:cacheprovider

# 3. Regenerate paper governance tables from the archived ledgers
python3 scripts/export_kdd_tables.py

# 4. Regenerate paper figures from the archived ledgers
python3 scripts/export_kdd_figures.py
# or open paper_figures.ipynb → Kernel → Restart & Run All
```

All paper tables and figures are produced from `experiments/ledgers/`; no
training re-execution is required to reproduce the reported numbers.

---

## Dry-run campaigns (no GPU, no Ollama)

A dry-run campaign exercises the full control-plane loop (lifecycle, guards,
ledger writes, memory) using the `DryRunWorker` (synthetic metrics) and the
`BaselineManager` (deterministic proposals). It validates governance behaviour
without executing real training.

```bash
# ResNet trigger node — 5-trial dry campaign
python3 scripts/run_campaign.py \
    --node resnet_trigger --campaign-id dry_resnet --budget 5 --dry-run

# OpenML bank-marketing — 5-trial dry campaign
python3 scripts/run_campaign.py \
    --node openml_bank_marketing --campaign-id dry_bm --budget 5 --dry-run

# Memory ablation dry-run (all three modes)
python3 scripts/run_memory_ablation.py \
    --node resnet_trigger --budget 3 --dry-run
```

---

## Paper campaigns — hardware and model requirements

### ResNet trigger (main result, Table 1 / Fig. 2)

**Hardware:** NVIDIA GPU with ≥ 8 GB VRAM (paper used NVIDIA L40S)  
**Data:** `nodes/ResNet_trigger/noise_traces_4000x8000.h5` and
          `nodes/ResNet_trigger/signal_vacuum_sum_crop_4000x8000.h5` (included)  
**Manager:** `deepseek-v4-flash` via OpenAI-compatible API  
**Worker:** `qwen2.5-coder:7b` via Ollama at `http://localhost:11434`

```bash
# Set up node environment (first time only)
cd nodes/ResNet_trigger && uv sync && cd ../..

# Memory ablation (3 arms × 5 seeds × 15 trials = 225 trials total)
python3 scripts/run_memory_ablation.py \
    --node resnet_trigger --budget 15 --execute-real-campaigns \
    --node-root nodes/ResNet_trigger \
    --model qwen2.5-coder:7b --host http://localhost:11434
```

Campaign IDs produced: `deepseek_resnet_none_s{1-5}`,
`deepseek_resnet_append_only_summary_s{1-5}`,
`deepseek_resnet_append_only_summary_with_rationale_s{1-5}`.

Reset integrity can be verified by checking `node_state_hash` at `budget_index=1`
across all seeds — all five should share prefix `9c0118f33fcb`:

```bash
python3 -c "
import json
for arm in ['none','append_only_summary','append_only_summary_with_rationale']:
  for s in range(1,6):
    path = f'experiments/ledgers/deepseek_resnet_{arm}_s{s}_trials.jsonl'
    r = json.loads(open(path).readline())
    print(arm[:4], f's{s}', r['node_state_hash'][:12])
"
```

### OpenML nodes (Table 2)

**Hardware:** CPU-only  
**Manager:** `deepseek-v4-flash` via OpenAI-compatible API

```bash
# bank-marketing, 5 seeds × 30 trials
for S in 1 2 3 4 5; do
  python3 scripts/run_campaign.py \
    --node openml_bank_marketing \
    --campaign-id deepseek_openml_bm_b30_s${S} \
    --budget 30 --manager prompt_manager \
    --memory-mode append_only_summary_with_rationale \
    --node-root nodes/openml_bank_marketing
done

# credit-g, 5 seeds × 30 trials (same pattern, --node openml_credit_g)
```

### Autoresearch Linux (Table 3 / Section 5.3)

**Hardware:** NVIDIA GPU (for the coding sub-agent's inference)  
**Manager:** `deepseek-v4-flash`; **Worker:** claw-style coding agent

```bash
# none-memory arm, 4 seeds × 30 trials
for S in 1 2 3 4; do
  python3 scripts/run_campaign.py \
    --node autoresearch_linux \
    --campaign-id deepseek_autoresearch_linux_none_s${S} \
    --budget 30 --manager prompt_manager --memory-mode none \
    --node-root nodes/autoresearch-macos
done
```

### LR synthetic (LangGraph ablation, Fig. 3)

**Hardware:** CPU-only  
**Manager:** `langgraph_manager` with `deepseek-v4-flash`

```bash
python3 scripts/run_memory_ablation.py \
    --node lr_synthetic --budget 30 \
    --execute-dry-campaigns    # or --execute-real-campaigns with API key
```

### MLP synthetic (memory-mode ablation, Fig. 3)

```bash
python3 scripts/run_memory_ablation.py \
    --node mlp_synthetic --budget 30 \
    --execute-real-campaigns \
    --node-root nodes/mlp_synthetic
```

### Stress trials (Section 4.4)

```bash
# Scope-violation stress: all trials should fail as invalid_edit_scope
python3 scripts/run_kdd_stress_trial.py --mode scope

# No-op patch stress: all trials should fail as no_op_patch
python3 scripts/run_kdd_stress_trial.py --mode noop
```

### Counterfactual (governed vs ungoverned, Fig. 4)

```bash
python3 scripts/run_counterfactual.py \
    --node openml_bank_marketing --budget 30
```

---

## Recover a stale pending trial

If a campaign crashes mid-trial:

```bash
python3 scripts/recover_pending.py --list
python3 scripts/recover_pending.py --inspect experiments/ledgers/<id>_pending.json
python3 scripts/recover_pending.py --mark-failed <id> --reason "OOM during training"
```

---

## Inspect a node spec

```bash
python3 scripts/inspect_node.py --node resnet_trigger
python3 scripts/inspect_node.py --node openml_bank_marketing
```

---

## Environment notes

| Component | Version |
|-----------|---------|
| Python (harness) | ≥ 3.11 (pinned in `.python-version`) |
| LangGraph | ≥ 1.0.0 |
| LangChain-Ollama | ≥ 0.3.0 |
| DeepSeek manager API | `deepseek-v4-flash` (OpenAI-compatible) |
| Ollama worker model | `qwen2.5-coder:7b` (temperature 0.2) |
| ResNet node Python | 3.12 (see `nodes/ResNet_trigger/pyproject.toml`) |
| OpenML / MLP nodes | 3.11+ |

Set `OPENAI_API_KEY` (or equivalent) for DeepSeek API access.
Set `OLLAMA_HOST` if Ollama is not on `http://localhost:11434`.

---

## Ledger provenance

Every trial in `experiments/ledgers/` is an append-only JSONL record.
Each record contains:

- `trial_id`, `campaign_id`, `budget_index` — identity
- `decision` — `kept | discarded | failed_invalid`
- `failure_category` — non-null on `failed_invalid`
- `parsed_metrics` — `val_auc` or `val_bpb` where applicable
- `provenance` — 5-tuple of content-addressed IDs (proposal→patch→run→metric→decision)
- `node_state_hash` — SHA-256 of editable file set at trial start

Governance metrics are recomputable with:

```bash
python3 -c "
from autoresearch.evaluation.campaign_summary import load_campaign_summary
m = load_campaign_summary('deepseek_resnet_none_s1')
print(m)
"
```
