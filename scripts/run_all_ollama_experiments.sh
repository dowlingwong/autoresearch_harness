#!/usr/bin/env bash
# run_all_ollama_experiments.sh
# Runs all remaining Ollama-dependent experiments for the KDD paper in one shot.
#
# Experiments:
#   1. P13-H arm C — rationale truncation (lg_trunc_short_rationale, budget=10, 50 tokens)
#   2. Rep2         — seed replication arm 2 (lg_ablation_rep2_*, budget=10)
#
# Prerequisites:
#   - Ollama running:       ollama serve
#   - Model loaded:         ollama pull qwen2.5-coder:7b
#   - lg_ablation2_* done:  experiments/ledgers/lg_ablation2_*_trials.jsonl (10 trials each)
#
# Run from repo root:
#   bash scripts/run_all_ollama_experiments.sh
#
# After completion:
#   python3 scripts/analyze_rationale_truncation.py  # 4-arm verbosity analysis
#   python3 scripts/analyze_seed_replication.py       # 3-rep ordering analysis

set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================================"
echo "  KDD Paper — All Remaining Ollama Experiments"
echo "  1. P13-H: Rationale truncation arm C (50 tokens)"
echo "  2. Rep2:  Seed replication arm 2 (budget=10)"
echo "========================================================"
echo ""

# ── Dependency check ─────────────────────────────────────────────────────────
echo "[1/12] Checking Python dependencies..."
uv run --extra dev python -c "import langchain_ollama" 2>/dev/null \
    || { echo "  Installing langchain-ollama..."; uv pip install langchain-ollama; }
uv run --extra dev python -c "import langgraph" 2>/dev/null \
    || { echo "  Installing langgraph..."; uv pip install langgraph; }
echo "  ✅ Dependencies OK"

# ── Ollama check ──────────────────────────────────────────────────────────────
echo ""
echo "[2/12] Checking Ollama at localhost:11434..."
if ! curl -s --connect-timeout 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ❌ Ollama not reachable. Start with: ollama serve"
    exit 1
fi
echo "  ✅ Ollama reachable"
curl -s http://localhost:11434/api/tags | python3 -c \
    "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; print('  Models:', models)"

# ── Verify prerequisite lg_ablation2_* arms ──────────────────────────────────
echo ""
echo "[3/12] Verifying prerequisite lg_ablation2_* arms (A, B, D)..."
for id in lg_ablation2_none lg_ablation2_append_only_summary lg_ablation2_append_only_summary_with_rationale; do
    path="experiments/ledgers/${id}_trials.jsonl"
    if [ ! -f "$path" ]; then
        echo "  ❌ Missing: $path"
        echo "     Run scripts/run_p8b_lg_ablation.sh first."
        exit 1
    fi
    n=$(wc -l < "$path")
    echo "  ✅ ${id}: ${n} trials"
done

# ── Fast-run environment ──────────────────────────────────────────────────────
export RESNET_TRIGGER_FAST_SEARCH=1
export RESNET_TRIGGER_FAST_N_SIGNAL=1000
export RESNET_TRIGGER_FAST_N_NOISE=1000
export RESNET_TRIGGER_FAST_TRACE_LEN=4096
export RESNET_TRIGGER_FAST_BATCH_SIZE=64
export RESNET_TRIGGER_FAST_EPOCHS=3
export RESNET_TRIGGER_FAST_SKIP_TEST=1
export RESNET_TRIGGER_EARLY_STOP_PATIENCE=2
export RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002
export RESNET_TRIGGER_DEVICE=cpu

MODEL="qwen2.5-coder:7b"
NODE_ROOT="nodes/ResNet_trigger"
BUDGET=10
TEMPERATURE=0.7

echo ""
echo "========================================================"
echo "  EXPERIMENT 1: P13-H Rationale Truncation (arm C)"
echo "========================================================"

# ── Arm C: short rationale ────────────────────────────────────────────────────
echo ""
echo "[4/12] Resetting node for arm C (lg_trunc_short_rationale)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_trunc_short_rationale

echo "[5/12] Running arm C: summary + short rationale (50 tokens)..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_trunc_short_rationale \
    --manager langgraph_manager \
    --memory-mode append_only_summary_with_rationale \
    --rationale-max-tokens 50 \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Arm C done"

echo ""
echo "[6/12] Running 4-arm rationale verbosity analysis..."
python3 scripts/analyze_rationale_truncation.py

echo ""
echo "========================================================"
echo "  EXPERIMENT 2: Seed Replication Rep2 (budget=10)"
echo "========================================================"

echo ""
echo "[7/12] Resetting node for rep2 arm 1 (lg_ablation_rep2_none)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation_rep2_none

echo "[8/12] Running rep2 arm 1: no memory..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_ablation_rep2_none \
    --manager langgraph_manager \
    --memory-mode none \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Rep2 arm 1 done"

echo ""
echo "[9/12] Resetting node for rep2 arm 2 (lg_ablation_rep2_append_only_summary)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation_rep2_append_only_summary

echo "[10/12] Running rep2 arm 2: summary memory..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_ablation_rep2_append_only_summary \
    --manager langgraph_manager \
    --memory-mode append_only_summary \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Rep2 arm 2 done"

echo ""
echo "[11/12] Resetting node for rep2 arm 3 (lg_ablation_rep2_append_only_summary_with_rationale)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation_rep2_append_only_summary_with_rationale

echo "[12/12] Running rep2 arm 3: full rationale memory..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_ablation_rep2_append_only_summary_with_rationale \
    --manager langgraph_manager \
    --memory-mode append_only_summary_with_rationale \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Rep2 arm 3 done"

echo ""
echo "========================================================"
echo "  COMBINED ANALYSIS"
echo "========================================================"

echo ""
echo "--- Rationale Verbosity (4-arm) ---"
python3 scripts/analyze_rationale_truncation.py

echo ""
echo "--- Seed Replication (3-rep) ---"
python3 scripts/analyze_seed_replication.py

echo ""
echo "========================================================"
echo "  All experiments complete."
echo ""
echo "  Next steps:"
echo "  1. Review both analyses above."
echo "  2. Update paper §5 with truncation finding and §5.3 with 3/3 replicates."
echo "  3. Recompile LaTeX."
echo "========================================================"
