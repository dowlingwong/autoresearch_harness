#!/usr/bin/env bash
# run_lg_seed_replication.sh
# Seed replication of the LangGraph memory ablation (rep 2 of 3).
#
# Purpose: verify the non-monotonic ordering (summary < none ≈ rationale on
# repeated-bad rate) holds in a second independent replicate, giving the paper
# the "ordering held in X of Y seed pairs" claim.
#
# Replicate 1 (primary):  lg_ablation2_*   (budget=10, v2)
# Replicate 2 (this):     lg_ablation_rep2_*  (budget=10, v2 settings)
# Replicate 3 (extended): lg_ablation3_*   (budget=20, v3)
#
# All three replicates use:
#   manager=langgraph_manager, model=qwen2.5-coder:7b, temperature=0.7,
#   avoidance prompt, deterministic patch bridge.
# Natural LLM stochasticity (not a manual seed) produces the different outcomes.
#
# Prerequisites:
#   - Ollama running: ollama serve
#   - Model loaded:   ollama pull qwen2.5-coder:7b
#
# Run from the repo root:
#   bash scripts/run_lg_seed_replication.sh
#
# After completion, run the analysis:
#   python3 scripts/analyze_seed_replication.py

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== LangGraph Ablation: Seed Replication 2 (budget=10, temp=0.7) ==="
echo "    Campaign IDs: lg_ablation_rep2_{none,append_only_summary,append_only_summary_with_rationale}"
echo "    Primary replicate: lg_ablation2_*  (already complete)"
echo ""

# ── Dependency check ────────────────────────────────────────────────────────
echo "[1/9] Checking dependencies..."
uv run --extra dev python -c "import langchain_ollama" 2>/dev/null \
    || { echo "  Installing langchain-ollama..."; uv pip install langchain-ollama; }
uv run --extra dev python -c "import langgraph" 2>/dev/null \
    || { echo "  Installing langgraph..."; uv pip install langgraph; }
echo "  ✅ Dependencies OK"

# ── Ollama check ─────────────────────────────────────────────────────────────
echo ""
echo "[2/9] Checking Ollama at localhost:11434..."
if ! curl -s --connect-timeout 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ❌ Ollama not reachable. Make sure 'ollama serve' is running."
    exit 1
fi
echo "  ✅ Ollama reachable"
curl -s http://localhost:11434/api/tags | python3 -c \
    "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; print('  Models:', models)"

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
echo "  BUDGET=$BUDGET  TEMPERATURE=$TEMPERATURE  MODEL=$MODEL  (same as primary replicate)"
echo ""

# ── ARM 1: no memory ─────────────────────────────────────────────────────────
echo "[3/9] Resetting node for rep2 arm 1 (lg_ablation_rep2_none)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation_rep2_none

echo "[4/9] Running rep2 arm 1: lg_ablation_rep2_none..."
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

# ── ARM 2: append-only summary ───────────────────────────────────────────────
echo ""
echo "[5/9] Resetting node for rep2 arm 2 (lg_ablation_rep2_append_only_summary)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation_rep2_append_only_summary

echo "[6/9] Running rep2 arm 2: lg_ablation_rep2_append_only_summary..."
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

# ── ARM 3: summary + rationale ───────────────────────────────────────────────
echo ""
echo "[7/9] Resetting node for rep2 arm 3 (lg_ablation_rep2_append_only_summary_with_rationale)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation_rep2_append_only_summary_with_rationale

echo "[8/9] Running rep2 arm 3: lg_ablation_rep2_append_only_summary_with_rationale..."
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

# ── Analysis ─────────────────────────────────────────────────────────────────
echo ""
echo "[9/9] Running cross-replicate ordering analysis..."
python3 scripts/analyze_seed_replication.py

echo ""
echo "=== Seed replication 2 complete ==="
echo "    Ledgers: experiments/ledgers/lg_ablation_rep2_*"
echo "    Primary: experiments/ledgers/lg_ablation2_*"
echo "    Analysis: python3 scripts/analyze_seed_replication.py"
