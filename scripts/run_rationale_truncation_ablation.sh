#!/usr/bin/env bash
# run_rationale_truncation_ablation.sh
# Four-arm rationale-verbosity ablation for the LangGraph memory ablation.
#
# Scientific question:
#   Is the non-monotonic ordering (none ≈ rationale > summary on repeated-bad rate)
#   caused by rationale verbosity? If truncating the rationale to 50 tokens makes
#   the rationale arm match or improve over the summary arm, verbosity is the cause.
#
# Arm design:
#   arm A — no memory                   (lg_ablation2_none — ALREADY RUN, reused)
#   arm B — summary only                (lg_ablation2_append_only_summary — ALREADY RUN, reused)
#   arm C — summary + short rationale   (lg_trunc_short_rationale — NEW, 50 tokens)
#   arm D — summary + full rationale    (lg_ablation2_append_only_summary_with_rationale — ALREADY RUN, reused)
#
# Arms A, B, D reuse existing lg_ablation2_* ledgers for clean comparison.
# All arms: qwen2.5-coder:7b, temperature=0.7, budget=10, avoidance prompt.
#
# Prerequisites:
#   - Ollama running: ollama serve
#   - Model loaded:   ollama pull qwen2.5-coder:7b
#   - lg_ablation2_* ledgers present (from run_p8b_lg_ablation.sh at BUDGET=10)
#
# Run from repo root:
#   bash scripts/run_rationale_truncation_ablation.sh
#
# Analyse results:
#   python3 scripts/analyze_rationale_truncation.py

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Rationale Verbosity Ablation (budget=10, temp=0.7) ==="
echo "    Arms A/B/D: existing lg_ablation2_* ledgers"
echo "    Arm C (new): lg_trunc_short_rationale (rationale-max-tokens=50)"
echo ""

# ── Dependency check ─────────────────────────────────────────────────────────
echo "[1/6] Checking dependencies..."
uv run --extra dev python -c "import langchain_ollama" 2>/dev/null \
    || { echo "  Installing langchain-ollama..."; uv pip install langchain-ollama; }
uv run --extra dev python -c "import langgraph" 2>/dev/null \
    || { echo "  Installing langgraph..."; uv pip install langgraph; }
echo "  ✅ Dependencies OK"

# ── Ollama check ──────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking Ollama at localhost:11434..."
if ! curl -s --connect-timeout 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ❌ Ollama not reachable. Start with 'ollama serve'."
    exit 1
fi
echo "  ✅ Ollama reachable"
curl -s http://localhost:11434/api/tags | python3 -c \
    "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; print('  Models:', models)"

# ── Verify existing arms ──────────────────────────────────────────────────────
echo ""
echo "[3/6] Verifying existing arm ledgers..."
for id in lg_ablation2_none lg_ablation2_append_only_summary lg_ablation2_append_only_summary_with_rationale; do
    path="experiments/ledgers/${id}_trials.jsonl"
    if [ ! -f "$path" ]; then
        echo "  ❌ Missing: $path"
        echo "     Run scripts/run_p8b_lg_ablation.sh first (BUDGET=10 variant)."
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
SHORT_RATIONALE_TOKENS=50

# ── Arm C — summary + short rationale (NEW) ──────────────────────────────────
echo ""
echo "[4/6] Resetting node for arm C (lg_trunc_short_rationale)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_trunc_short_rationale

echo "[5/6] Running arm C: short rationale (max ${SHORT_RATIONALE_TOKENS} tokens)..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_trunc_short_rationale \
    --manager langgraph_manager \
    --memory-mode append_only_summary_with_rationale \
    --rationale-max-tokens "$SHORT_RATIONALE_TOKENS" \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Arm C done"

# ── Analysis ──────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Running four-arm analysis..."
python3 scripts/analyze_rationale_truncation.py

echo ""
echo "=== Rationale verbosity ablation complete ==="
echo "    Arm A (none):          experiments/ledgers/lg_ablation2_none_trials.jsonl"
echo "    Arm B (summary):       experiments/ledgers/lg_ablation2_append_only_summary_trials.jsonl"
echo "    Arm C (short rationale, ${SHORT_RATIONALE_TOKENS} tok): experiments/ledgers/lg_trunc_short_rationale_trials.jsonl"
echo "    Arm D (full rationale): experiments/ledgers/lg_ablation2_append_only_summary_with_rationale_trials.jsonl"
