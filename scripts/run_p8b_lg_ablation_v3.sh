#!/usr/bin/env bash
# run_p8b_lg_ablation_v3.sh
# Priority 8b (v3): Fair memory ablation with LangGraph stochastic backend.
#
# Changes vs v2 (run_p8b_lg_ablation.sh):
#   - budget raised from 10 → 20 (denominator=19, precision=0.053/trial vs 0.111;
#     covers ~83% of 24 first-order proposals = 12 editable constants × 2 directions)
#   - campaign IDs: lg_ablation3_* (preserves v2 lg_ablation2_* evidence untouched)
#
# v2 (BUDGET=10, lg_ablation2_*) is kept as-is in run_p8b_lg_ablation.sh.
# v1 (BUDGET=5, edit_failed, pre-bridge, lg_ablation_*) is documented in
#    paper/notes/langgraph_edit_failed_finding.md.
#
# Prerequisites:
#   - Ollama running: ollama serve
#   - Model loaded:   ollama pull qwen2.5-coder:7b
#   - langchain-ollama installed (script checks below)
#
# Run from the repo root:
#   bash scripts/run_p8b_lg_ablation_v3.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Priority 8b v3: LangGraph Memory Ablation (budget=20, temp=0.7, avoidance prompt) ==="
echo "    (v2 backup: run_p8b_lg_ablation.sh, campaign IDs lg_ablation2_*, budget=10)"
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
BUDGET=20
TEMPERATURE=0.7

echo ""
echo "  BUDGET=$BUDGET  TEMPERATURE=$TEMPERATURE  MODEL=$MODEL"
echo "  Campaign IDs: lg_ablation3_{none,append_only_summary,append_only_summary_with_rationale}"
echo ""

# ── ARM 1: no memory ─────────────────────────────────────────────────────────
echo "[3/9] Resetting node for arm 1 (lg_ablation3_none)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation3_none

echo "[4/9] Running arm 1: lg_ablation3_none (memory-mode=none, budget=$BUDGET, temp=$TEMPERATURE)..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_ablation3_none \
    --manager langgraph_manager \
    --memory-mode none \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Arm 1 done"

# ── ARM 2: append-only summary ───────────────────────────────────────────────
echo ""
echo "[5/9] Resetting node for arm 2 (lg_ablation3_append_only_summary)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation3_append_only_summary

echo "[6/9] Running arm 2: lg_ablation3_append_only_summary (budget=$BUDGET, temp=$TEMPERATURE)..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_ablation3_append_only_summary \
    --manager langgraph_manager \
    --memory-mode append_only_summary \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Arm 2 done"

# ── ARM 3: summary + rationale ───────────────────────────────────────────────
echo ""
echo "[7/9] Resetting node for arm 3 (lg_ablation3_append_only_summary_with_rationale)..."
uv run --extra dev python scripts/reset_node_state.py \
    --node resnet_trigger --campaign-id lg_ablation3_append_only_summary_with_rationale

echo "[8/9] Running arm 3: lg_ablation3_append_only_summary_with_rationale (budget=$BUDGET, temp=$TEMPERATURE)..."
uv run --extra dev python scripts/run_kdd_main_campaign.py \
    --node resnet_trigger \
    --budget "$BUDGET" \
    --campaign-id lg_ablation3_append_only_summary_with_rationale \
    --manager langgraph_manager \
    --memory-mode append_only_summary_with_rationale \
    --node-root "$NODE_ROOT" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --no-export
echo "  ✅ Arm 3 done"

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "[9/9] Verification — governance metrics and proposal sequences:"
python3 - << 'PYEOF'
import json, pathlib

arms = [
    "lg_ablation3_none",
    "lg_ablation3_append_only_summary",
    "lg_ablation3_append_only_summary_with_rationale",
]
all_proposals = {}
for cid in arms:
    p = pathlib.Path(f"experiments/ledgers/{cid}_trials.jsonl")
    if not p.exists():
        print(f"  ❌ {cid}: ledger missing")
        continue
    recs = [json.loads(l) for l in p.open() if l.strip()]
    proposals = [r.get("proposal_summary", "?") for r in recs]
    decisions  = [r.get("decision", "?") for r in recs]
    params = [
        (r.get("extra") or {}).get("manager", {}).get("structured_edit", {}).get("symbol", "?")
        for r in recs
    ]
    kept    = sum(1 for d in decisions if d == "kept")
    discard = sum(1 for d in decisions if d == "discarded")
    failed  = sum(1 for d in decisions if d == "failed_invalid")

    # repeated-bad rate from last record
    last_rbc = (recs[-1].get("extra") or {}).get("manager", {}).get(
        "worker_repeated_bad_stats", {}).get("repeated_bad_count", "?")
    rbc_rate = f"{last_rbc}/{len(recs)}" if isinstance(last_rbc, int) else "?"

    # first trial switching away from T1 param
    first_param = params[0] if params else "?"
    switch_trial = next(
        (i + 1 for i, p_ in enumerate(params) if p_ != first_param), None
    )

    # unique params explored
    unique_params = sorted(set(p for p in params if p != "?"))

    # best val_auc
    import re
    best_auc = 0.0
    for r in recs:
        text = json.dumps(r.get("extra", {}))
        # current_best_after is stored in worker extra
        hits = re.findall(r'"current_best_after":\s*([\d.]+)', text)
        for h in hits:
            try:
                v = float(h)
                if v > best_auc:
                    best_auc = v
            except: pass

    print(f"\n  {cid}:")
    print(f"    kept={kept}  discarded={discard}  failed_invalid={failed}")
    print(f"    repeated_bad: {rbc_rate}  first_switch_trial: {switch_trial}")
    print(f"    unique params explored: {unique_params}")
    print(f"    best_val_auc (approx):  {best_auc:.6f}")
    all_proposals[cid] = proposals

# Check diversity
sets = [tuple(v) for v in all_proposals.values()]
if len(set(sets)) == 1:
    print("\n  ⚠️  Proposals IDENTICAL across all arms — memory may not be active.")
else:
    print("\n  ✅ Proposals DIFFER across arms.")

# Hypothesis check
print("\n  Hypothesis check (none > summary > rationale on repeated_bad_rate):")
for cid in arms:
    p = pathlib.Path(f"experiments/ledgers/{cid}_trials.jsonl")
    if not p.exists():
        continue
    recs = [json.loads(l) for l in p.open() if l.strip()]
    n = len(recs)
    rbc = (recs[-1].get("extra") or {}).get("manager", {}).get(
        "worker_repeated_bad_stats", {}).get("repeated_bad_count", 0)
    print(f"    {cid}: repeated_bad_rate={rbc}/{n}={rbc/n:.3f}")

PYEOF

echo ""
echo "=== P8b v3 complete. Ledgers: experiments/ledgers/lg_ablation3_* ==="
echo "    v2 backup still intact: experiments/ledgers/lg_ablation2_*"
echo "    Next: regenerate paper tables/figures targeting lg_ablation3_* or update"
echo "    05_results.tex / 06_discussion_limitations.tex with new numbers."
