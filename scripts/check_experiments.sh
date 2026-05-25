#!/usr/bin/env bash
# =============================================================================
# check_experiments.sh — quick sanity check for both Linux campaigns
# =============================================================================
# Run from the repo root on deepthought2:
#   bash scripts/check_experiments.sh
# Or from Mac via:
#   ssh dwong@deepthought2.etp.kit.edu 'bash /ceph/dwong/autoresearch_harness/scripts/check_experiments.sh'
# =============================================================================

REPO="/ceph/dwong/autoresearch_harness"
LEDGERS="$REPO/experiments/ledgers"

echo "============================================================"
echo " Experiment Health Check — $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. GPU + process status
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
  | awk -F', ' '{printf "  GPU %s (%s): util=%s%%  mem=%s/%s MiB\n",$1,$2,$3,$4,$5}'
else
  echo "  nvidia-smi not found"
fi

echo ""
echo "--- Active training processes ---"
pgrep -a python3 2>/dev/null | grep -E "train\.py|run_kdd" | head -8 \
  | sed 's/^/  /' || echo "  (none found)"

echo ""
echo "--- Ollama ---"
if curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  models=$(curl -s http://localhost:11434/api/tags | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(', '.join(m['name'] for m in d.get('models',[])))" 2>/dev/null)
  echo "  Running. Models: ${models:-none}"
else
  echo "  NOT RUNNING — campaigns will fail without it!"
  echo "  Start: OLLAMA_MODELS=/ceph/dwong/ollama_models OLLAMA_HOST=127.0.0.1:11434 \\"
  echo "           /ceph/dwong/ollama_bin/bin/ollama serve > /ceph/dwong/ollama.log 2>&1 &"
fi

# ---------------------------------------------------------------------------
# 2. ResNet campaigns
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " ResNet (3 arms × 3 seeds × 15 trials = 135 target)"
echo "============================================================"

python3 - <<'PYEOF'
import json, os, glob

LEDGERS = "/ceph/dwong/autoresearch_harness/experiments/ledgers"
ARMS = ["none", "append_only_summary", "append_only_summary_with_rationale"]
SEEDS = ["s1", "s2", "s3"]
TARGET = 15

total = 0
issues = []
for arm in ARMS:
    print(f"\n  arm: {arm}")
    for seed in SEEDS:
        path = f"{LEDGERS}/deepseek_resnet_{arm}_{seed}_trials.jsonl"
        if not os.path.exists(path):
            print(f"    {seed}: MISSING")
            issues.append(f"resnet/{arm}/{seed}: ledger missing")
            continue
        lines = open(path).readlines()
        n = len(lines)
        total += n
        decisions = {}
        wc_list = []
        bad = 0
        for l in lines:
            t = json.loads(l)
            d = t.get("decision", "?")
            decisions[d] = decisions.get(d, 0) + 1
            wc = t.get("wall_clock_seconds", 0) or 0
            if wc: wc_list.append(wc)
            if d == "failed_invalid": bad += 1
        avg_wc = sum(wc_list)/len(wc_list) if wc_list else 0
        status = "✓" if n >= TARGET else f"IN PROGRESS ({n}/{TARGET})"
        flag = " ← ALL FAILED" if bad == n and n > 0 else ""
        print(f"    {seed}: {status}  decisions={decisions}  avg_wall={avg_wc:.0f}s{flag}")
        if bad == n and n > 0:
            issues.append(f"resnet/{arm}/{seed}: all {n} trials failed_invalid")
        if avg_wc > 0 and avg_wc < 10:
            issues.append(f"resnet/{arm}/{seed}: suspiciously fast avg={avg_wc:.0f}s")

print(f"\n  Total trials: {total}/135")
PYEOF

# ---------------------------------------------------------------------------
# 3. Autoresearch campaigns
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Autoresearch (2 arms × 4 seeds × 30 trials = 240 target)"
echo "============================================================"

python3 - <<'PYEOF'
import json, os

LEDGERS = "/ceph/dwong/autoresearch_harness/experiments/ledgers"
ARMS = ["none", "append_only_summary"]
SEEDS = ["s1", "s2", "s3", "s4"]
TARGET = 30

total = 0
issues = []
for arm in ARMS:
    print(f"\n  arm: {arm}")
    for seed in SEEDS:
        path = f"{LEDGERS}/deepseek_autoresearch_linux_{arm}_{seed}_trials.jsonl"
        if not os.path.exists(path):
            print(f"    {seed}: not started yet")
            continue
        lines = open(path).readlines()
        n = len(lines)
        total += n
        decisions = {}
        wc_list = []
        metric_vals = []
        for l in lines:
            t = json.loads(l)
            d = t.get("decision", "?")
            decisions[d] = decisions.get(d, 0) + 1
            wc = t.get("wall_clock_seconds", 0) or 0
            if wc: wc_list.append(wc)
            mv = t.get("metric_value")
            if mv is not None: metric_vals.append(mv)
        avg_wc = sum(wc_list)/len(wc_list) if wc_list else 0
        status = "✓" if n >= TARGET else f"in progress ({n}/{TARGET})"
        all_fail = decisions.get("failed_invalid", 0) == n and n > 0
        flag = ""
        if all_fail and avg_wc < 30:
            flag = " ← fast failures (check PYTHONPATH)"
        elif all_fail and avg_wc > 200:
            flag = " ← metric_missing bug? (check claw_worker.py fix)"
        elif all_fail:
            flag = " ← all failed"
        best = f"  best_metric={min(metric_vals):.4f}" if metric_vals else ""
        print(f"    {seed}: {status}  decisions={decisions}  avg_wall={avg_wc:.0f}s{best}{flag}")

print(f"\n  Total trials: {total}/240")
PYEOF

# ---------------------------------------------------------------------------
# 4. Live log tail (last campaign to run)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Recent log activity (last 8 lines of newest master.log)"
echo "============================================================"
newest=$(ls -t "$REPO/logs/linux_autoresearch_"*/master.log 2>/dev/null | head -1)
if [[ -n "$newest" ]]; then
  echo "  $newest"
  tail -8 "$newest" | sed 's/^/  /'
fi
newest_resnet=$(ls -t "$REPO/logs/linux_resnet_"*/master.log 2>/dev/null | head -1)
if [[ -n "$newest_resnet" ]]; then
  echo ""
  echo "  $newest_resnet"
  tail -4 "$newest_resnet" | sed 's/^/  /'
fi

echo ""
echo "============================================================"
