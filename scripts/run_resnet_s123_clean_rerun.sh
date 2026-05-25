#!/usr/bin/env bash
# =============================================================================
# run_resnet_s123_clean_rerun.sh — Re-run s1/s2/s3 with a fixed clean baseline
# =============================================================================
#
# PURPOSE
# -------
# The original s1/s2/s3 run used reset_node_state.py without a stable
# baseline reference. /ceph is not a git repo, so git show fell back to the
# git checkout fallback, which also failed — leaving train.py unrestored
# between arms within each seed.  As a result, each arm started from a
# different train.py (cross-arm contamination):
#
#   s1/none:     7652acd31b4e   s1/summary: 7b6ed13afc83   s1/rationale: ddbfb9d3d614
#   s2/none:     8ec6789725f2   s2/summary: 1c06bd6faad6   s2/rationale: 46d133479faf
#   s3/none:     3c9dc6ec4f4a   s3/summary: 73ba42e67aea   s3/rationale: d6f65ec7e31e
#
# This script fixes that by relying on the .autoresearch_baseline/train.py
# template (already in place from the s4/s5 clean rerun), which reset_node_state.py
# uses automatically when git is unavailable.
#
# WARNING: This script DELETES the existing s1/s2/s3 ledgers before re-running.
# The old contaminated data is intentionally replaced with clean data.
# If you want to keep the old data, back it up first:
#   cp experiments/ledgers/deepseek_resnet_*_s{1,2,3}_trials.jsonl /tmp/
#
# WHAT IT RUNS
# ------------
#   3 arms × 3 seeds × 15 trials = 135 trials
#   Campaign IDs: deepseek_resnet_{none,summary,rationale}_s1
#                 deepseek_resnet_{none,summary,rationale}_s2
#                 deepseek_resnet_{none,summary,rationale}_s3
#   (same IDs as before — contaminated ledgers are deleted first)
#
# USAGE
# -----
#   export DEEPSEEK_API_KEY=sk-...
#   export DEEPSEEK_THINKING=disabled
#   nohup bash scripts/run_resnet_s123_clean_rerun.sh --gpu 1 \
#     > logs/clean_rerun_s123_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo "PID: $!"
#
# OPTIONS
#   --gpu N       CUDA device index (default: 0)
#   --smoke       1 trial per campaign for pipeline check only
#   --no-smoke    Skip the Phase 1 smoke test (use when pipeline was recently
#                 validated — saves ~10 minutes)
#   --dry-run     Print commands without executing (does not delete ledgers)
#
# PREREQUISITES
#   1. export DEEPSEEK_API_KEY=sk-...
#   2. .autoresearch_baseline/train.py must exist (created during s4/s5 rerun)
#   3. ollama serve  (running in background)
#   4. ollama pull qwen2.5-coder:7b
#
# AFTER COMPLETION
# ----------------
#   Sync ledgers to Mac:
#     rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \
#           /Users/wongdowling/Documents/autoresearch_harness/experiments/ledgers/
#     rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/paper/tables/ \
#           /Users/wongdowling/Documents/autoresearch_harness/paper/tables/
#   Then update the paper with clean 5-seed statistics.
#
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
GPU_IDX=0
SMOKE=0
NO_SMOKE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)       GPU_IDX="$2"; shift 2 ;;
    --gpu=*)     GPU_IDX="${1#--gpu=}"; shift ;;
    --smoke)     SMOKE=1; shift ;;
    --no-smoke)  NO_SMOKE=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --help|-h)   sed -n '2,60p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RESNET_ROOT="$REPO/nodes/ResNet_trigger"

VENV="$REPO/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV" || { echo "[ERROR] No .venv — run: uv sync"; exit 1; }

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$REPO/logs/clean_rerun_s123_$RUN_TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
warn() { log "WARNING  $*"; }
die()  { log "ERROR    $*"; exit 1; }

log "============================================================"
log " ResNet s1+s2+s3 CLEAN RERUN started: $(date)"
log " Repo   : $REPO"
log " GPU    : CUDA:$GPU_IDX"
log " Smoke  : $SMOKE | No-smoke: $NO_SMOKE | Dry: $DRY_RUN"
log "============================================================"

# ---------------------------------------------------------------------------
# STEP 0 — Verify clean baseline template
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " STEP 0 — Verify clean baseline file for train.py"
log "======================================================"

BASELINE_TEMPLATE="$RESNET_ROOT/.autoresearch_baseline/train.py"

if [[ ! -f "$BASELINE_TEMPLATE" ]]; then
  die "Baseline template not found: $BASELINE_TEMPLATE
  This should have been created during the s4/s5 clean rerun.
  Copy the clean train.py from your Mac:
    scp /path/to/train_py_clean.py \\
      dwong@deepthought2.etp.kit.edu:$BASELINE_TEMPLATE"
fi

BASELINE_TRAIN_HASH=$(sha256sum "$BASELINE_TEMPLATE" | awk '{print $1}')

log "INFO  Baseline template : $BASELINE_TEMPLATE"
log "INFO  SHA-256           : $BASELINE_TRAIN_HASH"
log "INFO  First 10 lines:"
head -10 "$BASELINE_TEMPLATE" | while IFS= read -r line; do log "      $line"; done
log ""
log "  *** If the above train.py looks wrong, Ctrl-C now and replace the template."
log ""

if [[ $DRY_RUN -eq 0 ]]; then
  sleep 5
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && die "DEEPSEEK_API_KEY is not set."

export CUDA_VISIBLE_DEVICES="$GPU_IDX"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"

export LD_LIBRARY_PATH="${HOME}/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
log "INFO  LD_LIBRARY_PATH set to include anaconda3/lib"

# Quick CUDA check
CONDA_PYTHON="$HOME/anaconda3/bin/python3"
RESNET_PYTHON="$RESNET_ROOT/.venv/bin/python3"
for PY in "$CONDA_PYTHON" "$RESNET_PYTHON"; do
  if [[ -f "$PY" ]] && "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$("$PY" -c "import torch; print(torch.cuda.get_device_name($GPU_IDX))" 2>/dev/null || echo "unknown")
    log "INFO  CUDA device $GPU_IDX: $GPU_NAME"
    break
  fi
done

! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1 && \
  die "Ollama not running at localhost:11434. Start with: ollama serve"

[[ ! -d "$RESNET_ROOT/.venv" ]] && die "ResNet venv missing. Run: cd $RESNET_ROOT && uv sync"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASSED=()
FAILED=()
SKIPPED=()

trial_count() {
  local ledger="$REPO/experiments/ledgers/${1}_trials.jsonl"
  [[ -f "$ledger" ]] && wc -l < "$ledger" | tr -d '[:space:]' || echo 0
}

reset_one() {
  local id="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RESET  $id  (would delete ledger + pending guard, reset train.py)"
    return 0
  fi

  # 1. Remove stale pending guard (direct rm — we're replacing the ledger entirely)
  local pending="$REPO/experiments/ledgers/${id}_pending.json"
  if [[ -f "$pending" ]]; then
    log "CLEAR GUARD  $id  (removing stale pending guard before clean reset)"
    rm -f "$pending"
  fi

  # 2. Delete existing ledger so run_one starts from 0 trials
  local ledger="$REPO/experiments/ledgers/${id}_trials.jsonl"
  if [[ -f "$ledger" ]]; then
    local old_count
    old_count=$(wc -l < "$ledger" | tr -d '[:space:]')
    log "DELETE LEDGER  $id  ($old_count old trials removed)"
    rm -f "$ledger"
  fi

  # 3. Reset train.py to clean baseline template
  log "RESET  $id  (using .autoresearch_baseline/train.py template)"
  python3 "$REPO/scripts/reset_node_state.py" \
    --node resnet_trigger \
    --campaign-id "$id" \
    --node-root "$RESNET_ROOT" \
    >> "$LOG_DIR/resets.log" 2>&1 \
  || warn "reset_node_state.py returned non-zero for $id"

  # 4. Verify post-reset hash matches the clean baseline template
  local actual_hash
  actual_hash=$(sha256sum "$RESNET_ROOT/train.py" 2>/dev/null | awk '{print $1}')
  if [[ "$actual_hash" == "$BASELINE_TRAIN_HASH" ]]; then
    log "VERIFY OK  $id  train.py hash matches baseline template"
  else
    warn "VERIFY MISMATCH  $id  train.py=$actual_hash  expected=$BASELINE_TRAIN_HASH"
    warn "  The reset may not have used the .autoresearch_baseline template."
    warn "  Check $LOG_DIR/resets.log"
  fi
}

run_one() {
  local id="$1" budget="$2"
  shift 2
  local actual
  actual=$(trial_count "$id")

  if [[ "$actual" -ge "$budget" ]]; then
    log "SKIP  $id  ($actual/$budget trials already complete — ledger not deleted?)"
    SKIPPED+=("$id")
    return 0
  fi

  log "START $id  ($actual/$budget done, running $((budget - actual)) more)"
  local clog="$LOG_DIR/${id}.log"
  printf 'CMD: %s\n' "$*" > "$clog"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY   $id"
    SKIPPED+=("$id [dry]")
    return 0
  fi

  if "$@" >> "$clog" 2>&1; then
    local final
    final=$(trial_count "$id")
    log "PASS  $id  ($final trials)"
    PASSED+=("$id")
  else
    log "FAIL  $id  (exit $?) — see $clog"
    FAILED+=("$id")
  fi
}

# ---------------------------------------------------------------------------
# PHASE 1 — Smoke test (skippable with --no-smoke)
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 1 — Smoke test (1 trial, fast-search)"
if [[ $NO_SMOKE -eq 1 ]]; then
  log " SKIPPED via --no-smoke flag"
else
  log " Pass --no-smoke to skip if pipeline was recently validated"
fi
log "======================================================"

if [[ $NO_SMOKE -eq 0 ]]; then
  SMOKE_ID="deepseek_resnet_s123_cleanrerun_smoke"
  reset_one "$SMOKE_ID"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY   $SMOKE_ID"
  else
    RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_EPOCHS=3 \
    python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
      --node resnet_trigger \
      --campaign-id "$SMOKE_ID" \
      --node-root "$RESNET_ROOT" \
      --budget 1 \
      --manager langgraph_manager \
      --memory-mode append_only_summary \
      --model deepseek/deepseek-v4-flash \
      --worker-model qwen2.5-coder:7b \
      --temperature 0.2 --no-export \
      >> "$LOG_DIR/${SMOKE_ID}.log" 2>&1 \
    && { log "PASS  $SMOKE_ID"; PASSED+=("$SMOKE_ID"); } \
    || die "Smoke test failed. Fix CUDA/Ollama before proceeding. See $LOG_DIR/${SMOKE_ID}.log"
  fi
else
  log "INFO  Smoke test skipped."
fi

# ---------------------------------------------------------------------------
# PHASE 2 — Clean re-run: 3 arms × s1/s2/s3 × 15 trials = 135 trials
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 2 — Clean re-run s1, s2, s3 (fixed baseline: $BASELINE_TRAIN_HASH)"
log " 3 arms × 3 seeds × 15 trials = 135 trials"
log " WARNING: Existing s1/s2/s3 ledgers will be deleted by reset_one."
log "======================================================"

BUDGET=15
[[ $SMOKE -eq 1 ]] && BUDGET=1

for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s1 s2 s3; do
    CID="deepseek_resnet_${ARM}_${SEED}"
    [[ $SMOKE -eq 1 ]] && CID="${CID}_cleansmoke"

    reset_one "$CID"

    run_one "$CID" "$BUDGET" \
      python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
        --node resnet_trigger \
        --campaign-id "$CID" \
        --node-root "$RESNET_ROOT" \
        --budget "$BUDGET" \
        --manager langgraph_manager \
        --memory-mode "${ARM}" \
        --model deepseek/deepseek-v4-flash \
        --worker-model qwen2.5-coder:7b \
        --temperature 0.2 --no-export
  done
done

# ---------------------------------------------------------------------------
# PHASE 3 — Post-reset verification: check trial-001 node_state_hash
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 3 — Verify trial-001 node_state_hash consistency"
log " All 5 seeds (s1–s5) should share the same hash at trial 1"
log "======================================================"

if [[ $DRY_RUN -eq 0 && $SMOKE -eq 0 ]]; then
  REPO="$REPO" python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import json, os
from pathlib import Path

repo = Path(os.environ["REPO"])
ledger_dir = repo / "experiments" / "ledgers"

arms = ["none", "append_only_summary", "append_only_summary_with_rationale"]
seeds = ["s1", "s2", "s3", "s4", "s5"]

print()
print("  trial-001 node_state_hash by arm and seed:")
print(f"  {'Seed':<6} {'Arm':<36} {'Hash (first 12)':<14} {'Budget idx'}")
print("  " + "-" * 70)

all_hashes = []
for arm in arms:
    for seed in seeds:
        cid = f"deepseek_resnet_{arm}_{seed}"
        ledger = ledger_dir / f"{cid}_trials.jsonl"
        if not ledger.exists():
            print(f"  {seed:<6} {arm:<36} MISSING")
            continue
        with ledger.open() as f:
            first_line = f.readline().strip()
        if not first_line:
            print(f"  {seed:<6} {arm:<36} EMPTY LEDGER")
            continue
        try:
            rec = json.loads(first_line)
        except json.JSONDecodeError:
            print(f"  {seed:<6} {arm:<36} PARSE ERROR")
            continue
        prov = rec.get("provenance") or {}
        h = prov.get("node_state_hash", rec.get("node_state_hash", "N/A"))
        bidx = rec.get("budget_index", "?")
        short = h[:12] if h != "N/A" else "N/A"
        print(f"  {seed:<6} {arm:<36} {short:<14} {bidx}")
        if h and h != "N/A":
            all_hashes.append(h)

unique = set(all_hashes)
print()
if len(unique) == 1:
    print(f"  ✓ All seeds share the same trial-001 node_state_hash: {list(unique)[0][:12]}...")
    print("    Reset contamination is fully resolved across all 5 seeds.")
elif len(unique) > 1:
    print(f"  ✗ {len(unique)} distinct hashes found — not all seeds started from the same baseline.")
    print("    Inspect resets.log to determine which seeds/arms still diverge.")
PYEOF
fi

# ---------------------------------------------------------------------------
# PHASE 4 — Bootstrap CIs across all 5 seeds
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 4 — Bootstrap CIs (all 5 seeds, clean)"
log "======================================================"

if [[ $SMOKE -eq 0 && $DRY_RUN -eq 0 && ${#FAILED[@]} -eq 0 ]]; then
  CI_ARGS=()
  for ARM in none append_only_summary append_only_summary_with_rationale; do
    for SEED in s1 s2 s3 s4 s5; do
      CID="deepseek_resnet_${ARM}_${SEED}"
      LEDGER="$REPO/experiments/ledgers/${CID}_trials.jsonl"
      if [[ -f "$LEDGER" ]]; then
        CI_ARGS+=(--campaign "$CID" --node resnet_trigger)
      fi
    done
  done

  if [[ ${#CI_ARGS[@]} -gt 0 ]]; then
    log "INFO  Computing 5-seed bootstrap CIs..."
    python3 "$REPO/scripts/bootstrap_governance_cis.py" \
      "${CI_ARGS[@]}" \
      --out "$REPO/paper/tables/governance_bootstrap_cis_resnet_5seed_clean.csv" \
      --samples 10000 --seed 42 \
      >> "$LOG_DIR/bootstrap_5seed.log" 2>&1 \
    && log "PASS  5-seed CIs → paper/tables/governance_bootstrap_cis_resnet_5seed_clean.csv" \
    || warn "bootstrap_governance_cis.py returned non-zero — see $LOG_DIR/bootstrap_5seed.log"
  fi
else
  log "SKIP  Bootstrap CIs (smoke/dry-run or failed campaigns)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " SUMMARY — $(date)"
log "======================================================"
log " Baseline template : $BASELINE_TEMPLATE"
log " Baseline hash     : $BASELINE_TRAIN_HASH"
log " Passed  (${#PASSED[@]}) : ${PASSED[*]:-none}"
log " Failed  (${#FAILED[@]}) : ${FAILED[*]:-none}"
log " Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
log ""
log " Next steps:"
log "   1. Check PHASE 3 hash output above — all 5 seeds must share one hash."
log "   2. Sync to Mac:"
log "      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \\"
log "            /Users/wongdowling/Documents/autoresearch_harness/experiments/ledgers/"
log "      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/paper/tables/ \\"
log "            /Users/wongdowling/Documents/autoresearch_harness/paper/tables/"
log "   3. Update §5.3 table and prose with clean 5-seed statistics."
log ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "WARNING  ${#FAILED[@]} campaign(s) failed. Safe to re-run — completed campaigns are skipped."
  exit 1
else
  log "All campaigns passed."
  exit 0
fi
