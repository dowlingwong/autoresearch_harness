#!/usr/bin/env bash
# =============================================================================
# run_linux_resnet_s4s5.sh — Add seeds s4 and s5 to the ResNet L40S ablation
# =============================================================================
#
# Extends the existing 3-seed (s1/s2/s3) ablation with 2 new seeds at budget=15,
# keeping budget consistent with existing seeds for clean 5-seed bootstrap CIs.
#
# Runs: 3 arms × 2 new seeds × 15 trials = 90 new trials
# Est.: ~12 hours sequential on L40S (based on observed avg ~479s/trial)
#
# USAGE (from repo root on the Linux server):
#   export DEEPSEEK_API_KEY=sk-...
#   export DEEPSEEK_THINKING=disabled
#   bash scripts/run_linux_resnet_s4s5.sh 2>&1 | tee logs/resnet_s4s5_$(date +%Y%m%d).log
#
# OPTIONS:
#   --gpu N       CUDA device index (default: 0)
#   --fast        Fast-search mode (~3-4x faster, less accurate — smoke only)
#   --smoke       1 trial per campaign for pipeline check
#   --dry-run     Print commands without executing
#   --no-reset    Skip node reset step
#
# PREREQUISITES (same as run_linux_resnet.sh):
#   1. export DEEPSEEK_API_KEY=sk-...
#   2. ollama serve  (running in background)
#   3. ollama pull qwen2.5-coder:7b
#   4. cd nodes/ResNet_trigger && uv sync
#   5. uv sync  (at repo root)
#
# WHAT THIS ADDS vs. EXISTING 3-SEED RUN:
#   New campaigns: deepseek_resnet_{none,summary,rationale}_s4
#                  deepseek_resnet_{none,summary,rationale}_s5
#   Existing campaigns (s1/s2/s3) are untouched — run_one skips completed ones.
#
# AFTER THIS COMPLETES:
#   Sync back to Mac:
#     rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \
#           /Users/wongdowling/Documents/autoresearch_harness/experiments/ledgers/
#   Then run on Mac: python3 scripts/compute_resnet_5seed_cis.py
#
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
GPU_IDX=0
FAST=0
SMOKE=0
NO_RESET=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)      GPU_IDX="$2"; shift 2 ;;
    --gpu=*)    GPU_IDX="${1#--gpu=}"; shift ;;
    --fast)     FAST=1; shift ;;
    --smoke)    SMOKE=1; shift ;;
    --no-reset) NO_RESET=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  sed -n '2,45p' "$0" | sed 's/^# \?//'; exit 0 ;;
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
LOG_DIR="$REPO/logs/resnet_s4s5_$RUN_TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
warn() { log "WARNING  $*"; }
die()  { log "ERROR    $*"; exit 1; }

log "============================================================"
log " ResNet s4+s5 extension runner started: $(date)"
log " Repo  : $REPO"
log " GPU   : CUDA:$GPU_IDX"
log " Fast  : $FAST | Smoke: $SMOKE | No-reset: $NO_RESET | Dry: $DRY_RUN"
log "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && die "DEEPSEEK_API_KEY is not set."

export CUDA_VISIBLE_DEVICES="$GPU_IDX"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"

# Ensure libcudnn.so.9 is resolvable when the conda python trains the model.
# The .venv doesn't bundle CUDA libraries; they live in $HOME/anaconda3/lib/.
# Without this, every trial crashes immediately with:
#   ImportError: libcudnn.so.9: cannot open shared object file
export LD_LIBRARY_PATH="${HOME}/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
log "INFO  LD_LIBRARY_PATH set to include anaconda3/lib"

if [[ $FAST -eq 1 ]]; then
  export RESNET_TRIGGER_FAST_SEARCH=1
  export RESNET_TRIGGER_FAST_EPOCHS=3
fi

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
  if [[ $NO_RESET -eq 1 || $DRY_RUN -eq 1 ]]; then
    log "SKIP-RESET  $id"
    return 0
  fi
  log "RESET $id"
  python3 "$REPO/scripts/reset_node_state.py" \
    --node resnet_trigger \
    --campaign-id "$id" \
    --node-root "$RESNET_ROOT" \
    >> "$LOG_DIR/resets.log" 2>&1 \
  || warn "reset_node_state.py returned non-zero for $id"
}

run_one() {
  local id="$1" budget="$2"
  shift 2
  local actual
  actual=$(trial_count "$id")

  if [[ "$actual" -ge "$budget" ]]; then
    log "SKIP  $id  ($actual/$budget trials already complete)"
    SKIPPED+=("$id")
    return 0
  fi

  log "START $id  ($actual/$budget done, running $((budget - actual)) more)"
  local clog="$LOG_DIR/${id}.log"
  printf 'CMD: %s\n' "$*" > "$clog"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY   $id  →  $*"
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
# PHASE 1 — Quick smoke test (1 trial, fast mode — skip if confident)
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 1 — Smoke test (1 trial, fast-search)"
log " Skip with --no-reset if pipeline was recently validated"
log "======================================================"

SMOKE_ID="deepseek_resnet_s4s5_smoke"
reset_one "$SMOKE_ID"

SMOKE_CMD=(
  python3 "$REPO/scripts/run_kdd_memory_ablation.py"
    --node resnet_trigger
    --campaign-id "$SMOKE_ID"
    --node-root "$RESNET_ROOT"
    --budget 1
    --manager langgraph_manager
    --memory-mode append_only_summary
    --model deepseek/deepseek-v4-flash
    --worker-model qwen2.5-coder:7b
    --temperature 0.2 --no-export
)

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY   $SMOKE_ID"
elif RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_EPOCHS=3 \
     "${SMOKE_CMD[@]}" >> "$LOG_DIR/${SMOKE_ID}.log" 2>&1; then
  log "PASS  $SMOKE_ID"
  PASSED+=("$SMOKE_ID")
else
  log "FAIL  $SMOKE_ID — see $LOG_DIR/${SMOKE_ID}.log"
  die "Smoke test failed. Fix CUDA/Ollama before running full ablation."
fi

# ---------------------------------------------------------------------------
# PHASE 2 — New seeds: 3 arms × s4 + s5 × budget 15 = 90 trials
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 2 — New seeds s4 and s5 (3 arms × 2 seeds × 15 trials)"
log "======================================================"

BUDGET=15
[[ $SMOKE -eq 1 ]] && BUDGET=1

for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s4 s5; do
    CID="deepseek_resnet_${ARM}_${SEED}"
    [[ $SMOKE -eq 1 ]] && CID="${CID}_smoke"

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
# PHASE 3 — Bootstrap CIs across all 5 seeds (s1–s5)
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 3 — Bootstrap CIs (5 seeds)"
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
      --out "$REPO/paper/tables/governance_bootstrap_cis_resnet_5seed.csv" \
      --samples 10000 --seed 42 \
      >> "$LOG_DIR/bootstrap_5seed.log" 2>&1 \
    && log "PASS  5-seed CIs → paper/tables/governance_bootstrap_cis_resnet_5seed.csv" \
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
log " Passed  (${#PASSED[@]}) : ${PASSED[*]:-none}"
log " Failed  (${#FAILED[@]}) : ${FAILED[*]:-none}"
log " Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
log ""
log " Next steps:"
log "   1. Sync ledgers to Mac:"
log "      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \\"
log "            /Users/wongdowling/Documents/autoresearch_harness/experiments/ledgers/"
log "   2. Update paper table (§5.3) with 5-seed CIs from:"
log "      paper/tables/governance_bootstrap_cis_resnet_5seed.csv"
log "   3. Update §5.3 prose: change '3 seeds' → '5 seeds', update CI widths"
log ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "WARNING  ${#FAILED[@]} campaign(s) failed. Safe to re-run — completed campaigns skipped."
  exit 1
else
  log "All campaigns passed."
  exit 0
fi
