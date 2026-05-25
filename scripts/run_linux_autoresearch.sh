#!/usr/bin/env bash
# =============================================================================
# run_linux_autoresearch.sh — autoresearch_linux campaign runner for deepthought2
# =============================================================================
#
# Runs 2 arms (none, append_only_summary) × 4 seeds × budget 30 on the
# autoresearch_linux node (CUDA, L40S). Each trial runs train.py for a
# fixed 5-minute wall-clock budget.
# Total: 240 trials, ~20 hr.
#
# USAGE (from repo root, after conda deactivate):
#   export DEEPSEEK_API_KEY=sk-...
#   export PYTHONPATH=/ceph/dwong/autoresearch_harness/src:$PYTHONPATH
#   bash scripts/run_linux_autoresearch.sh 2>&1 | tee logs/autoresearch_$(date +%Y%m%d).log
#
# MONITOR LIVE (separate terminal):
#   tail -f logs/linux_autoresearch_<timestamp>/master.log
#   tail -f logs/linux_autoresearch_<timestamp>/deepseek_autoresearch_linux_none_s1.log
#
# OPTIONS:
#   --gpu=N      CUDA device index (default: 0)
#   --budget=N   Trials per seed per arm (default: 30)
#   --seeds=N    Number of seeds (default: 4, runs s1..sN)
#   --smoke      1 trial only — pipeline validation, then exit
#   --dry-run    Print commands without executing
#
# PREREQUISITES:
#   1. conda deactivate  (clean shell)
#   2. export DEEPSEEK_API_KEY=sk-...
#   3. export PYTHONPATH=/ceph/dwong/autoresearch_harness/src:$PYTHONPATH
#   4. Ollama running with qwen2.5-coder:7b pulled
#   5. Data: cd nodes/autoresearch-macos && ~/anaconda3/bin/python3 prepare.py
#
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
GPU_IDX=0
BUDGET=30
N_SEEDS=4
SMOKE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu=*)    GPU_IDX="${1#--gpu=}"; shift ;;
    --gpu)      GPU_IDX="$2"; shift 2 ;;
    --budget=*) BUDGET="${1#--budget=}"; shift ;;
    --budget)   BUDGET="$2"; shift 2 ;;
    --seeds=*)  N_SEEDS="${1#--seeds=}"; shift ;;
    --seeds)    N_SEEDS="$2"; shift 2 ;;
    --smoke)    SMOKE=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  sed -n '2,35p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

ARMS=("none" "append_only_summary")
[[ $SMOKE -eq 1 ]] && BUDGET=1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
AR_ROOT="$REPO/nodes/autoresearch-macos"
CONDA_PYTHON="$HOME/anaconda3/bin/python3"

# ---------------------------------------------------------------------------
# Log setup
# ---------------------------------------------------------------------------
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$REPO/logs/linux_autoresearch_$RUN_TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
warn() { log "WARNING  $*"; }
die()  { log "ERROR    $*"; exit 1; }

log "============================================================"
log " autoresearch Linux runner started: $(date)"
log " Repo     : $REPO"
log " Node dir : $AR_ROOT"
log " GPU      : CUDA:$GPU_IDX"
log " Arms     : ${ARMS[*]}"
log " Seeds    : $N_SEEDS  |  Budget: $BUDGET/seed/arm  |  Smoke: $SMOKE"
log " Log dir  : $LOG_DIR"
log "------------------------------------------------------------"
log " MONITOR:  tail -f $MASTER_LOG"
log " PER-CAMPAIGN: tail -f $LOG_DIR/<campaign_id>.log"
log "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight checks — all checked BEFORE any campaign starts
# ---------------------------------------------------------------------------
log ""
log "--- Pre-flight checks ---"

[[ -z "${DEEPSEEK_API_KEY:-}" ]] && die "DEEPSEEK_API_KEY is not set. Export it before running."
[[ -z "${PYTHONPATH:-}" ]] && warn "PYTHONPATH not set — run: export PYTHONPATH=$REPO/src:\$PYTHONPATH"

[[ ! -f "$CONDA_PYTHON" ]] && die "Conda python not found: $CONDA_PYTHON"

if ! "$CONDA_PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  die "Conda Python has no CUDA. Check driver and torch installation."
fi
GPU_NAME=$("$CONDA_PYTHON" -c "import torch; print(torch.cuda.get_device_name($GPU_IDX))" 2>/dev/null || echo "unknown")
log "OK   CUDA device $GPU_IDX: $GPU_NAME"

if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  die "Ollama not running at localhost:11434. Start it first:
       OLLAMA_MODELS=/ceph/dwong/ollama_models OLLAMA_HOST=127.0.0.1:11434 \\
         /ceph/dwong/ollama_bin/bin/ollama serve > /ceph/dwong/ollama.log 2>&1 &"
fi
log "OK   Ollama reachable at localhost:11434"

if ! uv run python3 -c "from autoresearch.common.paths import REPO_ROOT; assert str(REPO_ROOT) == '$REPO'" 2>/dev/null; then
  die "REPO_ROOT mismatch — run: export PYTHONPATH=$REPO/src:\$PYTHONPATH and retry."
fi
log "OK   REPO_ROOT = $REPO"

if [[ ! -d "$AR_ROOT/data" ]]; then
  log "INFO data/ not found — running prepare.py (~10 min)..."
  if [[ $DRY_RUN -eq 0 ]]; then
    cd "$AR_ROOT"
    uv sync >> "$LOG_DIR/prepare.log" 2>&1 \
      || die "uv sync failed in node dir — see $LOG_DIR/prepare.log"
    uv run python3 prepare.py >> "$LOG_DIR/prepare.log" 2>&1 \
      || die "prepare.py failed — see $LOG_DIR/prepare.log"
    cd "$REPO"
    log "OK   prepare.py complete"
  else
    log "DRY  prepare.py would run here"
  fi
else
  log "OK   data/ found at $AR_ROOT/data"
fi

log "--- All checks passed ---"
log ""

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

run_campaign() {
  local id="$1" budget="$2"
  shift 2
  local actual
  actual=$(trial_count "$id")

  if [[ "$actual" -ge "$budget" ]]; then
    log "SKIP $id  ($actual/$budget trials already complete)"
    SKIPPED+=("$id")
    return 0
  fi

  local remaining=$((budget - actual))
  log "START $id  ($actual done, running $remaining more)"
  log "      → log: $LOG_DIR/${id}.log"

  local clog="$LOG_DIR/${id}.log"
  echo "CMD: $*" > "$clog"
  echo "Started: $(date)" >> "$clog"
  echo "---" >> "$clog"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY  $id"
    SKIPPED+=("$id [dry]")
    return 0
  fi

  if CUDA_VISIBLE_DEVICES="$GPU_IDX" "$@" >> "$clog" 2>&1; then
    local final
    final=$(trial_count "$id")
    log "PASS $id  ($final/$budget trials)"
    PASSED+=("$id")
  else
    local rc=$?
    log "FAIL $id  (exit $rc) — see $LOG_DIR/${id}.log"
    log "     Last 5 lines:"
    tail -5 "$clog" | while IFS= read -r line; do log "     | $line"; done
    FAILED+=("$id")
  fi
}

# ---------------------------------------------------------------------------
# PHASE 1 — Smoke test: 1 trial to validate full pipeline
# ---------------------------------------------------------------------------
log "======================================================"
log " PHASE 1 — Smoke test (1 trial, arm=none)"
log "======================================================"

SMOKE_ID="deepseek_autoresearch_linux_smoke"

if [[ $(trial_count "$SMOKE_ID") -ge 1 ]]; then
  log "SKIP smoke (already done)"
else
  log "START $SMOKE_ID"
  SMOKE_LOG="$LOG_DIR/${SMOKE_ID}.log"
  echo "Smoke test started: $(date)" > "$SMOKE_LOG"

  SMOKE_CMD=(
    uv run python3 "$REPO/scripts/run_kdd_memory_ablation.py"
      --node autoresearch_linux
      --campaign-id "$SMOKE_ID"
      --node-root "$AR_ROOT"
      --budget 1
      --manager langgraph_manager
      --memory-mode none
      --model deepseek/deepseek-v4-flash
      --worker-model qwen2.5-coder:7b
      --temperature 0.2 --no-export
  )

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY  smoke"
  elif CUDA_VISIBLE_DEVICES="$GPU_IDX" "${SMOKE_CMD[@]}" >> "$SMOKE_LOG" 2>&1; then
    log "PASS smoke — pipeline validated"
    log "     Run log: tail $LOG_DIR/${SMOKE_ID}.log"
    PASSED+=("$SMOKE_ID")
  else
    log "FAIL smoke (exit $?) — see $SMOKE_LOG"
    log "     Last 10 lines:"
    tail -10 "$SMOKE_LOG" | while IFS= read -r line; do log "     | $line"; done
    die "Smoke test failed. Fix the issue before running full campaigns."
  fi
fi

[[ $SMOKE -eq 1 ]] && { log "Smoke-only mode — exiting."; exit 0; }

# ---------------------------------------------------------------------------
# PHASE 2 — Full campaigns: 2 arms × N_SEEDS × BUDGET
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 2 — Full campaigns"
log " Arms: ${ARMS[*]}"
log " Seeds: s1..s$N_SEEDS | Budget: $BUDGET trials each"
log " Total: $((${#ARMS[@]} * N_SEEDS * BUDGET)) trials"
log " ETA:   ~$((${#ARMS[@]} * N_SEEDS * BUDGET * 5 / 60)) hr"
log "======================================================"

for ARM in "${ARMS[@]}"; do
  log ""
  log "--- Arm: $ARM ---"
  for i in $(seq 1 "$N_SEEDS"); do
    SEED="s$i"
    CID="deepseek_autoresearch_linux_${ARM}_${SEED}"

    run_campaign "$CID" "$BUDGET" \
      uv run python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
        --node autoresearch_linux \
        --campaign-id "$CID" \
        --node-root "$AR_ROOT" \
        --budget "$BUDGET" \
        --manager langgraph_manager \
        --memory-mode "$ARM" \
        --model deepseek/deepseek-v4-flash \
        --worker-model qwen2.5-coder:7b \
        --temperature 0.2 --no-export
  done
done

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
log " Logs: $LOG_DIR/"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "WARNING  ${#FAILED[@]} campaign(s) failed."
  log "         Re-running is safe — completed campaigns are skipped."
  exit 1
else
  log "All campaigns complete."
  log "Ledgers: $REPO/experiments/ledgers/deepseek_autoresearch_linux_*"
  exit 0
fi
