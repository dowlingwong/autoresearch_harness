#!/usr/bin/env bash
# =============================================================================
# run_linux_resnet.sh — ResNet ablation runner for Linux / NVIDIA GPU
# =============================================================================
#
# Resets and re-runs all 9 stale ResNet campaigns on the L40S (or any CUDA GPU).
# Each campaign had 15 trials that all failed overnight due to MPS instability
# on macOS. On CUDA this is resolved.
#
# USAGE (from repo root on the Linux server):
#   export DEEPSEEK_API_KEY=sk-...
#   export DEEPSEEK_THINKING=disabled
#   bash scripts/run_linux_resnet.sh 2>&1 | tee logs/linux_resnet_$(date +%Y%m%d).log
#
# OPTIONS:
#   --gpu N         CUDA device index to pin (default: 0)
#   --fast          Run in fast-search mode (fewer epochs, ~3-4x faster, less accurate)
#   --smoke         1 trial only per campaign — pipeline validation
#   --no-reset      Skip the reset step (re-run from current ledger state)
#   --dry-run       Print commands without executing
#
# PREREQUISITES:
#   1. export DEEPSEEK_API_KEY=sk-...
#   2. Ollama running: ollama serve  (in a separate terminal or as a service)
#   3. Worker model pulled: ollama pull qwen2.5-coder:7b
#   4. ResNet venv: cd nodes/ResNet_trigger && uv sync
#   5. Project venv: uv sync  (at repo root)
#
# ESTIMATED RUNTIME (L40S 46 GB, full GPU mode):
#   Smoke test (1 trial)         ~  5 min
#   Full ablation (9 × 15 = 135) ~  5-8 hr  (CUDA is much faster than MPS)
#   With --fast                  ~  2-3 hr
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
    --gpu)        GPU_IDX="$2"; shift 2 ;;
    --gpu=*)      GPU_IDX="${1#--gpu=}"; shift ;;
    --fast)       FAST=1; shift ;;
    --smoke)      SMOKE=1; shift ;;
    --no-reset)   NO_RESET=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --help|-h)    sed -n '2,40p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RESNET_ROOT="$REPO/nodes/ResNet_trigger"

# Activate project venv
VENV="$REPO/.venv/bin/activate"
if [[ -f "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV"
else
  echo "[ERROR] No .venv at $REPO/.venv — run: cd $REPO && uv sync"
  exit 1
fi

# ---------------------------------------------------------------------------
# Log setup
# ---------------------------------------------------------------------------
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$REPO/logs/linux_resnet_$RUN_TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
warn() { log "WARNING  $*"; }
die()  { log "ERROR    $*"; exit 1; }

log "============================================================"
log " Linux ResNet ablation runner started: $(date)"
log " Repo  : $REPO"
log " Node  : $RESNET_ROOT"
log " GPU   : CUDA:$GPU_IDX"
log " Fast  : $FAST | Smoke: $SMOKE | No-reset: $NO_RESET | Dry: $DRY_RUN"
log "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && die "DEEPSEEK_API_KEY is not set. Aborting."

# Use conda python if it has working CUDA, otherwise fall back to resnet venv
CONDA_PYTHON="$HOME/anaconda3/bin/python3"
RESNET_PYTHON="$RESNET_ROOT/.venv/bin/python3"

if [[ -f "$CONDA_PYTHON" ]] && "$CONDA_PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  CHECK_PYTHON="$CONDA_PYTHON"
  log "INFO  Using conda Python for CUDA check: $CONDA_PYTHON"
elif [[ -f "$RESNET_PYTHON" ]] && "$RESNET_PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  CHECK_PYTHON="$RESNET_PYTHON"
  log "INFO  Using ResNet venv Python for CUDA check: $RESNET_PYTHON"
else
  die "No Python with working CUDA found. Tried: $CONDA_PYTHON, $RESNET_PYTHON"
fi

GPU_NAME=$("$CHECK_PYTHON" -c "import torch; print(torch.cuda.get_device_name($GPU_IDX))" 2>/dev/null || echo "unknown")
log "INFO  CUDA device $GPU_IDX: $GPU_NAME"

if [[ ! -d "$RESNET_ROOT/.venv" ]]; then
  die "ResNet venv missing. Run: cd $RESNET_ROOT && uv sync"
fi

if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama does not appear to be running at localhost:11434."
  warn "Start it with: ollama serve"
  warn "Pull the worker model: ollama pull qwen2.5-coder:7b"
  die "Cannot proceed without Ollama (needed for the ClawWorker model)."
fi

# ---------------------------------------------------------------------------
# Environment: pin GPU, enable fast search if requested
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU_IDX"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"

if [[ $FAST -eq 1 ]]; then
  export RESNET_TRIGGER_FAST_SEARCH=1
  export RESNET_TRIGGER_FAST_EPOCHS=3
  log "INFO  Fast-search mode enabled (RESNET_TRIGGER_FAST_SEARCH=1, 3 epochs)"
fi

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

  log "START $id  ($actual/$budget done; running $((budget - actual)) more)"
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

reset_one() {
  local id="$1"
  if [[ $NO_RESET -eq 1 ]]; then
    log "SKIP-RESET  $id  (--no-reset set)"
    return 0
  fi
  log "RESET $id"
  if [[ $DRY_RUN -eq 0 ]]; then
    python3 "$REPO/scripts/reset_node_state.py" \
      --node resnet_trigger \
      --campaign-id "$id" \
      --node-root "$RESNET_ROOT" \
      >> "$LOG_DIR/resets.log" 2>&1 \
    || warn "reset_node_state.py returned non-zero for $id (check $LOG_DIR/resets.log)"
  fi
}

# ---------------------------------------------------------------------------
# PHASE 1 — Smoke test: 1 trial to validate CUDA pipeline
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 1 — Smoke test (1 trial, fast-search)"
log "======================================================"

SMOKE_ID="deepseek_resnet_smoke"
SMOKE_PASS=0

reset_one "$SMOKE_ID"

log "START $SMOKE_ID"
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
  SMOKE_PASS=1
elif RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_EPOCHS=3 \
     "${SMOKE_CMD[@]}" >> "$LOG_DIR/${SMOKE_ID}.log" 2>&1; then
  log "PASS  $SMOKE_ID"
  PASSED+=("$SMOKE_ID")
  SMOKE_PASS=1
else
  log "FAIL  $SMOKE_ID — see $LOG_DIR/${SMOKE_ID}.log"
  FAILED+=("$SMOKE_ID")
  die "Smoke test failed. Fix the CUDA/ClawWorker issue before running the full ablation."
fi

# ---------------------------------------------------------------------------
# PHASE 2 — Full ablation: 3 arms × 3 seeds × budget 15
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 2 — Full ablation (3 arms x 3 seeds x 15 trials)"
log "======================================================"

BUDGET=15
[[ $SMOKE -eq 1 ]] && BUDGET=1

for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s1 s2 s3; do
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
# PHASE 3 — Bootstrap CIs for ResNet (after all campaigns complete)
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 3 — Bootstrap CIs"
log "======================================================"

if [[ $SMOKE -eq 0 && $DRY_RUN -eq 0 && ${#FAILED[@]} -eq 0 ]]; then
  CI_ARGS=()
  for ARM in none append_only_summary append_only_summary_with_rationale; do
    for SEED in s1 s2 s3; do
      CID="deepseek_resnet_${ARM}_${SEED}"
      LEDGER="$REPO/experiments/ledgers/${CID}_trials.jsonl"
      if [[ -f "$LEDGER" ]]; then
        CI_ARGS+=(--campaign "$CID" --node resnet_trigger)
      fi
    done
  done

  if [[ ${#CI_ARGS[@]} -gt 0 ]]; then
    log "INFO  Computing bootstrap CIs for ResNet ablation..."
    python3 "$REPO/scripts/bootstrap_governance_cis.py" \
      "${CI_ARGS[@]}" \
      --out "$REPO/paper/tables/governance_bootstrap_cis_resnet.csv" \
      --samples 10000 --seed 42 \
      >> "$LOG_DIR/bootstrap_resnet.log" 2>&1 \
    && log "PASS  bootstrap CIs saved → paper/tables/governance_bootstrap_cis_resnet.csv" \
    || warn "bootstrap CIs returned non-zero — see $LOG_DIR/bootstrap_resnet.log"
  fi
else
  log "SKIP  Bootstrap CIs (smoke/dry-run mode or failed campaigns)"
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

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "WARNING  ${#FAILED[@]} campaign(s) failed. Check logs in $LOG_DIR/"
  log "         Re-running is safe — completed campaigns are skipped."
  exit 1
else
  log "All campaigns passed or were already complete."
  log "Results: experiments/ledgers/deepseek_resnet_*"
  log "Tables:  paper/tables/governance_bootstrap_cis_resnet.csv"
  exit 0
fi
