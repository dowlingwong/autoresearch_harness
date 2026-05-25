#!/usr/bin/env bash
# =============================================================================
# run_phase_e_local.sh — Phase E large-budget local experiments
# =============================================================================
#
# Runs all LocalWorker large-budget campaigns (no GPU, no Ollama required).
# Covers:
#
#   E1  lr_synthetic        budget 30 × 5 seeds × 3 arms  = 450 trials  (~30 min)
#   E2  mlp_synthetic       budget 30 × 3 seeds × 3 arms  = 270 trials  (~15 min)
#   E3  openml_credit_g     budget 30 × 5 seeds            = 150 trials  (~2–3 hr)
#   E4  openml_bank_mkt     budget 30 × 5 seeds            = 150 trials  (~3–5 hr)
#
# Total: 1,020 new trials (on top of existing b10/b20 campaigns).
#
# ResNet and Autoresearch are GPU/Ollama nodes handled on deepthought2 —
# they are intentionally excluded from this script.
#
# USAGE:
#   bash scripts/run_phase_e_local.sh [OPTIONS]
#
# OPTIONS:
#   --dry-run           Print commands without executing
#   --skip-lr           Skip E1 (lr_synthetic)
#   --skip-mlp          Skip E2 (mlp_synthetic)
#   --skip-openml       Skip E3+E4 (both OpenML nodes)
#   --only-lr           Run only E1
#   --only-mlp          Run only E2
#   --only-openml       Run only E3+E4
#   --jobs N            Max parallel campaigns (default: 4)
#                       Recommendation: 4–6 for LR/MLP, 2 for OpenML
#                       (each campaign makes 1 DeepSeek API call per trial)
#   --model MODEL       DeepSeek model string (default: deepseek/deepseek-v4-flash)
#   --temperature T     LLM temperature (default: 0.2)
#
# RESUME SAFETY:
#   Each campaign is skipped automatically if its ledger already has >= the
#   target number of trials. Safe to run again after interruption.
#
# PREREQUISITES:
#   export DEEPSEEK_API_KEY=sk-...
#   export DEEPSEEK_THINKING=disabled
#   uv sync (or .venv activated)
#
# SERVER CAMPAIGNS (ResNet, Autoresearch) are handled by separate scripts:
#   scripts/run_linux_resnet.sh      — ResNet Phase E (deepthought2)
#   scripts/run_linux_autoresearch.sh — Autoresearch Phase E (deepthought2)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRY_RUN=0
SKIP_LR=0
SKIP_MLP=0
SKIP_OPENML=0
ONLY=""
MAX_JOBS=4
MODEL="deepseek/deepseek-v4-flash"
TEMPERATURE=0.2
BUDGET_LR=30
BUDGET_MLP=30
BUDGET_OPENML=30
SEEDS_LR="s1 s2 s3 s4 s5"
SEEDS_MLP="s1 s2 s3"
SEEDS_OPENML="s1 s2 s3 s4 s5"
LR_ARMS="none append_only_summary append_only_summary_with_rationale"
MLP_ARMS="none summary rationale"  # short names used in MLP campaign IDs

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)       DRY_RUN=1 ;;
    --skip-lr)       SKIP_LR=1 ;;
    --skip-mlp)      SKIP_MLP=1 ;;
    --skip-openml)   SKIP_OPENML=1 ;;
    --only-lr)       ONLY="lr" ;;
    --only-mlp)      ONLY="mlp" ;;
    --only-openml)   ONLY="openml" ;;
    --jobs)          MAX_JOBS="$2"; shift ;;
    --model)         MODEL="$2"; shift ;;
    --temperature)   TEMPERATURE="$2"; shift ;;
    *) echo "[WARN] Unknown argument: $1" ;;
  esac
  shift
done

# Apply --only-* flags
[[ "$ONLY" == "lr" ]]    && SKIP_MLP=1 && SKIP_OPENML=1
[[ "$ONLY" == "mlp" ]]   && SKIP_LR=1  && SKIP_OPENML=1
[[ "$ONLY" == "openml" ]] && SKIP_LR=1 && SKIP_MLP=1

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR="$REPO/logs/phase_e_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }
warn() { echo "[$(date +%H:%M:%S)] WARN  $*" | tee -a "$MASTER_LOG" >&2; }

# ---------------------------------------------------------------------------
# Activate virtualenv
# ---------------------------------------------------------------------------
VENV="$REPO/.venv/bin/activate"
if [[ -f "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV"
  log "Virtualenv activated: $REPO/.venv"
else
  warn "No .venv found — assuming uv/system python has the harness installed."
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo ""
  echo "  ERROR: DEEPSEEK_API_KEY is not set."
  echo "  Run:  export DEEPSEEK_API_KEY=sk-..."
  echo "        export DEEPSEEK_THINKING=disabled"
  echo ""
  exit 1
fi

log "============================================================"
log " Phase E — Large-Budget Local Experiments"
log "============================================================"
log " Repo     : $REPO"
log " Log dir  : $LOG_DIR"
log " Model    : $MODEL  temp=$TEMPERATURE"
log " Max jobs : $MAX_JOBS"
log " Budgets  : LR=$BUDGET_LR  MLP=$BUDGET_MLP  OpenML=$BUDGET_OPENML"
log " Dry run  : $DRY_RUN"
log " Skip     : lr=$SKIP_LR  mlp=$SKIP_MLP  openml=$SKIP_OPENML"
log "============================================================"

# ---------------------------------------------------------------------------
# Tracking arrays
# ---------------------------------------------------------------------------
PASSED=()
FAILED=()
SKIPPED=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# trial_count <campaign_id>
trial_count() {
  local ledger="$REPO/experiments/ledgers/${1}_trials.jsonl"
  [[ -f "$ledger" ]] && wc -l < "$ledger" | tr -d '[:space:]' || echo 0
}

# run_campaign <campaign_id> <target_trials> <cmd...>
# Skips if ledger already has >= target_trials lines.
# Logs to $LOG_DIR/<campaign_id>.log
run_campaign() {
  local id="$1"
  local target="$2"
  shift 2
  local actual
  actual=$(trial_count "$id")

  if [[ "$actual" -ge "$target" ]]; then
    log "SKIP  $id  ($actual/$target trials already in ledger)"
    SKIPPED+=("$id")
    return 0
  fi

  local remaining=$(( target - actual ))
  log "START $id  ($actual/$target done; running $remaining more)"
  local clog="$LOG_DIR/${id}.log"
  printf "CMD: %s\n\n" "$*" > "$clog"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY   $id  → $*"
    SKIPPED+=("$id [dry-run]")
    return 0
  fi

  if "$@" >> "$clog" 2>&1; then
    local final
    final=$(trial_count "$id")
    log "PASS  $id  ($final/$target trials in ledger)"
    PASSED+=("$id")
  else
    local exit_code=$?
    log "FAIL  $id  (exit $exit_code) — see $clog"
    FAILED+=("$id")
  fi
}

# run_parallel <max_jobs> <campaign_id> <target> <cmd...>
# Runs up to MAX_JOBS campaigns in parallel using background processes.
# Call flush_jobs at the end of each group to wait for stragglers.
_JOB_PIDS=()
_JOB_IDS=()

run_parallel() {
  local max_jobs="$1"
  local id="$2"
  local target="$3"
  shift 3

  # Wait if at capacity
  while [[ ${#_JOB_PIDS[@]} -ge $max_jobs ]]; do
    local new_pids=()
    local new_ids=()
    for i in "${!_JOB_PIDS[@]}"; do
      local pid="${_JOB_PIDS[$i]}"
      local jid="${_JOB_IDS[$i]}"
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
        new_ids+=("$jid")
      else
        wait "$pid" 2>/dev/null || true
      fi
    done
    _JOB_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    _JOB_IDS=("${new_ids[@]+"${new_ids[@]}"}")
    [[ ${#_JOB_PIDS[@]} -ge $max_jobs ]] && sleep 1
  done

  # Check actual count to decide if we should even launch
  local actual
  actual=$(trial_count "$id")
  if [[ "$actual" -ge "$target" ]]; then
    log "SKIP  $id  ($actual/$target trials already in ledger)"
    SKIPPED+=("$id")
    return 0
  fi

  log "QUEUE $id  ($actual/$target done; running in background)"
  local clog="$LOG_DIR/${id}.log"
  printf "CMD: %s\n\n" "$*" > "$clog"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY   $id  → $*"
    SKIPPED+=("$id [dry-run]")
    return 0
  fi

  (
    if "$@" >> "$clog" 2>&1; then
      local final
      final=$(trial_count "$id")
      log "PASS  $id  ($final/$target trials in ledger)"
    else
      log "FAIL  $id  (exit $?) — see $clog"
    fi
  ) &
  _JOB_PIDS+=("$!")
  _JOB_IDS+=("$id")
}

flush_jobs() {
  if [[ ${#_JOB_PIDS[@]} -eq 0 ]]; then return 0; fi
  log "Waiting for ${#_JOB_PIDS[@]} background jobs to finish..."
  for pid in "${_JOB_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  _JOB_PIDS=()
  _JOB_IDS=()
  log "All background jobs done."
}

# ml_arm_to_memory_mode <arm_short_name>
# Maps short arm names (used in MLP campaign IDs) to --memory-mode values.
arm_to_mode() {
  case "$1" in
    none)      echo "none" ;;
    summary)   echo "append_only_summary" ;;
    rationale) echo "append_only_summary_with_rationale" ;;
    # LR arms are already full names
    *)         echo "$1" ;;
  esac
}

# =============================================================================
# E1 — lr_synthetic (budget 30, 5 seeds × 3 arms = 450 trials)
# =============================================================================
if [[ $SKIP_LR -eq 0 ]]; then
  log ""
  log "------------------------------------------------------------"
  log " E1 — lr_synthetic  budget=$BUDGET_LR  seeds=$SEEDS_LR"
  log "------------------------------------------------------------"

  for ARM in $LR_ARMS; do
    for SEED in $SEEDS_LR; do
      CID="deepseek_lr_${ARM}_b30_${SEED}"
      run_parallel "$MAX_JOBS" "$CID" "$BUDGET_LR" \
        python3 "$REPO/scripts/run_local_node_campaign.py" \
          --node lr_synthetic \
          --campaign-id "$CID" \
          --budget "$BUDGET_LR" \
          --manager langgraph_manager \
          --memory-mode "$ARM" \
          --model "$MODEL" \
          --temperature "$TEMPERATURE"
    done
  done
  flush_jobs

  log "E1 complete."
fi

# =============================================================================
# E2 — mlp_synthetic (budget 30, 3 seeds × 3 arms = 270 trials)
# =============================================================================
if [[ $SKIP_MLP -eq 0 ]]; then
  log ""
  log "------------------------------------------------------------"
  log " E2 — mlp_synthetic  budget=$BUDGET_MLP  seeds=$SEEDS_MLP"
  log "------------------------------------------------------------"

  for ARM in $MLP_ARMS; do
    MODE=$(arm_to_mode "$ARM")
    for SEED in $SEEDS_MLP; do
      CID="deepseek_mlp_${ARM}_b30_${SEED}"
      run_parallel "$MAX_JOBS" "$CID" "$BUDGET_MLP" \
        python3 "$REPO/scripts/run_local_node_campaign.py" \
          --node mlp_synthetic \
          --campaign-id "$CID" \
          --budget "$BUDGET_MLP" \
          --manager langgraph_manager \
          --memory-mode "$MODE" \
          --model "$MODEL" \
          --temperature "$TEMPERATURE"
    done
  done
  flush_jobs

  log "E2 complete."
fi

# =============================================================================
# E3 — openml_credit_g (budget 30, 5 seeds = 150 trials)
# E4 — openml_bank_marketing (budget 30, 5 seeds = 150 trials)
#
# OpenML campaigns run with lower parallelism (default 2) to be kind to the
# DeepSeek API and to avoid sklearn fit contention on shared CPU cores.
# Override with --jobs N if you have headroom.
# =============================================================================
if [[ $SKIP_OPENML -eq 0 ]]; then
  # Use at most 2 concurrent OpenML campaigns regardless of --jobs setting,
  # unless the user explicitly passes --jobs > 2.
  OPENML_JOBS=$(( MAX_JOBS < 2 ? MAX_JOBS : 2 ))

  log ""
  log "------------------------------------------------------------"
  log " E3 — openml_credit_g  budget=$BUDGET_OPENML  seeds=$SEEDS_OPENML"
  log "------------------------------------------------------------"

  for SEED in $SEEDS_OPENML; do
    CID="deepseek_openml_cg_b30_${SEED}"
    run_parallel "$OPENML_JOBS" "$CID" "$BUDGET_OPENML" \
      python3 "$REPO/scripts/run_openml_tabular_campaign.py" \
        --node openml_credit_g \
        --campaign-id "$CID" \
        --budget "$BUDGET_OPENML" \
        --manager langgraph_manager \
        --memory-mode append_only_summary \
        --model "$MODEL" \
        --temperature "$TEMPERATURE"
  done
  flush_jobs
  log "E3 complete."

  log ""
  log "------------------------------------------------------------"
  log " E4 — openml_bank_marketing  budget=$BUDGET_OPENML  seeds=$SEEDS_OPENML"
  log "------------------------------------------------------------"

  for SEED in $SEEDS_OPENML; do
    CID="deepseek_openml_bm_b30_${SEED}"
    run_parallel "$OPENML_JOBS" "$CID" "$BUDGET_OPENML" \
      python3 "$REPO/scripts/run_openml_tabular_campaign.py" \
        --node openml_bank_marketing \
        --campaign-id "$CID" \
        --budget "$BUDGET_OPENML" \
        --manager langgraph_manager \
        --memory-mode append_only_summary \
        --model "$MODEL" \
        --temperature "$TEMPERATURE"
  done
  flush_jobs
  log "E4 complete."
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================================"
log " Phase E Summary"
log "============================================================"
log " PASSED  : ${#PASSED[@]}  — ${PASSED[*]:-none}"
log " FAILED  : ${#FAILED[@]}  — ${FAILED[*]:-none}"
log " SKIPPED : ${#SKIPPED[@]} — (already done or dry-run)"
log ""

# Trial count summary
log " Final trial counts:"
for NODE_PAT in \
    "deepseek_lr_*_b30_*:lr_synthetic b30" \
    "deepseek_mlp_*_b30_*:mlp_synthetic b30" \
    "deepseek_openml_cg_b30_*:openml_credit_g b30" \
    "deepseek_openml_bm_b30_*:openml_bank_mkt b30"; do
  PAT="${NODE_PAT%%:*}"
  LABEL="${NODE_PAT##*:}"
  TOTAL=0
  for f in "$REPO/experiments/ledgers/"${PAT}_trials.jsonl; do
    [[ -f "$f" ]] && TOTAL=$(( TOTAL + $(wc -l < "$f" | tr -d '[:space:]') ))
  done
  log "   $LABEL : $TOTAL trials"
done

log ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  log " ⚠ Some campaigns failed. Check logs in: $LOG_DIR"
  log " Re-run the script — it will skip completed campaigns and retry failed ones."
  exit 1
else
  log " ✓ All Phase E local campaigns complete (or already done)."
  log " Logs: $LOG_DIR"
  log ""
  log " Next steps:"
  log "   1. Run bootstrap CIs:"
  log "      python3 scripts/bootstrap_governance_cis.py --all-phase-e"
  log "   2. Execute paper figures notebook:"
  log "      jupyter nbconvert --to notebook --execute paper_figures.ipynb"
fi
