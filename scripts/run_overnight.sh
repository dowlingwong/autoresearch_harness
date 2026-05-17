#!/usr/bin/env bash
# =============================================================================
# run_overnight.sh — Unattended overnight experiment runner
# =============================================================================
#
# Runs every pending campaign in dependency order, then exports analysis
# outputs (bootstrap CIs, paper tables, figures).
#
# USAGE (from repo root):
#   bash scripts/run_overnight.sh [OPTIONS] 2>&1 | tee logs/tonight.log
#
# OPTIONS:
#   --skip-resnet       Skip Phase C/D (ResNet). Use if GPU or Ollama not ready.
#   --skip-autoresearch Skip Phase E (autoresearch_macos).
#   --fast-resnet       Run ResNet in fast-search mode (3 epochs, CPU) — ~6 hr
#                       instead of ~17 hr. Metrics are less accurate but valid.
#   --skip-analysis     Skip bootstrap CI and table/figure export at the end.
#   --dry-run           Print every command that would run; don't execute.
#
# PREREQUISITES (all must be satisfied before running):
#   1.  export DEEPSEEK_API_KEY=sk-...
#   2.  export DEEPSEEK_THINKING=disabled   (avoids reasoning tokens, saves cost)
#   3.  Ollama running in a separate terminal: ollama serve
#   4.  Model pulled: ollama pull qwen2.5-coder:7b
#   5.  For ResNet: nodes/ResNet_trigger/.venv must exist (run: cd nodes/ResNet_trigger && uv sync)
#   6.  For autoresearch: nodes/autoresearch-macos/.venv + data must exist
#       (run: cd nodes/autoresearch-macos && uv sync && uv run prepare.py)
#   7.  Harness environment active:  source .venv/bin/activate  (or use uv run)
#
# ESTIMATED WALL CLOCK (serial, Apple Silicon M-series or similar):
#   Phase B  (fast LocalWorker nodes)        ~  1.5 hr
#   Phase C  (ResNet smoke)                  ~ 10 min
#   Phase D  (ResNet ablation, 9 arms × 15)  ~ 17 hr  (or ~6 hr with --fast-resnet)
#   Phase E  (autoresearch_macos, 4 seeds)   ~  8 hr
#   Phase F  (analysis exports)              ~  5 min
#   ─────────────────────────────────────────────────
#   Total without ResNet                     ~ 10 hr
#   Total with ResNet (fast mode)            ~ 16 hr
#   Total with ResNet (full GPU)             ~ 27 hr  ← split across two nights
#
# RESUME SAFETY:
#   Each campaign is skipped automatically if its ledger already contains
#   >= the target number of trials. Safe to re-run after interruption.
#
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Parse flags
# ---------------------------------------------------------------------------
SKIP_RESNET=0
SKIP_AUTORESEARCH=0
FAST_RESNET=0
SKIP_ANALYSIS=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --skip-resnet)       SKIP_RESNET=1 ;;
    --skip-autoresearch) SKIP_AUTORESEARCH=1 ;;
    --fast-resnet)       FAST_RESNET=1 ;;
    --skip-analysis)     SKIP_ANALYSIS=1 ;;
    --dry-run)           DRY_RUN=1 ;;
    *) echo "[ERROR] Unknown option: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Paths and log setup
# ---------------------------------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$REPO/logs/overnight_$RUN_TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

echo "============================================================" | tee -a "$MASTER_LOG"
echo " Overnight runner started: $(date)"                           | tee -a "$MASTER_LOG"
echo " Log dir : $LOG_DIR"                                          | tee -a "$MASTER_LOG"
echo " Options : skip_resnet=$SKIP_RESNET  skip_autoresearch=$SKIP_AUTORESEARCH  fast_resnet=$FAST_RESNET  dry_run=$DRY_RUN" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# ---------------------------------------------------------------------------
# 2. Pre-flight checks
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
warn() { log "⚠  WARN  $*"; }
die()  { log "✗  ERROR $*"; exit 1; }

[[ -z "${DEEPSEEK_API_KEY:-}" ]] && die "DEEPSEEK_API_KEY is not set. Aborting."

if [[ $SKIP_RESNET -eq 0 || $SKIP_AUTORESEARCH -eq 0 ]]; then
  if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama does not appear to be running at localhost:11434."
    warn "ResNet and autoresearch campaigns will fail. Consider --skip-resnet --skip-autoresearch."
  fi
fi

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
PASSED=()
FAILED=()
SKIPPED=()

# trial_count <campaign_id>
# Returns the number of lines in the campaign's JSONL ledger, or 0.
trial_count() {
  local ledger="$REPO/experiments/ledgers/${1}_trials.jsonl"
  if [[ -f "$ledger" ]]; then
    wc -l < "$ledger" | tr -d '[:space:]'
  else
    echo 0
  fi
}

# run_campaign <campaign_id> <expected_trials> <cmd...>
# Skips if ledger already has >= expected_trials lines.
# Logs stdout+stderr to LOG_DIR/<campaign_id>.log.
# Records pass/fail/skip in global arrays.
run_campaign() {
  local id="$1"
  local expected="$2"
  shift 2
  local actual
  actual=$(trial_count "$id")

  if [[ "$actual" -ge "$expected" ]]; then
    log "SKIP  $id  ($actual/$expected trials already in ledger)"
    SKIPPED+=("$id")
    return 0
  fi

  log "START $id  ($actual/$expected trials; running $((expected - actual)) more)"
  local clog="$LOG_DIR/${id}.log"
  echo "CMD: $*" > "$clog"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY   $id  → $*"
    SKIPPED+=("$id [dry-run]")
    return 0
  fi

  if "$@" >> "$clog" 2>&1; then
    local final
    final=$(trial_count "$id")
    log "PASS  $id  ($final trials in ledger)"
    PASSED+=("$id")
  else
    local exit_code=$?
    log "FAIL  $id  (exit $exit_code) — see $clog"
    FAILED+=("$id")
  fi
}

# reset_campaign <campaign_id> <node_name> <node_root>
# Wipes stale ledger + artifacts and restores editable files to HEAD.
reset_campaign() {
  local id="$1" node="$2" root="$3"
  log "RESET $id"
  if [[ $DRY_RUN -eq 0 ]]; then
    python3 "$REPO/scripts/reset_node_state.py" \
      --node "$node" \
      --campaign-id "$id" \
      --node-root "$root" \
      >> "$LOG_DIR/resets.log" 2>&1 || warn "reset_node_state.py returned non-zero for $id"
  fi
}

# ---------------------------------------------------------------------------
# 4. PHASE B — Fast LocalWorker campaigns (no GPU, no Ollama needed)
# ---------------------------------------------------------------------------
log ""
log "══════════════════════════════════════════════════════════════"
log " PHASE B — Fast LocalWorker campaigns"
log "══════════════════════════════════════════════════════════════"

# ── B1: mlagentbench_vectorization ──────────────────────────────────────────
for SEED in s1 s2; do
  run_campaign "deepseek_mlagentbench_${SEED}" 10 \
    python3 "$REPO/scripts/run_local_node_campaign.py" \
      --node mlagentbench_vectorization \
      --campaign-id "deepseek_mlagentbench_${SEED}" \
      --budget 10 --manager langgraph_manager \
      --memory-mode append_only_summary \
      --model deepseek/deepseek-v4-flash --temperature 0.2
done

# ── B2: lr_synthetic extra seeds (s4, s5 per arm) ───────────────────────────
for ARM in none append_only_summary append_only_summary_with_rationale; do
  for SEED in s4 s5; do
    run_campaign "deepseek_lr_${ARM}_${SEED}" 10 \
      python3 "$REPO/scripts/run_local_node_campaign.py" \
        --node lr_synthetic \
        --campaign-id "deepseek_lr_${ARM}_${SEED}" \
        --budget 10 --manager langgraph_manager \
        --memory-mode "${ARM}" \
        --model deepseek/deepseek-v4-flash --temperature 0.2
  done
done

# ── B3: mlp_synthetic — add missing none and rationale arms ─────────────────
run_campaign "deepseek_mlp_none_s1" 10 \
  python3 "$REPO/scripts/run_local_node_campaign.py" \
    --node mlp_synthetic \
    --campaign-id deepseek_mlp_none_s1 \
    --budget 10 --manager langgraph_manager \
    --memory-mode none \
    --model deepseek/deepseek-v4-flash --temperature 0.2

run_campaign "deepseek_mlp_rationale_s1" 10 \
  python3 "$REPO/scripts/run_local_node_campaign.py" \
    --node mlp_synthetic \
    --campaign-id deepseek_mlp_rationale_s1 \
    --budget 10 --manager langgraph_manager \
    --memory-mode append_only_summary_with_rationale \
    --model deepseek/deepseek-v4-flash --temperature 0.2

# ── B4: openml extra seeds ───────────────────────────────────────────────────
run_campaign "deepseek_openml_cg_s4" 20 \
  python3 "$REPO/scripts/run_openml_tabular_campaign.py" \
    --node openml_credit_g \
    --campaign-id deepseek_openml_cg_s4 \
    --budget 20 --manager langgraph_manager \
    --memory-mode append_only_summary \
    --model deepseek/deepseek-v4-flash --temperature 0.2

run_campaign "deepseek_openml_bm_s4" 20 \
  python3 "$REPO/scripts/run_openml_tabular_campaign.py" \
    --node openml_bank_marketing \
    --campaign-id deepseek_openml_bm_s4 \
    --budget 20 --manager langgraph_manager \
    --memory-mode append_only_summary \
    --model deepseek/deepseek-v4-flash --temperature 0.2

# ---------------------------------------------------------------------------
# 5. PHASE C — ResNet smoke test (validates ClawWorker + sklearn fix)
# ---------------------------------------------------------------------------
RESNET_ROOT="$REPO/nodes/ResNet_trigger"
RESNET_SMOKE_PASS=0

if [[ $SKIP_RESNET -eq 1 ]]; then
  log ""
  log "SKIP  Phase C/D — ResNet (--skip-resnet set)"
else
  log ""
  log "══════════════════════════════════════════════════════════════"
  log " PHASE C — ResNet smoke test"
  log "══════════════════════════════════════════════════════════════"

  if [[ ! -d "$RESNET_ROOT/.venv" ]]; then
    warn "ResNet venv not found at $RESNET_ROOT/.venv"
    warn "Run: cd nodes/ResNet_trigger && uv sync"
    warn "Skipping ResNet smoke + ablation."
    SKIP_RESNET=1
  else
    # Smoke: 1 trial in fast-search mode (always use fast-search for smoke)
    FAST_ENV="RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_EPOCHS=3 RESNET_TRIGGER_DEVICE=cpu"
    reset_campaign "deepseek_resnet_smoke" "resnet_trigger" "$RESNET_ROOT"

    if [[ $DRY_RUN -eq 0 ]]; then
      log "START deepseek_resnet_smoke (1 trial, fast-search CPU)"
      RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_EPOCHS=3 RESNET_TRIGGER_DEVICE=cpu \
      python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
        --node resnet_trigger \
        --campaign-id deepseek_resnet_smoke \
        --node-root "$RESNET_ROOT" \
        --budget 1 --manager langgraph_manager \
        --memory-mode append_only_summary \
        --model deepseek/deepseek-v4-flash \
        --worker-model qwen2.5-coder:7b \
        --temperature 0.2 --no-export \
        >> "$LOG_DIR/deepseek_resnet_smoke.log" 2>&1 \
      && RESNET_SMOKE_PASS=1 \
      || { warn "ResNet smoke FAILED — skipping Phase D. See $LOG_DIR/deepseek_resnet_smoke.log"; }

      if [[ $RESNET_SMOKE_PASS -eq 1 ]]; then
        log "PASS  deepseek_resnet_smoke"
        PASSED+=("deepseek_resnet_smoke")
      else
        FAILED+=("deepseek_resnet_smoke")
      fi
    else
      log "DRY   deepseek_resnet_smoke"
      RESNET_SMOKE_PASS=1
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 6. PHASE D — ResNet full ablation (3 arms × 3 seeds × budget 15)
# ---------------------------------------------------------------------------
if [[ $SKIP_RESNET -eq 0 && $RESNET_SMOKE_PASS -eq 1 ]]; then
  log ""
  log "══════════════════════════════════════════════════════════════"
  log " PHASE D — ResNet full ablation (budget 15)"
  log "══════════════════════════════════════════════════════════════"

  if [[ $FAST_RESNET -eq 1 ]]; then
    log "INFO  Using fast-search mode (RESNET_TRIGGER_FAST_SEARCH=1)"
    export RESNET_TRIGGER_FAST_SEARCH=1
    export RESNET_TRIGGER_FAST_EPOCHS=3
  fi

  for ARM in none append_only_summary append_only_summary_with_rationale; do
    for SEED in s1 s2 s3; do
      CID="deepseek_resnet_${ARM}_${SEED}"
      ACTUAL=$(trial_count "$CID")

      # If ledger exists but has < 15 trials AND the old trials were all failed
      # (sklearn bug), reset and rerun from scratch at budget 15.
      # If already at 15+, skip.
      if [[ "$ACTUAL" -ge 15 ]]; then
        log "SKIP  $CID  ($ACTUAL/15 trials already in ledger)"
        SKIPPED+=("$CID")
        continue
      fi

      # For the failed sklearn runs (10 trials, all failed_invalid), wipe and restart.
      if [[ "$ACTUAL" -gt 0 && "$ACTUAL" -lt 15 ]]; then
        log "INFO  $CID has $ACTUAL stale trials — resetting before rerun"
        reset_campaign "$CID" "resnet_trigger" "$RESNET_ROOT"
      fi

      run_campaign "$CID" 15 \
        python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
          --node resnet_trigger \
          --campaign-id "$CID" \
          --node-root "$RESNET_ROOT" \
          --budget 15 --manager langgraph_manager \
          --memory-mode "${ARM}" \
          --model deepseek/deepseek-v4-flash \
          --worker-model qwen2.5-coder:7b \
          --temperature 0.2 --no-export
    done
  done
fi

# ---------------------------------------------------------------------------
# 7. PHASE E — autoresearch_macos (ClawWorker, 4 seeds × budget 20)
# ---------------------------------------------------------------------------
AR_ROOT="$REPO/nodes/autoresearch-macos"

if [[ $SKIP_AUTORESEARCH -eq 1 ]]; then
  log ""
  log "SKIP  Phase E — autoresearch_macos (--skip-autoresearch set)"
else
  log ""
  log "══════════════════════════════════════════════════════════════"
  log " PHASE E — autoresearch_macos (ClawWorker)"
  log "══════════════════════════════════════════════════════════════"

  if [[ ! -d "$AR_ROOT/.venv" ]]; then
    warn "autoresearch-macos venv not found at $AR_ROOT/.venv"
    warn "Run: cd nodes/autoresearch-macos && uv sync && uv run prepare.py"
    warn "Skipping Phase E."
  elif [[ ! -d "$AR_ROOT/data" ]]; then
    warn "autoresearch-macos data/ not found — prepare.py has not been run."
    warn "Run: cd nodes/autoresearch-macos && uv run prepare.py"
    warn "Skipping Phase E."
  else
    # Smoke: 1 trial with baseline_manager (no API cost, verifies ClawWorker pipeline)
    SMOKE_ID="autoresearch_claw_smoke"
    if [[ $(trial_count "$SMOKE_ID") -ge 1 ]]; then
      log "SKIP  $SMOKE_ID (already done)"
      SKIPPED+=("$SMOKE_ID")
    else
      log "START $SMOKE_ID (1 trial, baseline_manager, ClawWorker pipeline check)"
      if [[ $DRY_RUN -eq 0 ]]; then
        python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
          --node autoresearch_macos \
          --campaign-id "$SMOKE_ID" \
          --node-root "$AR_ROOT" \
          --budget 1 --manager baseline_manager \
          --memory-mode append_only_summary \
          --worker-model qwen2.5-coder:7b \
          --no-export \
          >> "$LOG_DIR/${SMOKE_ID}.log" 2>&1 \
        && { log "PASS  $SMOKE_ID"; PASSED+=("$SMOKE_ID"); } \
        || { log "FAIL  $SMOKE_ID — see $LOG_DIR/${SMOKE_ID}.log"; FAILED+=("$SMOKE_ID"); }
      else
        log "DRY   $SMOKE_ID"
      fi
    fi

    # 4 seeds with DeepSeek manager
    for SEED in s1 s2 s3 s4; do
      CID="deepseek_autoresearch_${SEED}"
      reset_campaign "$CID" "autoresearch_macos" "$AR_ROOT"

      run_campaign "$CID" 20 \
        python3 "$REPO/scripts/run_kdd_memory_ablation.py" \
          --node autoresearch_macos \
          --campaign-id "$CID" \
          --node-root "$AR_ROOT" \
          --budget 20 --manager langgraph_manager \
          --memory-mode append_only_summary \
          --model deepseek/deepseek-v4-flash \
          --worker-model qwen2.5-coder:7b \
          --temperature 0.2 --no-export
    done
  fi
fi

# ---------------------------------------------------------------------------
# 8. PHASE F — Analysis: bootstrap CIs + paper tables + figures
# ---------------------------------------------------------------------------
if [[ $SKIP_ANALYSIS -eq 1 ]]; then
  log ""
  log "SKIP  Phase F — analysis (--skip-analysis set)"
else
  log ""
  log "══════════════════════════════════════════════════════════════"
  log " PHASE F — Bootstrap CIs + paper tables + figures"
  log "══════════════════════════════════════════════════════════════"

  if [[ $DRY_RUN -eq 0 ]]; then
    # Bootstrap CIs: all lr_synthetic campaigns (5 seeds × 3 arms = 15 campaigns)
    log "INFO  Computing bootstrap CIs for lr_synthetic ablation..."
    CI_ARGS=()
    for ARM in none append_only_summary append_only_summary_with_rationale; do
      for SEED in s1 s2 s3 s4 s5; do
        LEDGER="$REPO/experiments/ledgers/deepseek_lr_${ARM}_${SEED}_trials.jsonl"
        if [[ -f "$LEDGER" ]]; then
          CI_ARGS+=(--campaign "deepseek_lr_${ARM}_${SEED}" --node lr_synthetic)
        fi
      done
    done

    if [[ ${#CI_ARGS[@]} -gt 0 ]]; then
      python3 "$REPO/scripts/bootstrap_governance_cis.py" \
        "${CI_ARGS[@]}" \
        --out "$REPO/paper/tables/governance_bootstrap_cis_lr.csv" \
        --samples 10000 --seed 42 \
        >> "$LOG_DIR/bootstrap_lr.log" 2>&1 \
      && log "PASS  bootstrap CIs (lr_synthetic)" \
      || warn "bootstrap CIs (lr_synthetic) returned non-zero — see $LOG_DIR/bootstrap_lr.log"
    fi

    # Bootstrap CIs: ResNet campaigns (only if they exist)
    log "INFO  Computing bootstrap CIs for resnet_trigger ablation..."
    CI_ARGS_RESNET=()
    for ARM in none append_only_summary append_only_summary_with_rationale; do
      for SEED in s1 s2 s3; do
        LEDGER="$REPO/experiments/ledgers/deepseek_resnet_${ARM}_${SEED}_trials.jsonl"
        if [[ -f "$LEDGER" ]]; then
          CI_ARGS_RESNET+=(--campaign "deepseek_resnet_${ARM}_${SEED}" --node resnet_trigger)
        fi
      done
    done

    if [[ ${#CI_ARGS_RESNET[@]} -gt 0 ]]; then
      python3 "$REPO/scripts/bootstrap_governance_cis.py" \
        "${CI_ARGS_RESNET[@]}" \
        --out "$REPO/paper/tables/governance_bootstrap_cis_resnet.csv" \
        --samples 10000 --seed 42 \
        >> "$LOG_DIR/bootstrap_resnet.log" 2>&1 \
      && log "PASS  bootstrap CIs (resnet_trigger)" \
      || warn "bootstrap CIs (resnet_trigger) returned non-zero — see $LOG_DIR/bootstrap_resnet.log"
    fi

    # Paper tables (ResNet is the primary export target of export_kdd_tables.py)
    log "INFO  Exporting paper tables..."
    python3 "$REPO/scripts/export_kdd_tables.py" \
      --node resnet_trigger \
      >> "$LOG_DIR/export_tables.log" 2>&1 \
    && log "PASS  export_kdd_tables.py" \
    || warn "export_kdd_tables.py returned non-zero — see $LOG_DIR/export_tables.log"

    # Figures
    log "INFO  Exporting paper figures..."
    python3 "$REPO/scripts/export_kdd_figures.py" \
      >> "$LOG_DIR/export_figures.log" 2>&1 \
    && log "PASS  export_kdd_figures.py" \
    || warn "export_kdd_figures.py returned non-zero — see $LOG_DIR/export_figures.log"
  else
    log "DRY   Phase F — analysis scripts would run here"
  fi
fi

# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
log ""
log "══════════════════════════════════════════════════════════════"
log " SUMMARY — $(date)"
log "══════════════════════════════════════════════════════════════"
log " Passed  (${#PASSED[@]})  : ${PASSED[*]:-none}"
log " Failed  (${#FAILED[@]})  : ${FAILED[*]:-none}"
log " Skipped (${#SKIPPED[@]}) : ${SKIPPED[*]:-none}"
log ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "⚠  ${#FAILED[@]} campaign(s) failed. Check individual logs in $LOG_DIR/"
  log "   Re-running this script is safe — completed campaigns will be skipped."
  exit 1
else
  log "✅ All campaigns passed or were already complete."
  log "   Results are in experiments/ledgers/ and paper/tables/"
  exit 0
fi
