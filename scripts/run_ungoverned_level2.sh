#!/usr/bin/env bash
# =============================================================================
# run_ungoverned_level2.sh — Level 2 ungoverned counterfactual experiment
# =============================================================================
#
# Runs a small ungoverned campaign on the autoresearch_linux node to produce
# concrete observational evidence that governance guards are load-bearing, not
# merely decorative.
#
# Structure:
#   - GOVERNED run   : 1 seed × 10 trials, none memory, normal control plane
#   - UNGOVERNED run : 1 seed × 10 trials, none memory, guards disabled
#
# The autoresearch_linux node produces 100% runtime_error on the none arm
# (documented in governed campaigns s1–s4). This makes it the ideal Level 2
# substrate: failures are guaranteed and fast (~20–60s each), so the run
# completes in under 30 minutes, and the contrast is maximally stark.
#
# Expected outcome:
#   Governed ledger    : 10 records, each with trial_id, failure_category=
#                        runtime_error, provenance IDs, timestamps
#   Ungoverned ledger  : 0 records (all failed_invalid trials silently dropped)
#   Ungoverned obs log : 10 entries showing crashes happened but were not
#                        recorded in the ledger
#
# After this completes, run compare_governed_ungoverned.py to produce the
# paper's side-by-side comparison table.
#
# USAGE (from repo root on the Linux server):
#   export DEEPSEEK_API_KEY=sk-...
#   export DEEPSEEK_THINKING=disabled
#   bash scripts/run_ungoverned_level2.sh 2>&1 | tee logs/ungoverned_level2_$(date +%Y%m%d).log
#
# OPTIONS:
#   --gpu N       CUDA device index (default: 0)
#   --budget N    Trials per run (default: 10)
#   --dry-run     Print commands without executing
#
# PREREQUISITES (same as other Linux scripts):
#   1. export DEEPSEEK_API_KEY=sk-...
#   2. ollama serve  (running in background)
#   3. ollama pull qwen2.5-coder:7b
#   4. cd nodes/ResNet_trigger && uv sync  (not needed — autoresearch node)
#   5. uv sync  (at repo root)
#   6. Sync this script from Mac first:
#      rsync -avz /Users/wongdowling/Documents/autoresearch_harness/scripts/run_ungoverned_level2.sh \
#                 dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/scripts/
#      rsync -avz /Users/wongdowling/Documents/autoresearch_harness/scripts/compare_governed_ungoverned.py \
#                 dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/scripts/
#      rsync -avz /Users/wongdowling/Documents/autoresearch_harness/src/ \
#                 dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/src/
#
# AFTER THIS COMPLETES:
#   1. Run on server: python3 scripts/compare_governed_ungoverned.py
#   2. Sync back to Mac:
#      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \
#            /Users/wongdowling/Documents/autoresearch_harness/experiments/ledgers/
#      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/paper/tables/ \
#            /Users/wongdowling/Documents/autoresearch_harness/paper/tables/
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
GPU_IDX=0
BUDGET=10
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)      GPU_IDX="$2"; shift 2 ;;
    --gpu=*)    GPU_IDX="${1#--gpu=}"; shift ;;
    --budget)   BUDGET="$2"; shift 2 ;;
    --budget=*) BUDGET="${1#--budget=}"; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  sed -n '2,55p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
AUTORESEARCH_ROOT="$REPO/nodes/autoresearch-macos"

VENV="$REPO/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV" || { echo "[ERROR] No .venv — run: uv sync"; exit 1; }

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$REPO/logs/ungoverned_level2_$RUN_TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
warn() { log "WARNING  $*"; }
die()  { log "ERROR    $*"; exit 1; }

log "============================================================"
log " Level 2 Ungoverned Counterfactual Experiment"
log " Repo  : $REPO"
log " GPU   : CUDA:$GPU_IDX"
log " Budget: $BUDGET trials per run"
log " Dry   : $DRY_RUN"
log "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && die "DEEPSEEK_API_KEY is not set."

export CUDA_VISIBLE_DEVICES="$GPU_IDX"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"

! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1 && \
  die "Ollama not running at localhost:11434. Start with: ollama serve"

[[ ! -d "$AUTORESEARCH_ROOT" ]] && die "autoresearch node not found at $AUTORESEARCH_ROOT (expected nodes/autoresearch-macos)"

# ---------------------------------------------------------------------------
# Campaign IDs
# ---------------------------------------------------------------------------
# Use a dedicated seed (ung = ungoverned) to keep these separate from paper
# campaigns. The seed prefix makes them easy to exclude from aggregate stats.
GOVERNED_ID="deepseek_autoresearch_linux_none_ung_governed"
UNGOVERNED_ID="deepseek_autoresearch_linux_none_ung_ungoverned"

COMMON_ARGS=(
  python3 "$REPO/scripts/run_kdd_memory_ablation.py"
    --node autoresearch_linux
    --node-root "$AUTORESEARCH_ROOT"
    --budget "$BUDGET"
    --manager langgraph_manager
    --memory-mode none
    --model deepseek/deepseek-v4-flash
    --worker-model qwen2.5-coder:7b
    --temperature 0.2
    --no-export
)

# ---------------------------------------------------------------------------
# PHASE 1 — GOVERNED run (normal control plane, all guards active)
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 1 — GOVERNED run ($BUDGET trials, none memory)"
log " Campaign: $GOVERNED_ID"
log "======================================================"

log "INFO  Resetting node state for governed run..."
if [[ $DRY_RUN -eq 0 ]]; then
  python3 "$REPO/scripts/reset_node_state.py" \
    --node autoresearch_linux \
    --campaign-id "$GOVERNED_ID" \
    --node-root "$AUTORESEARCH_ROOT" \
    >> "$LOG_DIR/reset_governed.log" 2>&1 \
  || warn "reset returned non-zero for governed run (may be first run)"
fi

GOVERNED_CMD=("${COMMON_ARGS[@]}" --campaign-id "$GOVERNED_ID")

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY  GOVERNED CMD: ${GOVERNED_CMD[*]}"
else
  log "START  governed run..."
  if "${GOVERNED_CMD[@]}" >> "$LOG_DIR/${GOVERNED_ID}.log" 2>&1; then
    GOVERNED_RECORDS=$(wc -l < "$REPO/experiments/ledgers/${GOVERNED_ID}_trials.jsonl" 2>/dev/null || echo 0)
    log "PASS  governed run: $GOVERNED_RECORDS records in ledger"
  else
    die "Governed run failed — see $LOG_DIR/${GOVERNED_ID}.log"
  fi
fi

# ---------------------------------------------------------------------------
# PHASE 2 — UNGOVERNED run (guards disabled: no pending guard, no ledger append
#            on failure)
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 2 — UNGOVERNED run ($BUDGET trials, none memory)"
log " Campaign: $UNGOVERNED_ID"
log " Guards  : pending-guard DISABLED, ledger-append-on-failure DISABLED"
log "======================================================"

log "INFO  Resetting node state for ungoverned run..."
if [[ $DRY_RUN -eq 0 ]]; then
  python3 "$REPO/scripts/reset_node_state.py" \
    --node autoresearch_linux \
    --campaign-id "$UNGOVERNED_ID" \
    --node-root "$AUTORESEARCH_ROOT" \
    >> "$LOG_DIR/reset_ungoverned.log" 2>&1 \
  || warn "reset returned non-zero for ungoverned run (may be first run)"
fi

UNGOVERNED_CMD=("${COMMON_ARGS[@]}" --campaign-id "$UNGOVERNED_ID" --ungoverned)

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY  UNGOVERNED CMD: ${UNGOVERNED_CMD[*]}"
else
  log "START  ungoverned run..."
  if "${UNGOVERNED_CMD[@]}" >> "$LOG_DIR/${UNGOVERNED_ID}.log" 2>&1; then
    UNG_RECORDS=$(wc -l < "$REPO/experiments/ledgers/${UNGOVERNED_ID}_trials.jsonl" 2>/dev/null || echo 0)
    UNG_OBS=$(wc -l < "$REPO/experiments/ledgers/${UNGOVERNED_ID}_ungoverned_obs.jsonl" 2>/dev/null || echo 0)
    log "PASS  ungoverned run: $UNG_RECORDS records in ledger, $UNG_OBS in observation log"
  else
    die "Ungoverned run failed — see $LOG_DIR/${UNGOVERNED_ID}.log"
  fi
fi

# ---------------------------------------------------------------------------
# PHASE 3 — Comparison table
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " PHASE 3 — Side-by-side comparison"
log "======================================================"

COMPARE_SCRIPT="$REPO/scripts/compare_governed_ungoverned.py"

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY  COMPARE CMD: python3 $COMPARE_SCRIPT --governed $GOVERNED_ID --ungoverned $UNGOVERNED_ID"
elif [[ -f "$COMPARE_SCRIPT" ]]; then
  python3 "$COMPARE_SCRIPT" \
    --governed "$GOVERNED_ID" \
    --ungoverned "$UNGOVERNED_ID" \
    --out "$REPO/paper/tables/level2_governed_vs_ungoverned.csv" \
    2>&1 | tee -a "$MASTER_LOG" \
  && log "PASS  comparison table → paper/tables/level2_governed_vs_ungoverned.csv" \
  || warn "comparison script returned non-zero — check output above"
else
  warn "compare_governed_ungoverned.py not found — skipping comparison"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "======================================================"
log " SUMMARY — $(date)"
log "======================================================"
if [[ $DRY_RUN -eq 0 ]]; then
  GOVERNED_RECORDS=$(wc -l < "$REPO/experiments/ledgers/${GOVERNED_ID}_trials.jsonl" 2>/dev/null || echo "?")
  UNG_RECORDS=$(wc -l < "$REPO/experiments/ledgers/${UNGOVERNED_ID}_trials.jsonl" 2>/dev/null || echo "?")
  UNG_OBS=$(wc -l < "$REPO/experiments/ledgers/${UNGOVERNED_ID}_ungoverned_obs.jsonl" 2>/dev/null || echo "?")
  log "  Governed ledger    : $GOVERNED_RECORDS / $BUDGET records (expected: $BUDGET)"
  log "  Ungoverned ledger  : $UNG_RECORDS / $BUDGET records (expected: 0 for 100%-failure arm)"
  log "  Ungoverned obs log : $UNG_OBS / $BUDGET entries  (expected: $BUDGET)"
  log ""
  log "  The gap between '$GOVERNED_RECORDS governed records' and '$UNG_RECORDS ungoverned records'"
  log "  is the Level 2 paper evidence: governance makes $GOVERNED_RECORDS failures visible;"
  log "  without it, $BUDGET identical failures leave no ledger trace."
fi
log ""
log " Next steps:"
log "   1. Sync back to Mac:"
log "      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \\"
log "            /Users/wongdowling/Documents/autoresearch_harness/experiments/ledgers/"
log "      rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/paper/tables/ \\"
log "            /Users/wongdowling/Documents/autoresearch_harness/paper/tables/"
log "   2. Use paper/tables/level2_governed_vs_ungoverned.csv for §5.x or supplement"
log "   3. Update Level 1 analysis: python3 scripts/analyze_governance_counterfactual.py --csv paper/tables/governance_counterfactual.csv"
log ""
