#!/usr/bin/env bash
# run_counterfactual_server.sh
#
# Run the full N=30 governed vs. ungoverned counterfactual for autoresearch_linux
# on an anonymous NVIDIA GPU cluster.
#
# ── Step 1: sync repo from your laptop ────────────────────────────────────────
#   bash scripts/sync_to_deepthought2.sh --push
#
# ── Step 2: SSH into deepthought2 and run this script ─────────────────────────
#   ssh dwong@deepthought2.etp.kit.edu
#   cd /ceph/dwong/autoresearch_harness
#   bash scripts/run_counterfactual_server.sh
#
# ── Step 3: sync results back ─────────────────────────────────────────────────
#   (from laptop)
#   rsync -avz dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/experiments/ledgers/ \
#       experiments/ledgers/
#
# ── Step 4: analyze and produce LaTeX table ───────────────────────────────────
#   python3 scripts/analyze_counterfactual.py \
#       --summary experiments/ledgers/kdd_cf_arlinux_cf_summary.json \
#       --summary experiments/ledgers/kdd_cf_openml_cf_summary.json \
#       --output paper/tables/counterfactual_comparison.tex

set -euo pipefail

# ── Config — edit these if your paths differ ──────────────────────────────────
REPO_ROOT="/ceph/dwong/autoresearch_harness"
NODE_ROOT="${NODE_ROOT:-${REPO_ROOT}}"   # train.py lives at repo root on deepthought2
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
WORKER_MODEL="${WORKER_MODEL:-qwen2.5-coder:7b}"
BUDGET="${BUDGET:-30}"
BASE="kdd_cf_arlinux"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  autoresearch_linux counterfactual  ·  N=${BUDGET}/arm"
echo "  repo     : ${REPO_ROOT}"
echo "  node     : ${NODE_ROOT}"
echo "  model    : ${MODEL} @ ${OLLAMA_HOST}"
echo "  worker   : ${WORKER_MODEL}"
echo "═══════════════════════════════════════════════════════════════"

if [ ! -d "${NODE_ROOT}" ]; then
  echo "ERROR: node root does not exist: ${NODE_ROOT}"
  echo "  Clone or copy the autoresearch_linux node directory there first."
  exit 1
fi

# Check Ollama is reachable
if ! curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
  echo "ERROR: Ollama is not reachable at ${OLLAMA_HOST}"
  echo "  Start Ollama: ollama serve &"
  echo "  Pull models:  ollama pull ${MODEL}"
  echo "                ollama pull ${WORKER_MODEL}"
  exit 1
fi

echo ""
echo "Ollama reachable. Starting counterfactual run ..."
echo ""

# ── Install deps if needed ────────────────────────────────────────────────────
cd "${REPO_ROOT}"
if command -v uv &> /dev/null; then
  uv sync --quiet
  PYTHON="uv run python3"
else
  echo "WARN: uv not found, using system python3 (must be 3.11+)"
  PYTHON="python3"
fi

# ── Run both arms ─────────────────────────────────────────────────────────────
${PYTHON} scripts/run_counterfactual.py \
    --node autoresearch_linux \
    --base "${BASE}" \
    --budget "${BUDGET}" \
    --node-root "${NODE_ROOT}" \
    --model "${MODEL}" \
    --host "${OLLAMA_HOST}" \
    --worker-model "${WORKER_MODEL}" \
    --use-claw-worker \
    --memory-mode none

# ── Analyze and produce table ─────────────────────────────────────────────────
SUMMARY="${REPO_ROOT}/experiments/ledgers/${BASE}_cf_summary.json"
OPENML_SUMMARY="${REPO_ROOT}/experiments/ledgers/kdd_cf_openml_cf_summary.json"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Running analyze_counterfactual.py ..."
echo "═══════════════════════════════════════════════════════════════"

if [ -f "${OPENML_SUMMARY}" ]; then
  # Both nodes available — produce combined table
  ${PYTHON} scripts/analyze_counterfactual.py \
      --summary "${SUMMARY}" \
      --summary "${OPENML_SUMMARY}" \
      --output "${REPO_ROOT}/paper/tables/counterfactual_comparison.tex"
else
  # arlinux only
  ${PYTHON} scripts/analyze_counterfactual.py \
      --summary "${SUMMARY}" \
      --output "${REPO_ROOT}/paper/tables/counterfactual_comparison.tex"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Outputs:"
echo "  Ledgers → experiments/ledgers/${BASE}_gov_trials.jsonl"
echo "            experiments/ledgers/${BASE}_ung_trials.jsonl"
echo "            experiments/ledgers/${BASE}_ung_ungoverned_obs.jsonl"
echo "  Summary → experiments/ledgers/${BASE}_cf_summary.json"
echo "  LaTeX   → paper/tables/counterfactual_comparison.tex"
echo ""
echo "  Sync back to laptop:"
echo "    rsync -avz dwong@deepthought2.etp.kit.edu:${REPO_ROOT}/experiments/ledgers/ \\"
echo "        experiments/ledgers/"
echo "═══════════════════════════════════════════════════════════════"
