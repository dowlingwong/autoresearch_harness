#!/usr/bin/env bash
# sync_to_deepthought2.sh
#
# Rsync the autoresearch_harness repo to deepthought2.
#
# Usage:
#   bash scripts/sync_to_deepthought2.sh              # dry-run (shows what would change)
#   bash scripts/sync_to_deepthought2.sh --push        # actually sync
#   bash scripts/sync_to_deepthought2.sh --push --delete  # sync + remove remote-only files
#
# Prerequisites:
#   - SSH key auth to deepthought2 (or set REMOTE below to user@host:path)
#   - rsync available locally (macOS: brew install rsync)
#
# What is excluded (never synced to cluster):
#   - .venv/, node_modules/       — rebuilt on cluster via uv sync
#   - __pycache__/, *.pyc         — byte-compiled artefacts
#   - nodes/*/data/               — large raw datasets (too big to rsync routinely)
#   - nodes/*/models/             — pretrained model weights
#   - nodes/*/checkpoints/        — training checkpoints
#   - experiments/artifacts/      — patch diff + run log files (large; use --include-artifacts)
#   - .DS_Store, *.swp            — macOS / editor cruft
#
# What IS synced:
#   - All Python source (src/, scripts/, tests/)
#   - Configs (configs/)
#   - Ledgers (experiments/ledgers/) — the append-only audit trail
#   - Paper evidence (experiments/paper_evidence/) — curated paper artefacts
#   - Paper LaTeX (A-Governed-Harness-*/,  paper/)
#   - Plans (plan/)
#   - pyproject.toml, .python-version, ENVIRONMENT.md, CLAUDE.md, README.md

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
REMOTE="${DEEPTHOUGHT2_REMOTE:-dwong@deepthought2.etp.kit.edu:/ceph/dwong/autoresearch_harness/}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Flags ──────────────────────────────────────────────────────────────────────
DO_PUSH=false
DO_DELETE=false
INCLUDE_ARTIFACTS=false

for arg in "$@"; do
  case "$arg" in
    --push)              DO_PUSH=true ;;
    --delete)            DO_DELETE=true ;;
    --include-artifacts) INCLUDE_ARTIFACTS=true ;;
    --help|-h)
      echo "Usage: $0 [--push] [--delete] [--include-artifacts]"
      echo ""
      echo "  (no flags)          Dry-run — shows what would be transferred"
      echo "  --push              Actually transfer files"
      echo "  --delete            Remove files on remote that no longer exist locally"
      echo "  --include-artifacts Sync experiments/artifacts/ (patch diffs + run logs; can be large)"
      echo ""
      echo "Override remote with:  DEEPTHOUGHT2_REMOTE=user@host:/path bash $0 --push"
      exit 0
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Build rsync command ────────────────────────────────────────────────────────
RSYNC_ARGS=(
  --archive           # -a: recursive, preserve symlinks/perms/times/owner/group
  --compress          # -z: compress during transfer
  --human-readable    # -h: human-readable sizes
  --progress          # show per-file progress
  --stats             # show transfer summary
  # Exclusions — order matters: first match wins
  --exclude='.venv/'
  --exclude='node_modules/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='.DS_Store'
  --exclude='*.swp'
  --exclude='*.swo'
  --exclude='.pytest_cache/'
  --exclude='*.egg-info/'
  --exclude='.eggs/'
  --exclude='dist/'
  --exclude='build/'
  # Node working dirs — keep configs, exclude heavy data
  --exclude='nodes/*/data/'
  --exclude='nodes/*/raw_data/'
  --exclude='nodes/*/datasets/'
  --exclude='nodes/*/models/'
  --exclude='nodes/*/checkpoints/'
  --exclude='nodes/*/wandb/'
  --exclude='nodes/*/.git/'
  # Session / temp files
  --exclude='*.tmp'
  --exclude='*.lock'
)

if [ "$INCLUDE_ARTIFACTS" = false ]; then
  RSYNC_ARGS+=(--exclude='experiments/artifacts/')
fi

if [ "$DO_DELETE" = true ]; then
  RSYNC_ARGS+=(--delete)
fi

if [ "$DO_PUSH" = false ]; then
  RSYNC_ARGS+=(--dry-run)
  echo "═══════════════════════════════════════════════════════════════"
  echo "  DRY RUN — no files will be transferred"
  echo "  Run with --push to actually sync"
  echo "═══════════════════════════════════════════════════════════════"
fi

echo ""
echo "  Source : $REPO_ROOT/"
echo "  Target : $REMOTE"
echo "  Delete : $DO_DELETE"
echo "  Artifacts : $INCLUDE_ARTIFACTS"
echo ""

# ── Execute ────────────────────────────────────────────────────────────────────
rsync "${RSYNC_ARGS[@]}" "$REPO_ROOT/" "$REMOTE"

echo ""
if [ "$DO_PUSH" = true ]; then
  echo "Sync complete."
  echo ""
  echo "Next steps on deepthought2:"
  echo "  ssh ${REMOTE%%:*}"
  echo "  cd ${REMOTE#*:}"
  echo "  uv sync                          # install deps with Python 3.11"
  echo "  uv run python3 scripts/run_counterfactual.py --help"
else
  echo "Dry-run complete. Re-run with --push to transfer."
fi
