#!/usr/bin/env python3
"""Reset a node's local state so the next campaign starts from a clean baseline.

What this clears (unless --preserve is given):
  - .autoresearch_state.json   — legacy loop state (best_bpb, pending_experiment, …)
  - results.tsv                — legacy per-trial results table
  - experiment_memory.jsonl    — legacy memory event log
  - artifacts/<trial-*>/       — per-trial artifact directories written by ClawWorker

What this NEVER touches:
  - train.py / any editable file  — use  git checkout <ref> -- <path>  for that
  - experiments/ledgers/          — the append-only Stage 2 ledger
  - experiments/ledgers/*_pending.json — use scripts/recover_pending.py for stale guards

Usage examples
--------------
# Full reset (default) — wipe legacy state + per-trial artifacts:
  python3 scripts/reset_node.py --node-root nodes/ResNet_trigger

# Wipe only legacy state, keep artifacts:
  python3 scripts/reset_node.py --node-root nodes/ResNet_trigger --preserve artifacts

# Dry-run (print what would be removed):
  python3 scripts/reset_node.py --node-root nodes/ResNet_trigger --dry-run

# Reset + restore train.py to baseline git ref:
  python3 scripts/reset_node.py --node-root nodes/ResNet_trigger --restore-ref d589d88
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Files that live directly inside node_root
NODE_LOCAL_FILES = [
    ".autoresearch_state.json",
    "results.tsv",
    "experiment_memory.jsonl",
]

# Artifact subdirectory relative to repo root (not node_root)
REPO_ARTIFACTS_SUBDIR = "experiments/artifacts"


def _repo_root(node_root: Path) -> Path:
    """Walk up from node_root to find the git repo root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=node_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return Path(result.stdout.strip()).resolve()
    return node_root  # fallback: treat node_root as repo root


def _node_artifact_dirs(node_root: Path, artifacts_base: Path) -> list[Path]:
    """Return per-trial artifact dirs that belong to this node.

    ClawWorker writes to  experiments/artifacts/<campaign_id>/trial-NNN/
    We identify dirs whose names start with 'trial-' under any subdirectory.
    We keep them scoped to node_root's parent tree to avoid touching unrelated nodes.
    """
    dirs: list[Path] = []
    if not artifacts_base.exists():
        return dirs
    for campaign_dir in sorted(artifacts_base.iterdir()):
        if not campaign_dir.is_dir():
            continue
        for trial_dir in sorted(campaign_dir.iterdir()):
            if trial_dir.is_dir() and trial_dir.name.startswith("trial-"):
                dirs.append(trial_dir)
    return dirs


def reset_node(
    node_root: Path,
    *,
    preserve: set[str],
    dry_run: bool,
    restore_ref: str | None,
    artifacts_dir: Path | None,
) -> None:
    node_root = node_root.resolve()
    if not node_root.exists():
        sys.exit(f"ERROR: node-root does not exist: {node_root}")

    repo_root = _repo_root(node_root)
    artifacts_base = (
        artifacts_dir.resolve()
        if artifacts_dir is not None
        else repo_root / REPO_ARTIFACTS_SUBDIR
    )

    removed: list[str] = []
    skipped: list[str] = []

    # --- 1. Node-local legacy state files ---
    if "state" not in preserve:
        for name in NODE_LOCAL_FILES:
            path = node_root / name
            if path.exists():
                if dry_run:
                    print(f"[dry-run] would remove: {path}")
                else:
                    path.unlink()
                    print(f"Removed: {path}")
                removed.append(str(path))
            else:
                skipped.append(str(path))
    else:
        print("Preserving legacy state files (--preserve state).")

    # --- 2. Per-trial artifact directories ---
    if "artifacts" not in preserve:
        trial_dirs = _node_artifact_dirs(node_root, artifacts_base)
        if trial_dirs:
            for trial_dir in trial_dirs:
                if dry_run:
                    print(f"[dry-run] would remove dir: {trial_dir}")
                else:
                    shutil.rmtree(trial_dir)
                    print(f"Removed dir: {trial_dir}")
                removed.append(str(trial_dir))
        else:
            print(f"No trial artifact dirs found under {artifacts_base}")
    else:
        print("Preserving trial artifact dirs (--preserve artifacts).")

    # --- 3. Optionally restore an editable file to a git ref ---
    if restore_ref:
        rel = node_root.relative_to(repo_root)
        train_path = rel / "train.py"
        if dry_run:
            print(f"[dry-run] would run: git show {restore_ref}:{train_path} > {node_root / 'train.py'}")
        else:
            result = subprocess.run(
                ["git", "show", f"{restore_ref}:{train_path}"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(f"WARNING: could not restore train.py from {restore_ref}: {result.stderr.strip()}")
            else:
                (node_root / "train.py").write_text(result.stdout, encoding="utf-8")
                print(f"Restored train.py from {restore_ref}:{train_path}")

    # --- Summary ---
    print()
    if dry_run:
        print(f"[dry-run] {len(removed)} items would be removed.")
    else:
        print(f"Reset complete: {len(removed)} items removed, {len(skipped)} already absent.")
        print("Next step: run a campaign. The legacy loop will reinitialise state files on first use.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset a ResNet trigger node's local state for a fresh campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--node-root",
        required=True,
        help="Path to the node directory (e.g. nodes/ResNet_trigger).",
    )
    parser.add_argument(
        "--preserve",
        nargs="*",
        choices=["state", "artifacts"],
        default=[],
        help="Comma-separated list of items to preserve: state, artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without actually removing anything.",
    )
    parser.add_argument(
        "--restore-ref",
        metavar="GIT_REF",
        default=None,
        help=(
            "Git ref to restore train.py from (e.g. d589d88). "
            "Runs: git show <ref>:<node_rel>/train.py > <node_root>/train.py"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Override the artifacts directory (default: <repo_root>/experiments/artifacts).",
    )

    args = parser.parse_args()
    reset_node(
        node_root=Path(args.node_root),
        preserve=set(args.preserve or []),
        dry_run=args.dry_run,
        restore_ref=args.restore_ref,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
