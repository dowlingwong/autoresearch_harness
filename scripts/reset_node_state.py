#!/usr/bin/env python3
"""Reset a node to a clean, reproducible state before a campaign.

Usage
-----
# Dry-run: print what would be changed without touching anything
python3 scripts/reset_node_state.py --node resnet_trigger --dry-run

# Reset editable files to a fixed baseline ref instead of moving HEAD
python3 scripts/reset_node_state.py --node resnet_trigger --baseline-ref d589d88

# Reset node files only (does not touch any ledger or artifacts)
python3 scripts/reset_node_state.py --node resnet_trigger

# Reset node files + wipe a specific campaign's ledger and artifacts
python3 scripts/reset_node_state.py --node resnet_trigger --campaign-id kdd_main_5trial

This script is idempotent: running it twice produces the same result as running it once.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.common.paths import (
    ARTIFACTS_DIR,
    EXPERIMENTS_DIR,
    LEDGERS_DIR,
)
from autoresearch.nodes.registry import load_registered_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_checkout(rel_path: str, node_root: Path, dry_run: bool, baseline_ref: str | None) -> None:
    """Restore *rel_path* inside *node_root* to HEAD or a fixed baseline ref."""
    abs_path = node_root / rel_path
    ref_args = [baseline_ref] if baseline_ref else []
    label = f"git checkout {baseline_ref} --" if baseline_ref else "git checkout --"
    if dry_run:
        print(f"  [dry-run] {label} {abs_path}")
        return
    result = subprocess.run(
        ["git", "checkout", *ref_args, "--", str(abs_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Tolerate: if the file was never committed (e.g. in a fresh worktree),
        # git checkout will error; warn but continue.
        print(
            f"  [warn] {label} {abs_path} exited {result.returncode}: "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
    else:
        source = baseline_ref or "HEAD"
        print(f"  restored: {abs_path}  (source: {source})")


def _remove_file(path: Path, label: str, dry_run: bool) -> None:
    if not path.exists():
        print(f"  [skip] {label} not found: {path}")
        return
    if dry_run:
        print(f"  [dry-run] rm {path}  ({label})")
        return
    path.unlink()
    print(f"  removed: {path}  ({label})")


def _remove_dir(path: Path, label: str, dry_run: bool) -> None:
    if not path.exists():
        print(f"  [skip] {label} not found: {path}")
        return
    if dry_run:
        print(f"  [dry-run] rm -rf {path}  ({label})")
        return
    import shutil
    shutil.rmtree(path)
    print(f"  removed: {path}  ({label})")


# ---------------------------------------------------------------------------
# Core reset logic
# ---------------------------------------------------------------------------

def reset_node(
    node_name: str,
    *,
    node_root: Path,
    campaign_id: str | None,
    dry_run: bool,
    baseline_ref: str | None = None,
) -> None:
    """Restore the node's editable files and (optionally) wipe campaign data."""

    spec = load_registered_node(node_name, repo_root=ROOT)

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Resetting node: {node_name}")
    print(f"  node_root : {node_root}")
    if baseline_ref:
        print(f"  baseline  : {baseline_ref}")
    if campaign_id:
        print(f"  campaign  : {campaign_id}")
    print()

    # 1. Restore every editable file to its last-committed state.
    print("Step 1 — restore editable files:")
    for rel_path in spec.editable_paths:
        _git_checkout(rel_path, node_root, dry_run, baseline_ref)

    # 2. Always wipe legacy node-local state so every campaign starts from a
    #    clean baseline.  These files are recreated automatically on first use
    #    by the legacy autoresearch_worker loop:
    #      .autoresearch_state.json  — stores best_bpb / best_commit / pending
    #      results.tsv               — per-trial results table
    #      experiment_memory.jsonl   — legacy memory event log
    #
    #    If these are NOT cleared, the legacy ensure_autoresearch_baseline()
    #    sees an existing best_bpb and skips baseline re-establishment, causing
    #    every ablation arm to inherit the previous arm's best state and produce
    #    identical results.
    print("\nStep 2 — clear legacy node-local state:")
    for legacy_file in (".autoresearch_state.json", "results.tsv", "experiment_memory.jsonl"):
        _remove_file(node_root / legacy_file, f"legacy {legacy_file}", dry_run)

    # 3. (Optional) wipe campaign ledger + artifacts so the ablation starts fresh.
    if campaign_id:
        print("\nStep 3 — remove campaign data:")
        ledger = LEDGERS_DIR / f"{campaign_id}_trials.jsonl"
        _remove_file(ledger, "ledger", dry_run)

        # Pending guards, if any. The first form is canonical; the second is
        # kept for older ledgers created before guard names were normalised.
        for pending in (
            LEDGERS_DIR / f"{campaign_id}_pending.json",
            LEDGERS_DIR / f"{campaign_id}_trials_pending.json",
        ):
            _remove_file(pending, "pending guard", dry_run)

        events = EXPERIMENTS_DIR / "events" / f"{campaign_id}_events.jsonl"
        _remove_file(events, "event stream", dry_run)

        # Artifacts directory for this campaign.
        # Artifacts live under experiments/artifacts/<trial_id>/, not per-campaign,
        # so we look for trial dirs whose name starts with campaign_id.
        if ARTIFACTS_DIR.exists():
            matched = sorted(ARTIFACTS_DIR.glob(f"{campaign_id}*"))
            if matched:
                for art_dir in matched:
                    _remove_dir(art_dir, f"artifact dir ({art_dir.name})", dry_run)
            else:
                print(f"  [skip] no artifact dirs matching {campaign_id}* in {ARTIFACTS_DIR}")

    # 3. Report the post-reset hash of each editable file.
    if not dry_run:
        print("\nPost-reset file hashes (node_state_hash):")
        for rel_path in spec.editable_paths:
            abs_path = node_root / rel_path
            if abs_path.exists():
                digest = _sha256(abs_path)
                print(f"  {rel_path}: {digest}")
            else:
                print(f"  {rel_path}: [not found]")

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Node {node_name} reset complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reset a node to clean, reproducible state before a campaign.\n"
            "Restores editable files via git checkout and optionally removes "
            "ledger + artifacts for a given campaign-id."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--node",
        required=True,
        help="Registered node name (e.g. resnet_trigger).",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "If given, also wipe the JSONL ledger, pending guard, and artifact "
            "directories for this campaign so the run starts from scratch."
        ),
    )
    parser.add_argument(
        "--node-root",
        default=None,
        help=(
            "Filesystem path to the node directory containing the editable files. "
            "Defaults to nodes/<NodeName>/ relative to the repo root."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making any changes.",
    )
    parser.add_argument(
        "--baseline-ref",
        default=os.environ.get("AUTORESEARCH_BASELINE_REF"),
        help=(
            "Optional git ref used as the source for editable-file restore. "
            "Defaults to AUTORESEARCH_BASELINE_REF when set; otherwise HEAD."
        ),
    )
    args = parser.parse_args()

    # Resolve node root.
    if args.node_root:
        node_root = Path(args.node_root).resolve()
    else:
        # Convention: nodes/<name with underscores capitalised as stored on disk>.
        # We load the spec to get the canonical name, then fall back to a best-effort
        # glob over the nodes/ directory.
        spec = load_registered_node(args.node, repo_root=ROOT)
        nodes_dir = ROOT / "nodes"
        # Try an exact match first, then a case-insensitive glob. Prefer the
        # on-disk spelling even on case-insensitive filesystems.
        candidate = nodes_dir / spec.name
        matches = list(nodes_dir.glob("*"))
        lower_matches = [m for m in matches if m.name.lower() == spec.name.lower()]
        if lower_matches:
            candidate = lower_matches[0]
        node_root = candidate

    if not node_root.exists():
        print(
            f"[error] node_root does not exist: {node_root}\n"
            "Pass --node-root <path> to specify the correct location.",
            file=sys.stderr,
        )
        return 1

    reset_node(
        args.node,
        node_root=node_root,
        campaign_id=args.campaign_id,
        dry_run=args.dry_run,
        baseline_ref=args.baseline_ref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
