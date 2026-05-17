#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an isolated git worktree for one trial.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="Base ref for the isolated trial branch.",
    )
    parser.add_argument(
        "--worktrees-dir",
        default=str(ROOT / "experiments" / "worktrees"),
        help="Parent directory for campaign trial worktrees.",
    )
    parser.add_argument(
        "--approval-file",
        help="Optional approval JSON path. Defaults inside the worktree.",
    )
    args = parser.parse_args()

    branch = f"codex/{args.campaign_id}/{args.trial_id}"
    worktree = Path(args.worktrees_dir) / args.campaign_id / args.trial_id
    worktree.parent.mkdir(parents=True, exist_ok=True)
    if not worktree.exists():
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(worktree), args.base_ref],
            cwd=ROOT,
            check=True,
        )
    approval = Path(args.approval_file) if args.approval_file else worktree / ".trial_approval.json"
    approval.write_text(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "trial_id": args.trial_id,
                "branch": branch,
                "worktree": str(worktree),
                "base_ref": args.base_ref,
                "approved_for_master_apply": False,
                "created_at": _iso_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"worktree={worktree}")
    print(f"branch={branch}")
    print(f"approval={approval}")
    return 0


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
