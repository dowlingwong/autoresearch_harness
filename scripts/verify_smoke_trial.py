#!/usr/bin/env python3
"""Post-smoke acceptance gate.

Run after a 1-trial real smoke campaign to verify all correctness invariants:
  1. patch.diff exists and contains real diff lines (not a JSON file, not empty)
  2. No false no_op_patch label on a trial with a real diff
  3. Record has complete provenance (all required hash fields present)
  4. No pending guard remains

Usage:
    python3 scripts/verify_smoke_trial.py --campaign-id <id>
    python3 scripts/verify_smoke_trial.py --campaign-id real_smoke_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGERS_DIR = ROOT / "experiments" / "ledgers"


def _load_records(campaign_id: str) -> list[dict]:
    candidates = list(LEDGERS_DIR.glob(f"{campaign_id}*_trials.jsonl"))
    if not candidates:
        sys.exit(f"ERROR: no ledger found for campaign_id '{campaign_id}' in {LEDGERS_DIR}")
    ledger = candidates[0]
    print(f"Ledger: {ledger}")
    records = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _diff_is_real(patch_ref: str) -> tuple[bool, str]:
    """Return (ok, reason). ok=True iff patch_ref points to a real non-empty diff file."""
    if not patch_ref or not patch_ref.strip():
        return False, "patch_ref is blank"
    path = Path(patch_ref)
    if not path.exists():
        return False, f"patch file does not exist: {path}"
    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        return False, "patch file is empty"
    for line in content.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            return True, "ok"
    return False, "patch file has no diff lines (looks like JSON or whitespace-only)"


def verify(campaign_id: str) -> int:
    records = _load_records(campaign_id)
    if not records:
        print("FAIL: ledger is empty — no trials were recorded.")
        return 1

    failures: list[str] = []
    print(f"\nVerifying {len(records)} trial(s) for campaign '{campaign_id}':\n")

    for rec in records:
        tid = rec.get("trial_id", "?")
        decision = rec.get("decision", "?")
        failure_cat = rec.get("failure_category")
        patch_ref = rec.get("patch_ref", "")
        prov = rec.get("provenance") or {}
        print(f"  [{tid}]  decision={decision}  failure_category={failure_cat}")

        # --- Check 1: if a patch was attempted and trial was not a genuine no-op,
        #              patch.diff must exist and contain real diff lines.
        diff_ok, diff_reason = _diff_is_real(patch_ref)
        if decision in ("kept", "discarded"):
            # Valid trial — must have a real diff
            if not diff_ok:
                msg = f"[{tid}] Valid trial missing real patch.diff: {diff_reason} (patch_ref={patch_ref!r})"
                print(f"    FAIL  {msg}")
                failures.append(msg)
            else:
                print(f"    OK    patch.diff: {Path(patch_ref).name}")
        elif failure_cat == "no_op_patch":
            # Claimed no_op — must NOT have a real diff (otherwise it's a false positive)
            if diff_ok:
                msg = f"[{tid}] Trial labelled no_op_patch but patch.diff contains real diff lines: {patch_ref}"
                print(f"    FAIL  {msg}")
                failures.append(msg)
            else:
                print(f"    OK    genuine no_op (no diff expected)")

        # --- Check 2: provenance completeness
        # TrialProvenance dataclass fields (see memory/provenance.py)
        required_prov = ("proposal_id", "patch_id", "run_id", "metric_id", "decision_id")
        missing_prov = [k for k in required_prov if not prov.get(k)]
        if missing_prov:
            msg = f"[{tid}] Incomplete provenance — missing keys: {missing_prov}"
            print(f"    FAIL  {msg}")
            failures.append(msg)
        else:
            print(f"    OK    provenance complete")

    # --- Check 3: no pending guard
    guards = list(LEDGERS_DIR.glob("*_pending.json"))
    if guards:
        msg = f"Stale pending guard(s) found: {[str(g) for g in guards]}"
        print(f"\n  FAIL  {msg}")
        failures.append(msg)
    else:
        print(f"\n  OK    no pending guards")

    print()
    if failures:
        print(f"RESULT: FAIL — {len(failures)} issue(s):")
        for f in failures:
            print(f"  • {f}")
        return 1
    else:
        print("RESULT: PASS — all acceptance criteria met.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-smoke acceptance gate for a 1-trial real campaign.")
    parser.add_argument("--campaign-id", required=True, help="Campaign ID to verify (e.g. real_smoke_v3).")
    args = parser.parse_args()
    return verify(args.campaign_id)


if __name__ == "__main__":
    raise SystemExit(main())
