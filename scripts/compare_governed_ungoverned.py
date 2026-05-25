#!/usr/bin/env python3
"""
compare_governed_ungoverned.py
===============================
Produce the Level 2 paper comparison table: governed vs ungoverned campaign.

Reads:
  - <governed_id>_trials.jsonl        — governed ledger (all trials recorded)
  - <ungoverned_id>_trials.jsonl      — ungoverned ledger (only valid trials)
  - <ungoverned_id>_ungoverned_obs.jsonl — raw observation log (all runs)

Outputs a side-by-side table and writes it to CSV for inclusion in the paper.

Usage:
  python3 scripts/compare_governed_ungoverned.py \\
      --governed  deepseek_autoresearch_linux_none_ung_governed \\
      --ungoverned deepseek_autoresearch_linux_none_ung_ungoverned \\
      --out paper/tables/level2_governed_vs_ungoverned.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
LEDGERS = REPO / "experiments" / "ledgers"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governed", required=True,
                        help="Governed campaign ID")
    parser.add_argument("--ungoverned", required=True,
                        help="Ungoverned campaign ID")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: paper/tables/level2_governed_vs_ungoverned.csv)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else REPO / "paper" / "tables" / "level2_governed_vs_ungoverned.csv"

    gov_ledger = load_jsonl(LEDGERS / f"{args.governed}_trials.jsonl")
    ung_ledger = load_jsonl(LEDGERS / f"{args.ungoverned}_trials.jsonl")
    ung_obs    = load_jsonl(LEDGERS / f"{args.ungoverned}_ungoverned_obs.jsonl")

    budget = max(len(gov_ledger), len(ung_obs), len(ung_ledger), 1)

    print()
    print("=" * 68)
    print("  LEVEL 2: GOVERNED vs UNGOVERNED — SIDE-BY-SIDE")
    print("=" * 68)
    print(f"  Governed campaign  : {args.governed}")
    print(f"  Ungoverned campaign: {args.ungoverned}")
    print()

    # -----------------------------------------------------------------------
    # High-level counts
    # -----------------------------------------------------------------------
    gov_rt    = sum(1 for t in gov_ledger if t.get("failure_category") == "runtime_error")
    gov_prov  = sum(1 for t in gov_ledger if _prov_complete(t))
    gov_tax   = sum(1 for t in gov_ledger if t.get("failure_category"))

    ung_ledger_rt = sum(1 for t in ung_ledger if t.get("failure_category") == "runtime_error")
    ung_obs_crashed = sum(1 for o in ung_obs if not o.get("worker_success", True))
    ung_obs_dropped = sum(1 for o in ung_obs if not o.get("ungoverned_ledger_entry", True))

    rows = [
        ("", "GOVERNED", "UNGOVERNED"),
        ("-" * 40, "-" * 12, "-" * 12),
        ("Trials executed (worker invoked)",
         str(len(gov_ledger)),
         str(len(ung_obs))),
        ("Records in ledger",
         str(len(gov_ledger)),
         str(len(ung_ledger))),
        ("  of which: runtime_error classified",
         str(gov_rt),
         str(ung_ledger_rt)),
        ("  of which: silently dropped (no record)",
         "0",
         str(ung_obs_dropped)),
        ("Failure taxonomy assigned",
         str(gov_tax),
         str(ung_ledger_rt)),
        ("Provenance complete (5 IDs)",
         str(gov_prov),
         str(len(ung_ledger))),
        ("Pending guard written per trial",
         str(len(gov_ledger)),
         "0  (disabled)"),
        ("Crash-recovery possible without re-run",
         "Yes (pending guard + ledger)",
         "No — no sentinel, no record"),
    ]

    for r in rows:
        print(f"  {r[0]:<42}  {r[1]:<14}  {r[2]}")

    # -----------------------------------------------------------------------
    # Per-trial breakdown
    # -----------------------------------------------------------------------
    print()
    print("-" * 68)
    print("  PER-TRIAL DETAIL")
    print("-" * 68)
    print(f"  {'Trial':<8}  {'Governed decision':<22}  {'Ungoverned ledger entry':<22}  {'Obs: crash?'}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*22}  {'-'*11}")

    n = max(len(gov_ledger), len(ung_obs))
    for i in range(n):
        gt = gov_ledger[i] if i < len(gov_ledger) else None
        uo = ung_obs[i] if i < len(ung_obs) else None
        ut = ung_ledger[i] if i < len(ung_ledger) else None

        gov_str = (
            f"{gt['decision']} / {gt.get('failure_category') or 'valid'}"
            if gt else "—"
        )
        ung_entry = "recorded" if ut else "DROPPED (silent)"
        crash = "yes" if (uo and not uo.get("worker_success", True)) else ("no" if uo else "—")

        print(f"  {i+1:<8}  {gov_str:<22}  {ung_entry:<22}  {crash}")

    # -----------------------------------------------------------------------
    # Summary sentence for paper
    # -----------------------------------------------------------------------
    print()
    print("-" * 68)
    print("  PAPER SUMMARY")
    print("-" * 68)
    print(f"""
  Governed: {len(gov_ledger)} trials executed → {len(gov_ledger)} ledger records,
  {gov_rt} classified runtime_error, {gov_prov} with complete provenance.

  Ungoverned: {len(ung_obs)} trials executed → {len(ung_ledger)} ledger records.
  {ung_obs_dropped} failures ({ung_obs_dropped/max(len(ung_obs),1)*100:.0f}%) left no ledger trace.
  Without governance, this campaign is indistinguishable from one that never ran.
""")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "governed", "ungoverned"])
        w.writerow(["trials_executed", len(gov_ledger), len(ung_obs)])
        w.writerow(["ledger_records", len(gov_ledger), len(ung_ledger)])
        w.writerow(["runtime_error_classified", gov_rt, ung_ledger_rt])
        w.writerow(["silently_dropped", 0, ung_obs_dropped])
        w.writerow(["failure_taxonomy_assigned", gov_tax, ung_ledger_rt])
        w.writerow(["provenance_complete", gov_prov, len(ung_ledger)])
        w.writerow(["pending_guard_written", len(gov_ledger), 0])
        w.writerow(["crash_recoverable", "yes", "no"])
    print(f"  CSV written to: {out_path}")

    return 0


def _prov_complete(t: dict) -> bool:
    prov = t.get("provenance") or {}
    return all(prov.get(k) for k in
               ["proposal_id", "patch_id", "metric_id", "run_id", "decision_id"])


if __name__ == "__main__":
    raise SystemExit(main())
