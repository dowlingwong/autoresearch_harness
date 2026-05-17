#!/usr/bin/env python3
"""
analyze_seed_replication.py
Cross-replicate ordering analysis for the LangGraph memory ablation.

Checks whether the non-monotonic ordering
  repeated_bad_rate(none) >= repeated_bad_rate(summary)
  AND
  repeated_bad_rate(rationale) >= repeated_bad_rate(summary)
holds across all available replicates.

Replicates recognised:
  rep1  ->  lg_ablation2_*          (budget=10, primary result)
  rep2  ->  lg_ablation_rep2_*      (budget=10, second independent replicate)
  rep3  ->  lg_ablation3_*          (budget=20, extended replicate)
"""

import json, pathlib, sys

REPLICATES = {
    "rep1 (b=10)": {
        "none":     "lg_ablation2_none",
        "summary":  "lg_ablation2_append_only_summary",
        "rationale":"lg_ablation2_append_only_summary_with_rationale",
    },
    "rep2 (b=10)": {
        "none":     "lg_ablation_rep2_none",
        "summary":  "lg_ablation_rep2_append_only_summary",
        "rationale":"lg_ablation_rep2_append_only_summary_with_rationale",
    },
    "rep3 (b=20)": {
        "none":     "lg_ablation3_none",
        "summary":  "lg_ablation3_append_only_summary",
        "rationale":"lg_ablation3_append_only_summary_with_rationale",
    },
}

LEDGER_DIR = pathlib.Path("experiments/ledgers")


def load_arm(campaign_id: str) -> dict | None:
    p = LEDGER_DIR / f"{campaign_id}_trials.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.open() if l.strip()]
    if not recs:
        return None
    n = len(recs)
    decisions = [r.get("decision", "") for r in recs]
    kept   = decisions.count("kept")
    disc   = decisions.count("discarded")
    failed = decisions.count("failed_invalid")
    last   = recs[-1]
    rbc    = (last.get("extra") or {}).get("manager", {}).get(
               "worker_repeated_bad_stats", {}).get("repeated_bad_count", 0)
    # RBR denominator = n-1 (first trial cannot be repeated-bad)
    rbr    = rbc / (n - 1) if n > 1 else 0.0
    best   = max((r.get("current_best_before") or 0) for r in recs)
    params = sorted({
        (r.get("extra") or {}).get("manager", {}).get("structured_edit", {}).get("symbol", "?")
        for r in recs
    } - {"?"})
    return dict(n=n, kept=kept, disc=disc, failed=failed,
                rbc=rbc, rbr=rbr, best=best, params=params)


def ordering_holds(arms: dict) -> tuple[bool, str]:
    """Return (holds, reason). Ordering: summary RBR <= none AND summary RBR <= rationale."""
    s = arms["summary"]["rbr"]
    n = arms["none"]["rbr"]
    r = arms["rationale"]["rbr"]
    holds = (s <= n) and (s <= r)
    reason = f"summary={s:.3f}  none={n:.3f}  rationale={r:.3f}"
    return holds, reason


print("=" * 68)
print("  Cross-Replicate Memory Ablation Ordering Analysis")
print("  Hypothesis: RBR(summary) <= RBR(none)  AND  RBR(summary) <= RBR(rationale)")
print("=" * 68)

available = {}
for rep_label, ids in REPLICATES.items():
    arms = {arm: load_arm(cid) for arm, cid in ids.items()}
    if all(v is None for v in arms.values()):
        print(f"\n  {rep_label}: NOT YET RUN — ledgers missing")
        continue
    if any(v is None for v in arms.values()):
        missing = [arm for arm, v in arms.items() if v is None]
        print(f"\n  {rep_label}: PARTIAL — missing arms: {missing}")
        continue

    available[rep_label] = arms
    holds, reason = ordering_holds(arms)
    status = "✅ HOLDS" if holds else "❌ FAILS"
    print(f"\n  {rep_label}:  {status}")
    print(f"    {reason}")
    for arm_name, d in arms.items():
        print(f"    {arm_name:<12} n={d['n']:>2}  kept={d['kept']}  disc={d['disc']}"
              f"  fail={d['failed']}  rbc={d['rbc']}  rbr={d['rbr']:.3f}"
              f"  best_auc={d['best']:.4f}  params={d['params']}")

if not available:
    print("\n  No replicates found. Run run_lg_seed_replication.sh first.")
    sys.exit(0)

# Summary across all available replicates
print("\n" + "=" * 68)
n_reps   = len(available)
n_holds  = sum(1 for arms in available.values() if ordering_holds(arms)[0])
print(f"\n  ORDERING HELD IN {n_holds}/{n_reps} REPLICATES")

if n_holds == n_reps:
    print("  → Strong: non-monotonic ordering (summary < others) is consistent")
    print("    across all available budget levels and independent runs.")
elif n_holds > n_reps / 2:
    print("  → Partial: ordering holds in the majority of replicates.")
    print("    Report as 'held in X of Y seed pairs' in the paper.")
else:
    print("  → Weak: ordering did not consistently hold.")
    print("    Revise paper claims — treat non-monotonic result as tentative.")

# Paper statement
print("\n  Paper statement:")
held_labels = [lbl for lbl, arms in available.items() if ordering_holds(arms)[0]]
fail_labels = [lbl for lbl, arms in available.items() if not ordering_holds(arms)[0]]
if n_holds == n_reps:
    budgets = set()
    for lbl in held_labels:
        budgets.add("10" if "b=10" in lbl else "20")
    print(f"    'The non-monotonic ordering (summary < none ≈ rationale on")
    print(f"     repeated-bad rate) held in all {n_reps} independent replicates")
    print(f"     (budget ∈ {{{', '.join(sorted(budgets))}}}), supporting the")
    print(f"     robustness of the exploration-diversity effect.'")
else:
    print(f"    'The non-monotonic ordering held in {n_holds} of {n_reps} replicates")
    print(f"     ({', '.join(held_labels)}); it failed in {', '.join(fail_labels)}.")
    print(f"     The pattern should be treated as suggestive rather than robust.'")

print()
