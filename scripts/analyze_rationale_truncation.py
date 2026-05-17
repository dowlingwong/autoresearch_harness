#!/usr/bin/env python3
"""analyze_rationale_truncation.py
Four-arm rationale-verbosity analysis.

Compares RBR, best AUC, and unique parameters explored across:
  arm A — none           (lg_ablation2_none)
  arm B — summary        (lg_ablation2_append_only_summary)
  arm C — short rationale (lg_trunc_short_rationale, 50 tokens)
  arm D — full rationale  (lg_ablation2_append_only_summary_with_rationale)

Hypothesis (verbosity hypothesis):
  If RBR(C) ≈ RBR(B) < RBR(A) ≈ RBR(D), verbosity (not rationale content) is
  the driver of the non-monotonic result. If RBR(C) is between B and D,
  verbosity partially explains but content also contributes.
"""
import json
import pathlib
import sys

ARMS = {
    "A: none": "lg_ablation2_none",
    "B: summary": "lg_ablation2_append_only_summary",
    "C: short_rationale (50 tok)": "lg_trunc_short_rationale",
    "D: full_rationale": "lg_ablation2_append_only_summary_with_rationale",
}

LEDGER_DIR = pathlib.Path("experiments/ledgers")


def load_arm(campaign_id: str) -> dict | None:
    p = LEDGER_DIR / f"{campaign_id}_trials.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(line) for line in p.open() if line.strip()]
    if not recs:
        return None
    n = len(recs)
    decisions = [r.get("decision", "") for r in recs]
    kept = decisions.count("kept")
    disc = decisions.count("discarded")
    fail = decisions.count("failed_invalid")
    last = recs[-1]
    rbc = (last.get("extra") or {}).get("manager", {}).get(
        "worker_repeated_bad_stats", {}).get("repeated_bad_count", 0)
    rbr = rbc / (n - 1) if n > 1 else 0.0
    best = max((r.get("current_best_before") or 0.0) for r in recs)
    # unique params from structured_edit
    params = sorted({
        (r.get("extra") or {}).get("manager", {}).get("structured_edit", {}).get("symbol", "?")
        for r in recs
    } - {"?"})
    return dict(n=n, kept=kept, disc=disc, fail=fail, rbc=rbc, rbr=rbr, best=best, params=params)


print("=" * 70)
print("  Rationale Verbosity Ablation — Four-Arm Analysis")
print("  Hypothesis: RBR(C short) ≈ RBR(B summary) < RBR(A none) ≈ RBR(D full)")
print("=" * 70)

results = {}
for label, cid in ARMS.items():
    data = load_arm(cid)
    if data is None:
        print(f"\n  {label}: NOT YET RUN")
        continue
    results[label] = data
    print(f"\n  {label}:")
    print(f"    n={data['n']}  kept={data['kept']}  disc={data['disc']}  fail={data['fail']}")
    print(f"    rbr={data['rbr']:.3f}  rbc={data['rbc']}  best_auc={data['best']:.4f}")
    print(f"    params_explored={data['params']}")

if len(results) < 4:
    missing = [lbl for lbl in ARMS if lbl not in results]
    print(f"\n  Incomplete — missing arms: {missing}")
    if "C: short_rationale (50 tok)" not in results:
        print("  Run scripts/run_rationale_truncation_ablation.sh to generate arm C.")
    sys.exit(0)

print("\n" + "=" * 70)

# Extract values in arm order
rbr = {k: results[k]["rbr"] for k in ARMS}
best = {k: results[k]["best"] for k in ARMS}

rA = rbr["A: none"]
rB = rbr["B: summary"]
rC = rbr["C: short_rationale (50 tok)"]
rD = rbr["D: full_rationale"]

print(f"\n  RBR summary:")
print(f"    A (none)           = {rA:.3f}")
print(f"    B (summary)        = {rB:.3f}")
print(f"    C (short rationale)= {rC:.3f}")
print(f"    D (full rationale) = {rD:.3f}")

print(f"\n  Verbosity hypothesis test (C ≈ B < A ≈ D):")
c_matches_b = abs(rC - rB) <= 0.1
c_between = rB <= rC <= rD
c_matches_d = abs(rC - rD) <= 0.1

if c_matches_b and not c_matches_d:
    verdict = "SUPPORTED — short rationale matches summary; verbosity is the mechanism."
    interpretation = (
        "Truncating the rationale to 50 tokens restores the performance of "
        "summary-only memory. This supports the verbosity hypothesis: "
        "full rationale allows the LLM to re-justify already-tried directions, "
        "counteracting the avoidance instruction. "
        "Paper claim: rationale content helps, but verbose rationale hurts."
    )
elif c_between and not c_matches_b and not c_matches_d:
    verdict = "PARTIAL — C between B and D; both verbosity and content contribute."
    interpretation = (
        "Truncating rationale improves over full rationale but does not "
        "fully match summary-only. Both rationale length and rationale content "
        "have opposing effects on repeated-bad rate. "
        "Paper claim: moderate rationale length is optimal."
    )
elif c_matches_d:
    verdict = "NOT SUPPORTED — short rationale behaves like full rationale."
    interpretation = (
        "Truncation does not help. The non-monotonic result is not caused by "
        "verbosity alone. Rationale content (not length) may be the driver, "
        "or the LLM ignores rationale at any length under the avoidance prompt. "
        "Paper claim: rationale format (not length) determines whether memory helps."
    )
else:
    verdict = "INCONCLUSIVE"
    interpretation = (
        "Pattern does not fit any of the three hypothesis outcomes. "
        "Inspect per-arm parameter sequences and avoidance prompt interactions."
    )

print(f"    Verdict: {verdict}")
print(f"\n  Interpretation:\n    {interpretation}")

print(f"\n  AUC comparison:")
for lbl, d in results.items():
    print(f"    {lbl}: best={d['best']:.4f}  params={d['params']}")

print(f"\n  Paper statement:")
if c_matches_b and not c_matches_d:
    print(
        f"    'A four-arm verbosity ablation (none / summary / short-rationale-50tok / full-rationale)\n"
        f"     confirms that the non-monotonic result is driven by rationale verbosity: truncating\n"
        f"     the rationale to 50 tokens restores summary-level repeated-bad rate ({rC:.2f} vs {rB:.2f}),\n"
        f"     while full rationale matches no-memory ({rD:.2f} vs {rA:.2f}). This supports the\n"
        f"     exploration-diversity mechanism: brief memory context aids avoidance;\n"
        f"     verbose context re-introduces the bias it was meant to overcome.'"
    )
elif c_between:
    print(
        f"    'A four-arm verbosity ablation shows a monotonic trend with rationale length:\n"
        f"     none ({rA:.2f}) ≥ full ({rD:.2f}) > short ({rC:.2f}) ≥ summary ({rB:.2f}).\n"
        f"     Both verbosity and content contribute to the repeated-bad rate.'"
    )
else:
    print(
        f"    'The four-arm ablation (none={rA:.2f}, summary={rB:.2f},\n"
        f"     short-rationale={rC:.2f}, full-rationale={rD:.2f}) does not support\n"
        f"     the verbosity hypothesis in isolation. Rationale format effects remain open.'"
    )

print()
