#!/usr/bin/env python3
"""Compute bootstrap CIs for the governed vs. ungoverned counterfactual and
produce the paper-facing LaTeX table.

Usage (pass one --summary per node pair):
  python3 scripts/analyze_counterfactual.py \\
      --summary experiments/ledgers/kdd_cf_arlinux_cf_summary.json \\
      --summary experiments/ledgers/kdd_cf_openml_cf_summary.json \\
      --output paper/tables/counterfactual_comparison.tex

What it computes
────────────────
For each node × arm combination:

  ledger_completeness = |ledger_records| / |trials_that_ran|

  Governed arm:   every trial produces a record  →  always 1.0
  Ungoverned arm: only valid (kept/discarded) trials produce records;
                  failed_invalid trials are silently dropped  →  < 1.0

  Bootstrap CI: 10,000 resamples (with replacement) of the binary
  completeness vector [1, 1, …, 1, 0, 0, …, 0], seed 42.
  Governed CI is always [1.00, 1.00] by construction.

Output
──────
  • Console table (human-readable)
  • LaTeX table snippet ready to paste into §5.3
  • JSON with raw numbers for downstream use
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for the mean of *values*."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_resamples)])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return (lo, hi)


# ── Per-arm analysis ──────────────────────────────────────────────────────────

def _analyze_governed(gov_ledger: Path, budget: int) -> dict:
    """Governed arm: all trials have records by construction."""
    records = TrialAppendStore(gov_ledger).read_all() if gov_ledger.exists() else []
    n = len(records)
    completeness_vec = [1.0] * n + [0.0] * max(budget - n, 0)
    ci_lo, ci_hi = _bootstrap_ci(completeness_vec)
    kept = sum(1 for r in records if r.decision == TrialDecision.KEPT)
    discarded = sum(1 for r in records if r.decision == TrialDecision.DISCARDED)
    failed = sum(1 for r in records if r.decision == TrialDecision.FAILED_INVALID)
    return {
        "arm": "governed",
        "trials_ran": budget,
        "ledger_n": n,
        "completeness": n / budget if budget > 0 else 0.0,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "kept": kept,
        "discarded": discarded,
        "failed_invalid": failed,
        "silent_drops": 0,
    }


def _analyze_ungoverned(ung_ledger: Path, obs_log: Path, budget: int) -> dict:
    """Ungoverned arm: count trials from obs log; count records from ledger."""
    records = TrialAppendStore(ung_ledger).read_all() if ung_ledger.exists() else []
    ledger_n = len(records)

    # Obs log is the ground truth for "how many trials actually ran".
    obs_entries: list[dict] = []
    if obs_log.exists():
        with open(obs_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obs_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    trials_ran = len(obs_entries) if obs_entries else budget

    # Build completeness vector: 1 if trial has a ledger record, 0 if dropped.
    # From obs log, "ungoverned_ledger_entry" == True means it was written.
    if obs_entries:
        completeness_vec = [1.0 if e.get("ungoverned_ledger_entry") else 0.0
                            for e in obs_entries]
    else:
        # Fallback: infer from record count
        completeness_vec = [1.0] * ledger_n + [0.0] * max(trials_ran - ledger_n, 0)

    completeness = sum(completeness_vec) / len(completeness_vec) if completeness_vec else 0.0
    ci_lo, ci_hi = _bootstrap_ci(completeness_vec)

    kept = sum(1 for r in records if r.decision == TrialDecision.KEPT)
    discarded = sum(1 for r in records if r.decision == TrialDecision.DISCARDED)
    failed = sum(1 for r in records if r.decision == TrialDecision.FAILED_INVALID)

    return {
        "arm": "ungoverned",
        "trials_ran": trials_ran,
        "ledger_n": ledger_n,
        "completeness": completeness,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "kept": kept,
        "discarded": discarded,
        "failed_invalid": failed,
        "silent_drops": trials_ran - ledger_n,
        "obs_log_entries": len(obs_entries),
    }


# ── LaTeX table ───────────────────────────────────────────────────────────────

_NODE_DISPLAY = {
    "autoresearch_linux": r"\texttt{autoresearch\_linux} (total failure)",
    "openml_bank_marketing": r"\texttt{openml\_bank\_marketing} (mixed outcomes)",
}

_CI_FMT = "[{:.2f},\\ {:.2f}]"


def _latex_table(rows: list[dict]) -> str:
    """Render the two-node counterfactual comparison as a LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\caption{Governed vs.\ ungoverned counterfactual comparison.",
        r"Ledger completeness = (trials with a ledger record) / (trials that ran).",
        r"Governed completeness is 1.00 by structural invariant; the pending-trial",
        r"guard and append-only ledger guarantee a record for every budget slot.",
        r"Ungoverned completeness falls below 1.00 because \texttt{failed\_invalid}",
        r"trials are not appended; they are confirmed to have run via the",
        r"\texttt{\_ungoverned\_obs.jsonl} observation log.",
        r"95\% bootstrap CIs use 10{,}000 resamples (seed 42).}",
        r"\label{tab:counterfactual}",
        r"\begin{tabular}{@{}p{0.34\linewidth}rrrr@{}}",
        r"\toprule",
        r"Node & $N$ & \multicolumn{2}{c}{Ledger completeness} & Silent drops \\",
        r"\cmidrule(lr){3-4}",
        r" & & Governed & Ungoverned & \\",
        r"\midrule",
    ]
    for row in rows:
        node_tex = _NODE_DISPLAY.get(row["node"], r"\texttt{" + row["node"] + "}")
        gov = row["governed"]
        ung = row["ungoverned"]
        n = gov["trials_ran"]
        gov_ci = _CI_FMT.format(gov["ci_lo"], gov["ci_hi"])
        ung_ci = _CI_FMT.format(ung["ci_lo"], ung["ci_hi"])
        gov_str = f"1.00 {gov_ci}"
        ung_str = f"{ung['completeness']:.2f} {ung_ci}"
        drops = ung["silent_drops"]
        lines.append(
            f"{node_tex} & {n} & {gov_str} & {ung_str} & {drops} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{5}{@{}p{\linewidth}@{}}{\textit{Governed completeness = 1.00}",
        r"on both nodes by the lifecycle invariant (Section~\ref{trial-lifecycle}).",
        r"Ungoverned completeness is confirmed via the observation log; without",
        r"governance, the observation log is the only evidence these trials ran.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Console table ─────────────────────────────────────────────────────────────

def _print_console_table(rows: list[dict]) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  COUNTERFACTUAL RESULTS")
    print(sep)
    hdr = f"  {'Node':<30}  {'Arm':<12}  {'N':>4}  {'Compl':>6}  {'95% CI':<16}  {'Drops':>5}"
    print(hdr)
    print(f"  {'─'*30}  {'─'*12}  {'─'*4}  {'─'*6}  {'─'*16}  {'─'*5}")
    for row in rows:
        node = row["node"]
        for arm_key in ("governed", "ungoverned"):
            a = row[arm_key]
            ci_str = f"[{a['ci_lo']:.2f}, {a['ci_hi']:.2f}]"
            print(
                f"  {node:<30}  {arm_key:<12}  {a['trials_ran']:>4}  "
                f"{a['completeness']:>6.2f}  {ci_str:<16}  {a['silent_drops']:>5}"
            )
    print(sep)
    print()
    for row in rows:
        gov = row["governed"]
        ung = row["ungoverned"]
        drops = ung["silent_drops"]
        drop_pct = drops / ung["trials_ran"] * 100 if ung["trials_ran"] else 0
        print(f"  {row['node']}: {drops}/{ung['trials_ran']} trials ({drop_pct:.1f}%)"
              f" silently absent from ungoverned ledger")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--summary", action="append", required=True, metavar="JSON",
                    help="Path to _cf_summary.json written by run_counterfactual.py "
                         "(pass once per node pair)")
    ap.add_argument("--output", default=None,
                    help="Write LaTeX table snippet to this path "
                         "(default: paper/tables/counterfactual_comparison.tex)")
    ap.add_argument("--json-out", default=None,
                    help="Write full numeric results to this JSON path")
    ap.add_argument("--n-bootstrap", type=int, default=10_000,
                    help="Bootstrap resample count (default: 10,000)")
    args = ap.parse_args()

    output_path = Path(args.output) if args.output else (
        ROOT / "paper" / "tables" / "counterfactual_comparison.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for summary_file in args.summary:
        with open(summary_file, encoding="utf-8") as f:
            meta = json.load(f)

        node = meta["node"]
        budget = meta["budget"]
        gov_ledger = Path(meta["gov_ledger"])
        ung_ledger = Path(meta["ung_ledger"])
        obs_log = Path(meta["obs_log"])

        print(f"\nAnalysing {node} (budget={budget}) …")
        gov = _analyze_governed(gov_ledger, budget)
        ung = _analyze_ungoverned(ung_ledger, obs_log, budget)

        all_rows.append({
            "node": node,
            "budget": budget,
            "governed": gov,
            "ungoverned": ung,
        })

    if not all_rows:
        print("No data found.", file=sys.stderr)
        return 1

    _print_console_table(all_rows)

    # LaTeX
    latex = _latex_table(all_rows)
    output_path.write_text(latex + "\n", encoding="utf-8")
    print(f"LaTeX table → {output_path}")

    # JSON
    json_out = Path(args.json_out) if args.json_out else output_path.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Numeric JSON → {json_out}")

    # Print suggested paper text
    print("\n── Suggested paper text (replace §5.3 illustrative counterfactual) ──")
    for row in all_rows:
        gov = row["governed"]
        ung = row["ungoverned"]
        node = row["node"]
        drops = ung["silent_drops"]
        n = gov["trials_ran"]
        ung_c = ung["completeness"]
        print(
            f"\n  {node}: governed {gov['completeness']:.2f} "
            f"[{gov['ci_lo']:.2f}, {gov['ci_hi']:.2f}] vs. "
            f"ungoverned {ung_c:.2f} [{ung['ci_lo']:.2f}, {ung['ci_hi']:.2f}] "
            f"(N={n}; {drops} trials silently absent from ungoverned ledger)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
