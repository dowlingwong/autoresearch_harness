#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for governed campaign metrics.

Standalone — reads JSONL ledgers directly with no harness import.
Supports both trial-level bootstrap (per campaign) and seed-level bootstrap
(pooling campaigns that share the same node + arm + budget).

Usage
-----
    python3 scripts/compute_bootstrap_cis.py

Outputs
-------
    paper/tables/bootstrap_cis_trial_level.csv   — per-campaign trial bootstrap
    paper/tables/bootstrap_cis_seed_level.csv    — per (node, arm, budget) seed bootstrap
    paper/tables/bootstrap_cis_summary.csv       — human-readable summary for paper
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGERS = ROOT / "experiments" / "ledgers"
OUT_DIR = ROOT / "paper" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_SAMPLES = 10_000
SEED = 42
CI_LEVEL = 0.95
CI_LO = (1 - CI_LEVEL) / 2          # 0.025
CI_HI = 1 - (1 - CI_LEVEL) / 2     # 0.975

# ---------------------------------------------------------------------------
# Campaign registry — maps (node, arm, budget_tag, seed) → campaign_id
# ---------------------------------------------------------------------------
# Budget tags: "b10" for original b10 campaigns, "b30" for phase-E reruns.
# Each entry is (campaign_id_pattern, node, arm_key, budget_tag)
# arm_key short forms: "none", "summary", "rationale"

CAMPAIGN_GROUPS = {
    "lr_synthetic": {
        "metric": "val_score",
        "direction": "maximize",
        "arms": {
            "none":      "none",
            "summary":   "append_only_summary",
            "rationale": "append_only_summary_with_rationale",
        },
        "budgets": {
            "b10": "deepseek_lr_{arm}_s{seed}",
            "b30": "deepseek_lr_{arm}_b30_s{seed}",
        },
        "seeds": [1, 2, 3, 4, 5],
    },
    "mlp_synthetic": {
        "metric": "val_score",
        "direction": "maximize",
        "arms": {
            "none":      "none",
            "summary":   "summary",
            "rationale": "rationale",
        },
        "budgets": {
            "b10": "deepseek_mlp_{arm}_s{seed}",
            "b30": "deepseek_mlp_{arm}_b30_s{seed}",
        },
        "seeds_b10": [1],   # only s1 at b10
        "seeds": [1, 2, 3],
    },
    "openml_credit_g": {
        "metric": "val_auc",
        "direction": "maximize",
        "arms": {"summary": "append_only_summary"},
        "budgets": {
            "b20": "deepseek_openml_cg_s{seed}",
            "b30": "deepseek_openml_cg_b30_s{seed}",
        },
        "seeds_b20": [1, 2, 3, 4],
        "seeds": [1, 2, 3, 4, 5],
    },
    "openml_bank_marketing": {
        "metric": "val_auc",
        "direction": "maximize",
        "arms": {"summary": "append_only_summary"},
        "budgets": {
            "b20": "deepseek_openml_bm_s{seed}",
            "b30": "deepseek_openml_bm_b30_s{seed}",
        },
        "seeds_b20": [1, 2, 3, 4],
        "seeds": [1, 2, 3, 4, 5],
    },
    "resnet_trigger": {
        "metric": "val_auc",
        "direction": "maximize",
        "arms": {
            "none":      "none",
            "summary":   "append_only_summary",
            "rationale": "append_only_summary_with_rationale",
        },
        "budgets": {
            "b15": "deepseek_resnet_{arm}_s{seed}",
        },
        "seeds": [1, 2, 3],
    },
}


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def read_ledger(campaign_id: str) -> list[dict]:
    path = LEDGERS / f"{campaign_id}_trials.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Per-trial metric extraction
# ---------------------------------------------------------------------------

def trial_metrics(records: list[dict], metric_key: str) -> dict:
    """Extract scalar trial-level features for bootstrapping."""
    n = len(records)
    if n == 0:
        return {}
    kept = [r for r in records if r.get("decision") == "kept"]
    invalid = [r for r in records if r.get("decision") == "failed_invalid"]
    discarded = [r for r in records if r.get("decision") == "discarded"]

    def has_provenance(r):
        p = r.get("provenance") or {}
        return all(p.get(k) for k in ("proposal_id", "patch_id", "run_id", "metric_id", "decision_id"))

    metric_values = []
    for r in records:
        pm = r.get("parsed_metrics") or {}
        v = pm.get(metric_key)
        if v is not None:
            metric_values.append(float(v))

    best_metric = max(metric_values) if metric_values else None
    initial = metric_values[0] if metric_values else None

    return {
        "n": n,
        "acceptance_rate": len(kept) / n,
        "invalid_rate": len(invalid) / n,
        "discard_rate": len(discarded) / n,
        "provenance_completeness": sum(has_provenance(r) for r in records) / n,
        "best_metric": best_metric,
        "net_gain": (best_metric - initial) if (best_metric is not None and initial is not None) else None,
    }


# ---------------------------------------------------------------------------
# Trial-level bootstrap (resample trials within a single campaign)
# ---------------------------------------------------------------------------

def bootstrap_trial_level(
    records: list[dict],
    metric_key: str,
    n_samples: int,
    rng: random.Random,
) -> dict[str, tuple[float, float]]:
    """Returns {metric_name: (ci_low, ci_high)} at 95% from trial resampling."""
    n = len(records)
    if n < 2:
        return {}

    boot: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_samples):
        sample = [records[rng.randint(0, n - 1)] for _ in range(n)]
        m = trial_metrics(sample, metric_key)
        for k, v in m.items():
            if k != "n" and v is not None:
                boot[k].append(v)

    cis = {}
    for k, vals in boot.items():
        vals_sorted = sorted(vals)
        lo = vals_sorted[int(CI_LO * len(vals_sorted))]
        hi = vals_sorted[int(CI_HI * len(vals_sorted))]
        cis[k] = (lo, hi)
    return cis


# ---------------------------------------------------------------------------
# Seed-level bootstrap (resample seed best-metrics across seeds)
# ---------------------------------------------------------------------------

def bootstrap_seed_level(
    seed_bests: list[float],
    n_samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Bootstrap CI over per-seed best metrics."""
    n = len(seed_bests)
    if n < 2:
        return (seed_bests[0], seed_bests[0]) if seed_bests else (float("nan"), float("nan"))
    boot_means = []
    for _ in range(n_samples):
        sample = [seed_bests[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo = boot_means[int(CI_LO * len(boot_means))]
    hi = boot_means[int(CI_HI * len(boot_means))]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)

    trial_rows = []
    seed_rows = []
    summary_rows = []

    for node_name, cfg in CAMPAIGN_GROUPS.items():
        metric_key = cfg["metric"]
        direction = cfg["direction"]
        arms = cfg["arms"]        # {short_name: full_arm_string}
        budgets = cfg["budgets"]  # {budget_tag: pattern}

        for budget_tag, pattern in budgets.items():
            seeds = cfg.get(f"seeds_{budget_tag}", cfg.get("seeds", []))

            for arm_short, arm_full in arms.items():
                seed_best_metrics = []
                seed_acceptance_rates = []
                seed_invalid_rates = []
                campaigns_found = []

                for seed_n in seeds:
                    cid = pattern.format(arm=arm_full, seed=seed_n)
                    records = read_ledger(cid)
                    if not records:
                        continue
                    campaigns_found.append(cid)

                    # Trial-level bootstrap
                    point = trial_metrics(records, metric_key)
                    cis = bootstrap_trial_level(records, metric_key, BOOTSTRAP_SAMPLES, rng)

                    trial_rows.append({
                        "node": node_name,
                        "arm": arm_short,
                        "budget": budget_tag,
                        "seed": f"s{seed_n}",
                        "campaign_id": cid,
                        "n_trials": point.get("n", 0),
                        "acceptance_rate": round(point.get("acceptance_rate", float("nan")), 4),
                        "ar_ci_lo": round(cis.get("acceptance_rate", (float("nan"),))[0], 4),
                        "ar_ci_hi": round(cis.get("acceptance_rate", (float("nan"), float("nan")))[1], 4),
                        "invalid_rate": round(point.get("invalid_rate", float("nan")), 4),
                        "ir_ci_lo": round(cis.get("invalid_rate", (float("nan"),))[0], 4),
                        "ir_ci_hi": round(cis.get("invalid_rate", (float("nan"), float("nan")))[1], 4),
                        "best_metric": round(point.get("best_metric") or float("nan"), 5),
                        "bm_ci_lo": round(cis.get("best_metric", (float("nan"),))[0], 5),
                        "bm_ci_hi": round(cis.get("best_metric", (float("nan"), float("nan")))[1], 5),
                        "provenance_completeness": round(point.get("provenance_completeness", float("nan")), 4),
                        "direction": direction,
                    })

                    if point.get("best_metric") is not None:
                        seed_best_metrics.append(point["best_metric"])
                    seed_acceptance_rates.append(point.get("acceptance_rate", float("nan")))
                    seed_invalid_rates.append(point.get("invalid_rate", float("nan")))

                # Seed-level bootstrap
                if seed_best_metrics:
                    mean_bm = statistics.mean(seed_best_metrics)
                    sd_bm = statistics.stdev(seed_best_metrics) if len(seed_best_metrics) > 1 else 0.0
                    ci_lo, ci_hi = bootstrap_seed_level(seed_best_metrics, BOOTSTRAP_SAMPLES, rng)
                    mean_ar = statistics.mean(seed_acceptance_rates)
                    mean_ir = statistics.mean(seed_invalid_rates)

                    seed_rows.append({
                        "node": node_name,
                        "arm": arm_short,
                        "budget": budget_tag,
                        "n_seeds": len(seed_best_metrics),
                        "n_trials_total": sum(trial_rows[i]["n_trials"] for i in range(len(trial_rows))
                                             if trial_rows[i]["node"] == node_name
                                             and trial_rows[i]["arm"] == arm_short
                                             and trial_rows[i]["budget"] == budget_tag),
                        "mean_best_metric": round(mean_bm, 5),
                        "sd_best_metric": round(sd_bm, 5),
                        "seed_ci_lo": round(ci_lo, 5),
                        "seed_ci_hi": round(ci_hi, 5),
                        "mean_acceptance_rate": round(mean_ar, 4),
                        "mean_invalid_rate": round(mean_ir, 4),
                        "seed_best_metrics": " | ".join(f"{v:.5f}" for v in seed_best_metrics),
                        "direction": direction,
                        "campaigns": " | ".join(campaigns_found),
                    })

                    # Paper-facing summary row
                    ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                    summary_rows.append({
                        "node": node_name,
                        "arm": arm_short,
                        "budget": budget_tag,
                        "n_seeds": len(seed_best_metrics),
                        "mean_best_metric": f"{mean_bm:.4f}",
                        "95pct_CI": ci_str,
                        "sd": f"{sd_bm:.4f}",
                        "mean_AR%": f"{mean_ar*100:.1f}",
                        "mean_IR%": f"{mean_ir*100:.1f}",
                        "note": ("CI too wide — add seeds" if (ci_hi - ci_lo) > 0.05
                                 else "CI acceptable"),
                    })

                    print(f"  {node_name:25s}  {arm_short:12s}  {budget_tag:4s}  "
                          f"n_seeds={len(seed_best_metrics)}  "
                          f"mean={mean_bm:.4f}  sd={sd_bm:.4f}  "
                          f"95%CI={ci_str}")

    # Write outputs
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows → {path}")

    write_csv(OUT_DIR / "bootstrap_cis_trial_level.csv", trial_rows)
    write_csv(OUT_DIR / "bootstrap_cis_seed_level.csv", seed_rows)
    write_csv(OUT_DIR / "bootstrap_cis_summary.csv", summary_rows)

    # Print table for paper
    print("\n" + "="*80)
    print("SEED-LEVEL SUMMARY (for paper tables)")
    print("="*80)
    print(f"{'Node':<25} {'Arm':<12} {'Bdgt':<5} {'N':>2} {'Mean':>8} {'95%CI':>22} {'SD':>8} {'AR%':>6} {'IR%':>6}")
    print("-"*80)
    for r in summary_rows:
        print(f"{r['node']:<25} {r['arm']:<12} {r['budget']:<5} {r['n_seeds']:>2} "
              f"{r['mean_best_metric']:>8} {r['95pct_CI']:>22} {r['sd']:>8} "
              f"{r['mean_AR%']:>6} {r['mean_IR%']:>6}")


if __name__ == "__main__":
    main()
