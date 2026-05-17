#!/usr/bin/env python3
"""Bootstrap confidence intervals for governed campaign metrics.

This script is intentionally ledger-only: it does not re-run campaigns. Use it
to quantify uncertainty in existing short-budget evidence, and use the same
interface after larger P0 reruns land.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.evaluation.metrics import compute_campaign_metrics
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialRecord
from autoresearch.nodes.registry import load_registered_node


METRICS = (
    "acceptance_rate",
    "invalid_rate",
    "complete_provenance_rate",
    "artifact_capture_completeness",
    "best_metric",
    "net_gain",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CIs for governance metrics from campaign ledgers.",
    )
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        help=(
            "Campaign id or path to *_trials.jsonl. Repeat for multiple campaigns. "
            "Campaign ids are resolved under experiments/ledgers."
        ),
    )
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        help=(
            "Registered node name matching --campaign order, or one node reused for all campaigns."
        ),
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--out",
        default=str(ROOT / "paper" / "tables" / "governance_bootstrap_cis.csv"),
    )
    args = parser.parse_args()

    nodes = args.node
    if len(nodes) not in {1, len(args.campaign)}:
        parser.error("--node must be provided once or once per --campaign")

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    for index, campaign in enumerate(args.campaign):
        node_name = nodes[0] if len(nodes) == 1 else nodes[index]
        node_spec = load_registered_node(node_name, repo_root=ROOT)
        records_path = _records_path(campaign)
        records = TrialAppendStore(records_path).read_all()
        point = compute_campaign_metrics(
            records,
            metric_name=node_spec.metric_name,
            metric_direction=node_spec.metric_direction,  # type: ignore[arg-type]
        )
        cis = _bootstrap_cis(
            records,
            metric_name=node_spec.metric_name,
            metric_direction=node_spec.metric_direction,
            samples=args.samples,
            rng=rng,
        )
        for metric in METRICS:
            low, high = cis.get(metric, (None, None))
            rows.append(
                {
                    "campaign_id": point.campaign_id or _campaign_id(records_path),
                    "node_id": node_spec.name,
                    "metric": metric,
                    "estimate": getattr(point, metric),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_samples": args.samples,
                    "n_trials": len(records),
                    "note": "trial-level bootstrap; use seed-level bootstrap when replicated seeds are available",
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


def _records_path(campaign: str) -> Path:
    path = Path(campaign)
    if path.exists():
        return path
    candidate = ROOT / "experiments" / "ledgers" / f"{campaign}_trials.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"could not resolve campaign ledger: {campaign}")


def _campaign_id(records_path: Path) -> str:
    stem = records_path.stem
    return stem[: -len("_trials")] if stem.endswith("_trials") else stem


def _bootstrap_cis(
    records: list[TrialRecord],
    *,
    metric_name: str,
    metric_direction: str,
    samples: int,
    rng: random.Random,
) -> dict[str, tuple[float | None, float | None]]:
    if not records:
        return {metric: (None, None) for metric in METRICS}
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    n = len(records)
    for _ in range(samples):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        metrics = compute_campaign_metrics(
            sample,
            metric_name=metric_name,
            metric_direction=metric_direction,  # type: ignore[arg-type]
        )
        for metric in METRICS:
            value = getattr(metrics, metric)
            if value is not None:
                values[metric].append(float(value))
    return {metric: _percentile_ci(vals) for metric, vals in values.items()}


def _percentile_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    values = sorted(values)
    return (_percentile(values, 0.025), _percentile(values, 0.975))


def _percentile(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = p * (len(values) - 1)
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    weight = pos - low
    return values[low] * (1.0 - weight) + values[high] * weight


if __name__ == "__main__":
    raise SystemExit(main())
