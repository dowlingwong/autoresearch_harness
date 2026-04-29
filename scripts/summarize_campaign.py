#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.reporting.export_tables import export_campaign_tables


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a Stage 2 campaign trial ledger.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--records")
    parser.add_argument("--metric-name", default="val_auc")
    parser.add_argument("--metric-direction", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--out-dir", default=str(ROOT / "paper" / "tables"))
    args = parser.parse_args()

    records_path = Path(args.records) if args.records else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    summary = load_campaign_summary(records_path, metric_name=args.metric_name, metric_direction=args.metric_direction)
    outputs = export_campaign_tables(summary, args.out_dir)
    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "records": str(records_path),
                "metrics": summary.metrics.to_dict(),
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

