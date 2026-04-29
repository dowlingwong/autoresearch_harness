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
    parser = argparse.ArgumentParser(description="Export deterministic paper table CSVs.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--records")
    parser.add_argument("--out-dir", default=str(ROOT / "paper" / "tables"))
    args = parser.parse_args()
    records = Path(args.records) if args.records else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    summary = load_campaign_summary(records)
    outputs = export_campaign_tables(summary, args.out_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

