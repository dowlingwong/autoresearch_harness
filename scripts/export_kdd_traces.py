#!/usr/bin/env python3
"""Export per-campaign trace files for the KDD AAE 2026 submission.

Each trace file is a JSONL where every line is one trial's complete audit
record: ordered lifecycle events joined with the final ledger record. A
campaign-level summary line is prepended.

Absolute paths are rebased relative to the repository root so traces are
portable across machines. Strings matching secret-like patterns (API keys,
tokens) are redacted.

Usage
-----
  # Export all 6 evidence campaigns (default):
  python3 scripts/export_kdd_traces.py

  # Export specific campaigns:
  python3 scripts/export_kdd_traces.py \\
      --campaigns kdd_main_5trial kdd_stress_scope kdd_stress_noop

  # Custom output directory:
  python3 scripts/export_kdd_traces.py --output-dir experiments/traces/

Output
------
  experiments/traces/{campaign_id}.jsonl   — one file per campaign

Each line in the output file has the shape:

  {
    "record_type": "campaign_summary" | "trial_trace",
    "campaign_id": "...",
    "trial_id": "..." | null,
    "budget_index": int | null,
    "decision": "..." | null,
    "validity_status": "..." | null,
    "events": [ <ordered lifecycle events> ],
    "ledger_record": { <full trial record, paths rebased and redacted> }
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LEDGERS_DIR = ROOT / "experiments" / "ledgers"
EVENTS_DIR = ROOT / "experiments" / "events"
TRACES_DIR = ROOT / "experiments" / "traces"

# Campaigns included in the KDD 2026 submission by default
DEFAULT_CAMPAIGNS = [
    "kdd_main_5trial",
    "ablation_none",
    "ablation_append_only_summary",
    "ablation_append_only_summary_with_rationale",
    "kdd_stress_scope",
    "kdd_stress_noop",
]

# Regex patterns whose values are replaced with <REDACTED>
_SECRET_PATTERNS = [
    re.compile(r'sk-[A-Za-z0-9]{20,}'),          # OpenAI-style keys
    re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*'), # Bearer tokens
    re.compile(r'api[_\-]?key["\s]*[:=]["\s]*\S+', re.IGNORECASE),
    re.compile(r'token["\s]*[:=]["\s]*[A-Za-z0-9\-._]{16,}', re.IGNORECASE),
    re.compile(r'password["\s]*[:=]["\s]*\S+', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Path rebasing
# ---------------------------------------------------------------------------

def _rebase_path(value: str) -> str:
    """Convert an absolute path to a repo-relative path where possible."""
    p = Path(value)
    if not p.is_absolute():
        return value
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        pass
    # Try to find the repo root directory name inside the path and rebase
    repo_name = ROOT.name
    parts = p.parts
    try:
        idx = parts.index(repo_name)
        return str(Path(*parts[idx + 1:]))
    except ValueError:
        return value


def _redact_secrets(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("<REDACTED>", value)
    return value


def _clean_string(value: str) -> str:
    """Rebase paths and redact secrets in a string value."""
    value = _redact_secrets(value)
    # Rebase if it looks like an absolute path
    if value.startswith("/") or (len(value) > 2 and value[1] == ":"):
        value = _rebase_path(value)
    return value


def _clean_value(obj: Any) -> Any:
    """Recursively clean an object: rebase paths, redact secrets."""
    if isinstance(obj, str):
        return _clean_string(obj)
    if isinstance(obj, dict):
        return {k: _clean_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_value(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] JSON error at {path.name}:{lineno}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def build_campaign_trace(
    campaign_id: str,
    ledger_dir: Path = LEDGERS_DIR,
    events_dir: Path = EVENTS_DIR,
) -> list[dict]:
    """Return a list of trace records for one campaign.

    The first record is always a ``campaign_summary``; the remaining records
    are ``trial_trace`` entries in budget-index order.
    """
    ledger_path = ledger_dir / f"{campaign_id}_trials.jsonl"
    events_path = events_dir / f"{campaign_id}_events.jsonl"

    trials = _read_jsonl(ledger_path)
    events = _read_jsonl(events_path)

    # Index trial records by trial_id
    trial_index: dict[str, dict] = {r["trial_id"]: r for r in trials if "trial_id" in r}

    # Partition events into campaign-level and per-trial buckets
    campaign_events: list[dict] = []
    trial_events: dict[str, list[dict]] = {}
    for event in events:
        tid = event.get("trial_id")
        if tid:
            trial_events.setdefault(tid, []).append(event)
        else:
            campaign_events.append(event)

    # Campaign summary line
    campaign_summary_event = next(
        (e for e in campaign_events if e.get("event_type") == "campaign_started"), {}
    )
    campaign_completed_event = next(
        (e for e in campaign_events if e.get("event_type") == "campaign_completed"), {}
    )
    output: list[dict] = [
        {
            "record_type": "campaign_summary",
            "campaign_id": campaign_id,
            "trial_id": None,
            "budget_index": None,
            "decision": None,
            "validity_status": None,
            "total_trials": len(trials),
            "events": _clean_value(campaign_events),
            "campaign_started_payload": _clean_value(campaign_summary_event.get("payload", {})),
            "campaign_completed_payload": _clean_value(campaign_completed_event.get("payload", {})),
            "ledger_record": None,
        }
    ]

    # One trial_trace per ledger record, in budget_index order
    sorted_trials = sorted(trials, key=lambda r: int(r.get("budget_index") or 0))
    for record in sorted_trials:
        tid = record.get("trial_id", "")
        ordered_events = trial_events.get(tid, [])
        # Sort by timestamp then event_id for determinism
        ordered_events = sorted(
            ordered_events,
            key=lambda e: (e.get("timestamp", ""), e.get("event_id", "")),
        )
        output.append(
            {
                "record_type": "trial_trace",
                "campaign_id": campaign_id,
                "trial_id": tid,
                "budget_index": record.get("budget_index"),
                "decision": record.get("decision"),
                "validity_status": record.get("validity_status"),
                "failure_category": record.get("failure_category"),
                "proposal_summary": record.get("proposal_summary"),
                "events": _clean_value(ordered_events),
                "ledger_record": _clean_value(record),
            }
        )

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export per-campaign trace JSONL files for KDD AAE 2026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--campaigns",
        nargs="+",
        default=DEFAULT_CAMPAIGNS,
        help="Campaign IDs to export (default: all 6 KDD evidence campaigns).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TRACES_DIR),
        help=f"Directory to write trace files (default: {TRACES_DIR}).",
    )
    parser.add_argument(
        "--ledger-dir",
        default=str(LEDGERS_DIR),
        help="Directory containing *_trials.jsonl ledger files.",
    )
    parser.add_argument(
        "--events-dir",
        default=str(EVENTS_DIR),
        help="Directory containing *_events.jsonl event stream files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = Path(args.ledger_dir)
    events_dir = Path(args.events_dir)

    results: dict[str, dict] = {}
    all_ok = True

    for campaign_id in args.campaigns:
        ledger_path = ledger_dir / f"{campaign_id}_trials.jsonl"
        events_path = events_dir / f"{campaign_id}_events.jsonl"

        missing = []
        if not ledger_path.exists():
            missing.append(f"ledger ({ledger_path.name})")
        if not events_path.exists():
            missing.append(f"events ({events_path.name})")

        if missing:
            print(f"  [warn] {campaign_id}: missing {', '.join(missing)} — skipping", file=sys.stderr)
            all_ok = False
            continue

        print(f"Exporting {campaign_id} ...", end=" ", flush=True)
        trace_records = build_campaign_trace(campaign_id, ledger_dir, events_dir)

        out_path = out_dir / f"{campaign_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for record in trace_records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

        trial_count = sum(1 for r in trace_records if r["record_type"] == "trial_trace")
        event_count = sum(len(r["events"]) for r in trace_records)
        size_kb = out_path.stat().st_size / 1024
        print(f"{trial_count} trials, {event_count} events → {out_path.name} ({size_kb:.1f} KB)")
        results[campaign_id] = {
            "path": str(out_path.relative_to(ROOT)),
            "trials": trial_count,
            "events": event_count,
            "size_kb": round(size_kb, 1),
        }

    print()
    print(json.dumps(results, indent=2, sort_keys=True))
    _verify(results, out_dir)
    return 0 if all_ok else 1


def _verify(results: dict, out_dir: Path) -> None:
    print("\nVerification:")
    for campaign_id, info in results.items():
        path = out_dir / f"{campaign_id}.jsonl"
        ok = path.exists() and path.stat().st_size > 0
        print(f"  {'✅' if ok else '❌'} {campaign_id}: {info['trials']} trials, "
              f"{info['events']} events, {info['size_kb']} KB")
    print()
    print("Trace files written. Absolute paths have been rebased to repo-relative.")
    print("Secret-like strings (API keys, tokens, passwords) are redacted.")


if __name__ == "__main__":
    raise SystemExit(main())
