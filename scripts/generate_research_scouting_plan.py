#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a breadth/depth scouting plan from node context.")
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--context",
        help="Research context markdown. Defaults to paper/notes/<node>_research_context.md.",
    )
    parser.add_argument(
        "--output",
        help="Output markdown. Defaults to paper/notes/<node>_scouting_plan.md.",
    )
    args = parser.parse_args()

    context = Path(args.context) if args.context else ROOT / "paper" / "notes" / f"{args.node}_research_context.md"
    output = Path(args.output) if args.output else ROOT / "paper" / "notes" / f"{args.node}_scouting_plan.md"
    context_sha = _sha256(context) if context.exists() else "missing"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"# Research Scouting Plan: {args.node}",
                "",
                f"Context: `{context}`",
                f"Context SHA-256: `{context_sha}`",
                "",
                "## Breadth Pass",
                "",
                "- Enumerate 5-8 independent hypothesis families before selecting concrete trials.",
                "- Require each family to name the editable symbol, expected metric movement, and risk.",
                "- Reject families that touch frozen data, dependencies, or evaluation code.",
                "",
                "## Depth Pass",
                "",
                "- For the top 2 families, produce bounded structured edits only.",
                "- Check the current effective config before proposing each edit.",
                "- Stop a family after two degraded or invalid outcomes unless new evidence changes the premise.",
                "",
                "## Apply Gate",
                "",
                "- Promote only kept trials with complete provenance, non-empty patch hash, and changed effective config.",
                "- Keep discarded and invalid trials in the ledger for audit, but do not apply them to master state.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote={output}")
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
