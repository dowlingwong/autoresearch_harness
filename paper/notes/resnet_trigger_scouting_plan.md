# Research Scouting Plan: resnet_trigger

Context: `/Users/wongdowling/Documents/autoresearch_harness/paper/notes/resnet_trigger_research_context.md`
Context SHA-256: `c40c6ee5b30bf70e9e3590fe6090c616a02f19571ec1ce22212e7f5d370fe2bf`

## Breadth Pass

- Enumerate 5-8 independent hypothesis families before selecting concrete trials.
- Require each family to name the editable symbol, expected metric movement, and risk.
- Reject families that touch frozen data, dependencies, or evaluation code.

## Depth Pass

- For the top 2 families, produce bounded structured edits only.
- Check the current effective config before proposing each edit.
- Stop a family after two degraded or invalid outcomes unless new evidence changes the premise.

## Apply Gate

- Promote only kept trials with complete provenance, non-empty patch hash, and changed effective config.
- Keep discarded and invalid trials in the ledger for audit, but do not apply them to master state.
