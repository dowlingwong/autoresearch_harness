# Memory Index

This directory is the synthesized project memory layer for `autoresearch-macos`.

It sits between:

- raw operational records such as [`results.tsv`](../results.tsv) and [`experiment_memory.jsonl`](../experiment_memory.jsonl)
- future higher-level retrieval or wiki tooling

The goal is not to replace exact run logs. The goal is to keep a compact, maintained set of topic pages that accumulate what the manager and worker learn over time.

## Topics

- [Current strategy](./project_current_strategy.md) - Current operating model, control-plane shape, and near-term execution priorities.
- [Failed ideas](./project_failed_ideas.md) - Approaches or patterns that did not work, or that should be treated cautiously.
- [Promising ideas](./project_promising_ideas.md) - Directions worth revisiting or expanding in future experiment cycles.
- [External references](./reference_external_sources.md) - Pointers to papers, repos, dashboards, or notes that are useful but live outside the codebase.

## Memory policy

- Update existing topic files before creating new ones.
- Prefer synthesized conclusions over dumping raw logs.
- Treat concrete file/function/flag references as historical claims and re-verify before acting on them.
- Keep the exact experiment ledger in `results.tsv`; keep the append-only event log in `experiment_memory.jsonl`.
- Use this directory for cross-run synthesis, strategy, and retrieval-friendly summaries.
