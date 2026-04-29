# Experiment Protocol

Stage 2 uses fixed-budget campaigns.

Protocol:

1. Load node spec.
2. Load manager and memory mode.
3. For each budget unit, request one bounded proposal.
4. Run worker under editable-path constraints.
5. Validate changed files.
6. Parse node metric.
7. Apply keep/discard/failed-invalid decision.
8. Append one trial record.
9. Export campaign and governance metrics.

Memory ablation modes:

- `none`
- `append_only_summary`
- `append_only_summary_with_rationale`

