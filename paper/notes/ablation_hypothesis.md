# Memory Ablation Hypothesis

Pre-stated before running Experiment B (per scientific practice).

We hypothesise that:

```text
repeated_bad_rate(none) > repeated_bad_rate(append_only_summary)
                        > repeated_bad_rate(append_only_summary_with_rationale)
```

Grounded in:
- OpenAI (2026): "give the agent a map, not a manual" - structured failure context outperforms raw history.
- Bockeler (2026): feedforward guides (rationale) + feedback sensors (repeated-bad detector) together reduce recurrence more than feedback alone.

If the result is flat or reversed, it is a valid negative finding and will be reported as such.
