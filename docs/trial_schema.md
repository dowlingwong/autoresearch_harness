# Trial Schema

Every Stage 2 trial is represented by an append-only `TrialRecord`.

Required groups:

- identity: trial id, campaign id, node id, budget index
- modes: manager, worker, memory
- proposal: summary, rationale, targeted files
- provenance: patch, run, metric, decision ids
- execution: status, validity, failure category, logs
- metrics: parsed metrics, current best, delta
- decision: kept, discarded, or failed-invalid

Records are appended to JSONL ledgers and should not be overwritten.

