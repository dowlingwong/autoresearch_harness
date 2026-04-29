# Architecture

The Stage 2 system is a governed autonomous experimentation framework.

Core layers:

1. Manager proposes bounded experiment changes.
2. Control plane owns lifecycle, budget, decisions, and records.
3. Worker runtime applies bounded code changes and returns structured results.
4. Experiment node declares editable paths, commands, metrics, and validity rules.
5. Memory/audit layer stores append-only trial records and compressed manager context.
6. Evaluation/reporting layer exports deterministic paper metrics and CSVs.

The core contribution is the control plane, not the worker backend.

