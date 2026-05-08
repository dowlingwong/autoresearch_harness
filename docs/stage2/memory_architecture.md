# Memory Architecture

Memory has two roles:

- auditability: preserve raw append-only trial records
- manager context: provide compressed summaries to guide proposals

Raw records are JSONL ledgers. Manager context is produced by memory mode:

- `none`: node and budget context only
- `append_only_summary`: prior proposal and result summaries
- `append_only_summary_with_rationale`: summaries plus rationale, failures, and repeated-bad warnings

